import argparse
import base64
import json
import os
import pathlib
import time
import uuid
import requests
from typing import List, Dict, Any
from credentials import CLIENT_SECRET, CLIENT_ID

"""
RVT -> IFC via Autodesk Platform Services (APS)
- Creates (if necessary) a bucket in OSS (EMEA)
- Uploads .rvt via “signed S3 upload” (new stream)
- Runs Model Derivative job in IFC2x3 / IFC4
- Pulls manifest until completion
- Downloads .IFC result
"""

APS_BASE = "https://developer.api.autodesk.com"

CLIENT_ID     = CLIENT_ID
CLIENT_SECRET = CLIENT_SECRET


# ----------------- API Additional -----------------

def _b64_urn(object_id: str) -> str:
    return base64.b64encode(object_id.encode()).decode().rstrip("=")

# ----------------- authorisation -----------------

def get_token(scopes: str = "bucket:create bucket:read data:read data:write") -> str:
    r = requests.post(
        f"{APS_BASE}/authentication/v2/token",
        data={"grant_type": "client_credentials", "scope": scopes},
        auth=(CLIENT_ID, CLIENT_SECRET),
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Auth failed: {r.status_code} {r.text}")
    return r.json()["access_token"]

def _headers_auth(token: str, region: str = "US", extra: Dict[str, str] | None = None) -> Dict[str, str]:
    h = {"Authorization": f"Bearer {token}", "x-ads-region": region}
    if extra:
        h.update(extra)
    return h


# ----------------- OSS Bucket APS(Autodesk Platform Services) -----------------

def ensure_bucket(token: str, bucket_key: str, policy: str = "temporary", region: str = "US") -> bool:
    """
    Creates bucket if it isn't exist
    True - if exists
    False - if not
    """
    header = _headers_auth(token, region, {"Content-Type": "application/json"})
    r = requests.post(
        f"{APS_BASE}/oss/v2/buckets",
        headers=header,
        json={"bucketKey": bucket_key, "policyKey": policy},
        timeout=30,
    )
    if r.status_code == 200:
        print(f"[bucket] created: {bucket_key} ({policy}, {region})")
        return True
    if r.status_code == 409:
        print(f"[bucket] exists: {bucket_key} No Access due to different Clients bucket")
        return False
    raise RuntimeError(f"Bucket creating failed: {r.status_code} {r.text}")


# ----------------- SIGNED S3 UPLOAD -----------------

def s3_signed_upload(token: str, bucket_key: str, object_name: str, local_path: str, region: str = "US") -> Dict[str, Any]:
    """
    Новый поток загрузки:
      1) GET /signeds3upload - Get url and upload key
      2) PUT - give headers from the response
      3) POST /signeds3upload - gives
    returns JSON with meta
    """
    size = os.path.getsize(local_path)
    hdr = _headers_auth(token, region)

    q = {"contentLength": str(size), "parts": "1"}
    r1 = requests.get(
        f"{APS_BASE}/oss/v2/buckets/{bucket_key}/objects/{object_name}/signeds3upload",
        headers=hdr, params=q, timeout=30
    )
    if r1.status_code != 200:
        raise RuntimeError(f"GET signed s3 upload failed: {r1.status_code} {r1.text}")

    resp = r1.json()
    upload_key = resp["uploadKey"]
    urls = resp.get("urls")

    if isinstance(urls, list):
        first = urls[0]
    else:
        first = urls

    if isinstance(first, str):
        signed_url = first
        headers_for_s3 = {}
    elif isinstance(first, dict):
        signed_url = first.get("url")
        headers_for_s3 = first.get("headers", {}) or {}
    else:
        raise RuntimeError(f"Unsupported signed URL format: {type(first)}")

    if not signed_url:
        raise RuntimeError("Empty signed URL in response")

    # Gives data into s3 bucket
    with open(local_path, "rb") as f:
        put = requests.put(signed_url, data=f, headers=headers_for_s3, timeout=600)
    if put.status_code not in (200, 201):
        raise RuntimeError(f"S3 PUT failed: {put.status_code} {put.text}")

    etag = (put.headers.get("ETag") or put.headers.get("Etag") or "").strip('"')

    # uploading to OSS
    part = {"partNumber": 1}
    if etag:
        part["etag"] = etag

    finalize_body = {"uploadKey": upload_key, "parts": [part]}
    r3 = requests.post(
        f"{APS_BASE}/oss/v2/buckets/{bucket_key}/objects/{object_name}/signeds3upload",
        headers={**hdr, "Content-Type": "application/json"},
        data=json.dumps(finalize_body), timeout=60
    )
    if r3.status_code not in (200, 201):
        raise RuntimeError(f"POST signeds3upload failed: {r3.status_code} {r3.text}")
    return r3.json()


# ----------------- MODEL DERIVATIVE: RVT → IFC -----------------

def submit_derivative_ifc(token: str, urn_b64: str, ifc_version: str = "ifc4"):
    """
    Поставить задачу перевода в IFC (ifc4 | ifc2x3).
    """
    if ifc_version.lower() not in ("ifc4", "ifc2x3"):
        raise ValueError("Returned file not in .ifc4 or .ifc2x3")

    header = _headers_auth(token, extra={"Content-Type": "application/json"})
    job = {
        "input": {"urn": urn_b64},
        "output": {
            "formats": [{
                "type": "ifc",
                "advanced": {"ifcVersion": ifc_version.lower()}
            }]
        }
    }
    r = requests.post(
        f"{APS_BASE}/modelderivative/v2/designdata/job",
        headers=header, data=json.dumps(job), timeout=60
    )
    if r.status_code not in (200, 202):
        raise RuntimeError(f"MD job submit failed: {r.status_code} {r.text}")


def wait_manifest(token: str, urn_b64: str, poll_sec: int = 6, max_wait_min: int = 45) -> Dict[str, Any]:
    """
    Making Manifest for debugging if smth fails
    """
    header = _headers_auth(token)
    deadline = time.time() + max_wait_min * 60
    last_status = None
    while time.time() < deadline:
        r = requests.get(f"{APS_BASE}/modelderivative/v2/designdata/{urn_b64}/manifest",
                         headers=header, timeout=30)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status")
            if status != last_status:
                print(f"[manifest] status: {status}")
                last_status = status
            if status in ("success", "failed", "timeout"):
                return data
        elif r.status_code == 202:
            pass
        else:
            print(f"[manifest] {r.status_code} {r.text}")
        time.sleep(poll_sec)
    raise TimeoutError("Manifest polling timed out")


def find_by_ext(manifest: Dict[str, Any], exts: tuple[str, ...] = (".ifc",)) -> List[str]:
    found: List[str] = []

    def walk(nodes):
        for n in (nodes or []):
            if n.get("type") == "resource":
                urn = n.get("urn", "")
                if any(ext.lower() in urn.lower() for ext in exts):
                    found.append(urn)
            walk(n.get("children"))

    walk(manifest.get("derivatives"))
    return found


def download_derivative(token: str, urn_b64: str, derivative_urn: str, out_path: str):
    """Downloading derivatives"""
    hdr = _headers_auth(token)
    url = f"{APS_BASE}/modelderivative/v2/designdata/{urn_b64}/manifest/{derivative_urn}"
    r = requests.get(url, headers=hdr, timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed: {r.status_code} {r.text}")
    with open(out_path, "wb") as f:
        f.write(r.content)


# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rvt")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--region", default="EMEA", choices=["US", "EMEA"],)
    ap.add_argument("--ifcver", default="ifc4", choices=["ifc4", "ifc2x3"])
    ap.add_argument("--outfile", default=None) # default output is ifc
    ap.add_argument("--save-manifest", default=None) # optional for debugging
    args = ap.parse_args()

    src = pathlib.Path(args.rvt)
    if not src.is_file():
        raise FileNotFoundError(f"File not found: {src}")

    print(">>Getting Tokens")
    tok = get_token()

    # Bucket solution
    bucket_key = args.bucket or f"{CLIENT_ID.lower()}-rvt2ifc-{uuid.uuid4().hex[:8]}"
    me_created = ensure_bucket(tok, bucket_key, policy="temporary", region=args.region)
    if not me_created:
        print("Bucket created by another application(APS) so NO ACCESS")

    print(">>>Loading S3 signed URL")
    object_name = src.name
    meta = s3_signed_upload(tok, bucket_key, object_name, str(src), region=args.region)
    object_id = meta["objectId"]
    urn_b64 = _b64_urn(object_id)
    print("URN:", urn_b64)

    print(f">>>Start converting to {args.ifcver.upper()}")
    submit_derivative_ifc(tok, urn_b64, ifc_version=args.ifcver)
    manifest = wait_manifest(tok, urn_b64)

    if args.save_manifest:
        with open(args.save_manifest, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)

    if manifest.get("status") != "success":
        raise RuntimeError("Convert is not succeed:\n" + json.dumps(manifest, ensure_ascii=False, indent=2))

    # Look for ifc in all the mess
    durns = find_by_ext(manifest, exts=(".ifc",))
    if not durns:
        raise RuntimeError("IFC not found in manifest")

    out_ifc = args.outfile or src.with_suffix(".ifc").name
    print(f">>>Downloading IFC - {out_ifc}")
    download_derivative(tok, urn_b64, durns[0], out_ifc)
    print("Converting is done ", out_ifc)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        raise

# python3 rvt_to_ifc.py ENGIE_14.23-0120_PAMEIJER_ROTTERDAM.rvt --region EMEA --ifcver ifc4 --save-manifest manifest.json
