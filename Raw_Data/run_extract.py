# run_extract_v26_locked.py
# Запуск: python run_extract_v26_locked.py

import csv
import json
import math
import os
import glob
from collections import Counter

import ifcopenshell
import ifcopenshell.geom
from ifcopenshell.util.placement import get_local_placement


ifc_folder = "/Users/d1feds/PycharmProjects/ExperimentingProject1/Raw_Data/ifc_files"
OUT_PREFIX    = "room_list_all"


MIN_ROOM_AREA_M2 = 0.5


DEVICE_BBOX_MARGIN_M = 2.0

Z_MARGIN = 0.25
MAX_SNAP_DIST_M_SOCKETS = 0.4
MAX_SNAP_DIST_M_LAMPS   = 0.3
LAMP_KEYWORDS = [
    "light", "lamp", "luminaire", "armatuur", "armature", "led",
    "sylvania",
]
SOCKET_KEYWORDS = [
    "contactdoos", "wandcontactdoos", "wcd",
    "socket", "outlet", "stopcontact",
    "plug", "steckdose", "schuko",
]
EXCLUDE_FROM_LAMPS = SOCKET_KEYWORDS

DEVICE_TYPES = [
    "IfcFlowTerminal",
    "IfcElectricAppliance",
    "IfcOutlet",
    "IfcSwitchingDevice",
    "IfcDistributionControlElement",
    "IfcBuildingElementProxy",
]


def string_verify_not_none(v):
    return "" if v is None else str(v)

def get_scale_m(ifc):
    try:
        from ifcopenshell.util.unit import calculate_unit_scale
        return float(calculate_unit_scale(ifc))
    except Exception:
        return 1.0

def get_bbox(product, settings, scale):
    try:
        shape = ifcopenshell.geom.create_shape(settings, product)
        v = shape.geometry.verts
        if not v:
            return None
        xs = v[0::3]; ys = v[1::3]; zs = v[2::3]
        return (
            (min(xs)*scale, min(ys)*scale, min(zs)*scale),
            (max(xs)*scale, max(ys)*scale, max(zs)*scale)
        )
    except Exception:
        return None

def get_point(product, scale):
    try:
        M = get_local_placement(product.ObjectPlacement)
        return (M[0][3]*scale, M[1][3]*scale, M[2][3]*scale)
    except Exception:
        return None

def bbox_union(bboxes):
    if not bboxes:
        return None
    minx = min(b[0][0] for b in bboxes)
    miny = min(b[0][1] for b in bboxes)
    minz = min(b[0][2] for b in bboxes)
    maxx = max(b[1][0] for b in bboxes)
    maxy = max(b[1][1] for b in bboxes)
    maxz = max(b[1][2] for b in bboxes)
    return (minx, miny, minz), (maxx, maxy, maxz)

def center_of_union(union_bbox):
    (minx, miny, minz), (maxx, maxy, maxz) = union_bbox
    return ((minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2)

def expand_bbox_xy(b, margin):
    (minx, miny, minz), (maxx, maxy, maxz) = b
    return (minx-margin, miny-margin, minz), (maxx+margin, maxy+margin, maxz)

def intersects_xy(a, b):
    (aminx, aminy, _), (amaxx, amaxy, _) = a
    (bminx, bminy, _), (bmaxx, bmaxy, _) = b
    return not (amaxx < bminx or bmaxx < aminx or amaxy < bminy or bmaxy < aminy)

def apply_offset(objs, dx, dy, dz):
    for o in objs:
        o["x"] += dx
        o["y"] += dy
        o["z"] += dz

def compute_auto_offset(rooms, device_points):
    if not rooms or not device_points:
        return (0.0, 0.0, 0.0)
    ru = bbox_union([(tuple(r["bbox_min"]), tuple(r["bbox_max"])) for r in rooms])
    if not ru:
        return (0.0, 0.0, 0.0)
    du = bbox_union([((x, y, z), (x, y, z)) for (x, y, z) in device_points])
    if not du:
        return (0.0, 0.0, 0.0)
    rc = center_of_union(ru)
    dc = center_of_union(du)
    return (rc[0]-dc[0], rc[1]-dc[1], rc[2]-dc[2])

def score_scale(space_union, dev_union):
    if not space_union or not dev_union:
        return float("inf")
    (sx0, sy0, _), (sx1, sy1, _) = space_union
    (dx0, dy0, _), (dx1, dy1, _) = dev_union
    sdx = max(sx1 - sx0, 1e-9)
    sdy = max(sy1 - sy0, 1e-9)
    ddx = max(dx1 - dx0, 1e-9)
    ddy = max(dy1 - dy0, 1e-9)

    rx = sdx / ddx
    ry = sdy / ddy
    return abs(math.log(rx)) + abs(math.log(ry))


# DEVICES

def classify(el):
    text = " ".join([
        s(getattr(el, "Name", "")),
        s(getattr(el, "ObjectType", "")),
        s(getattr(el, "PredefinedType", "")),
        s(getattr(el, "Tag", "")),
        s(getattr(el, "Description", "")),
    ]).lower()

    is_socket = any(k in text for k in SOCKET_KEYWORDS)
    is_lamp = any(k in text for k in LAMP_KEYWORDS) and not any(k in text for k in EXCLUDE_FROM_LAMPS)
    return is_lamp, is_socket

def extract_devices(ifc, scale):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    lamps, sockets = [], []
    stat_l, stat_s = Counter(), Counter()
    pts_all = []

    for t in DEVICE_TYPES:
        try:
            elems = ifc.by_type(t)
        except Exception:
            continue

        for el in elems:
            is_lamp, is_socket = classify(el)
            if not is_lamp and not is_socket:
                continue

            pt = get_point(el, scale)
            if pt is None:
                bbox = get_bbox(el, settings, scale)
                if not bbox:
                    continue
                (a, b, c), (d, e, f) = bbox
                pt = ((a+d)/2, (b+e)/2, (c+f)/2)

            rec = {
                "id": s(getattr(el, "GlobalId", "")),
                "ifc_type": el.is_a(),
                "name": s(getattr(el, "Name", "")),
                "object_type": s(getattr(el, "ObjectType", "")),
                "predefined_type": s(getattr(el, "PredefinedType", "")),
                "tag": s(getattr(el, "Tag", "")),
                "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2]),
            }
            pts_all.append((rec["x"], rec["y"], rec["z"]))

            if is_lamp:
                lamps.append(rec)
                stat_l[el.is_a()] += 1
            if is_socket:
                sockets.append(rec)
                stat_s[el.is_a()] += 1

    return lamps, sockets, stat_l, stat_s, pts_all



def extract_spaces_raw(ifc, scale):
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    spaces = []
    for sp in ifc.by_type("IfcSpace"):
        bbox = get_bbox(sp, settings, scale)
        if not bbox:
            continue

        (minx, miny, minz), (maxx, maxy, maxz) = bbox
        dx, dy, dz = (maxx-minx), (maxy-miny), (maxz-minz)
        if dx <= 0 or dy <= 0:
            continue

        spaces.append({
            "id": s(getattr(sp, "GlobalId", "")),
            "name": s(getattr(sp, "Name", "")),
            "bbox_min": [minx, miny, minz],
            "bbox_max": [maxx, maxy, maxz],
            "length": dx,
            "width": dy,
            "height": dz,
            "area": dx * dy,
            "lamps": [],
            "sockets": [],
        })

    return spaces

def pick_best_arch_scale(arch_ifc, arch_unit_scale, dev_union):
    candidates = [arch_unit_scale, 1.0]
    best = None
    best_score = float("inf")
    best_union = None
    best_count = 0

    for sc in candidates:
        spaces = extract_spaces_raw(arch_ifc, sc)
        su = bbox_union([(tuple(r["bbox_min"]), tuple(r["bbox_max"])) for r in spaces]) if spaces else None
        score = score_scale(su, dev_union)
        score2 = score + (0.0 if len(spaces) > 0 else 10.0)
        if score2 < best_score:
            best_score = score2
            best = sc
            best_union = su
            best_count = len(spaces)

    print(f"[ARCH scale pick] unit_scale={arch_unit_scale} -> chosen={best} (spaces={best_count}) score={best_score:.3f}")
    print(f"[ARCH scale pick] chosen rooms union bbox: {best_union}")
    return best


def filter_spaces_by_devices(spaces, dev_union):
    if not spaces or not dev_union:
        return spaces

    dev_exp = expand_bbox_xy(dev_union, DEVICE_BBOX_MARGIN_M)

    kept = []
    for r in spaces:
        rb = (tuple(r["bbox_min"]), tuple(r["bbox_max"]))
        if not intersects_xy(rb, dev_exp):
            continue
        if r["area"] < MIN_ROOM_AREA_M2:
            continue
        kept.append(r)

    return kept


def rect_distance_sq_xy(x, y, r):
    minx, miny, _ = r["bbox_min"]
    maxx, maxy, _ = r["bbox_max"]

    dx = 0.0
    if x < minx: dx = (minx - x)
    elif x > maxx: dx = (x - maxx)

    dy = 0.0
    if y < miny: dy = (miny - y)
    elif y > maxy: dy = (y - maxy)

    return dx*dx + dy*dy

def assign_with_snap(rooms, objs, key, max_snap_dist_m):
    assigned_inside = 0
    assigned_snap = 0

    max_snap_sq = max_snap_dist_m * max_snap_dist_m

    for o in objs:
        x, y, z = o["x"], o["y"], o["z"]

        inside_candidates = []
        snap_candidates = []

        for r in rooms:
            minx, miny, minz = r["bbox_min"]
            maxx, maxy, maxz = r["bbox_max"]

            inside_xy = (minx <= x <= maxx and miny <= y <= maxy)
            inside_z  = (minz - Z_MARGIN <= z <= maxz + Z_MARGIN)

            if inside_xy and inside_z:
                cx = (minx + maxx) / 2
                cy = (miny + maxy) / 2
                cz = (minz + maxz) / 2
                d = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                inside_candidates.append((d, r))
            else:
                dxy_sq = rect_distance_sq_xy(x, y, r)
                if dxy_sq <= max_snap_sq and inside_z:
                    snap_candidates.append((dxy_sq, r))

        chosen = None

        if inside_candidates:
            _, chosen = min(inside_candidates, key=lambda v: v[0])
            assigned_inside += 1
        elif snap_candidates:
            _, chosen = min(snap_candidates, key=lambda v: v[0])
            assigned_snap += 1

        if chosen is None:
            continue

        r = chosen
        o2 = dict(o)
        o2["x_norm"] = (x - r["bbox_min"][0]) / (r["length"] if r["length"] else 1.0)
        o2["y_norm"] = (y - r["bbox_min"][1]) / (r["width"]  if r["width"]  else 1.0)
        o2["z_norm"] = (z - r["bbox_min"][2]) / (r["height"] if r["height"] else 1.0)
        o2["assigned_mode"] = "inside" if inside_candidates else "snap"
        r[key].append(o2)

    print(f"{key} assigned inside: {assigned_inside} | snapped: {assigned_snap} | total: {assigned_inside+assigned_snap} / {len(objs)}")
    return assigned_inside + assigned_snap

def export_all(buildings):
    with open(OUT_PREFIX + ".json", "w", encoding="utf-8") as f:
        json.dump(buildings, f, ensure_ascii=False, indent=2)

    with open(OUT_PREFIX + ".csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow([
            "room_id", "room_name", "room_length","room_width","room_height", "area_m2",
            "component_type", "component_id",
            "object_type",
            "x_m", "y_m", "z_m", "x_norm", "y_norm", "z_norm"
        ])
        for r in buildings:
                for comp_type, key in [("lamp", "lamps"), ("socket", "sockets")]:
                    for c in r.get(key, []):
                        wr.writerow([
                            r["id"], r["name"], r["length"], r["width"],r["height"], r.get("area", ""),
                            comp_type, c.get("id", ""),
                            c.get("object_type", ""),
                            c.get("x", ""), c.get("y", ""), c.get("z", ""),
                            c.get("x_norm", ""), c.get("y_norm", ""), c.get("z_norm", "")
                        ])

def pick_ifc_file(building_dir, kind: str):
    pattern = "**/*.ifc"
    files = glob.glob(os.path.join(building_dir, pattern), recursive=True)
    if not files:
        return None

    kind_low = kind.lower()

    def score(p):
        base = os.path.basename(p).lower()
        if kind_low == "arch" and base == "arch.ifc":
            return (0, len(p))
        if kind_low == "mep" and base == "mep.ifc":
            return (0, len(p))

        if kind_low in base:
            return (1, len(p))

        if kind_low == "arch" and "arch" in base:
            return (2, len(p))
        if kind_low == "mep" and "mep" in base:
            return (2, len(p))

        return (9, len(p))

    candidates = []
    for f in files:
        b = os.path.basename(f).lower()
        if kind_low == "arch" and ("arch" in b):
            candidates.append(f)
        elif kind_low == "mep" and ("mep" in b):
            candidates.append(f)

    if not candidates:
        return None

    candidates.sort(key=score)
    return candidates[0]

def main():
    building_dirs = [
        os.path.join(ifc_folder, d)
        for d in sorted(os.listdir(ifc_folder))
        if os.path.isdir(os.path.join(ifc_folder, d))
    ]
    rooms = []
    for dir in building_dirs:
        bname = os.path.basename(dir)
        print(bname)
        arch_path = pick_ifc_file(dir, "arch")
        mep_path = pick_ifc_file(dir, "mep")
        arch = ifcopenshell.open(arch_path)
        mep = ifcopenshell.open(mep_path)

        arch_unit_scale = get_scale_m(arch)
        mep_scale = get_scale_m(mep)

        print("ARCH unit scale:", arch_unit_scale)
        print("MEP  unit scale:", mep_scale)

        lamps, sockets, sl, ss, dev_pts = extract_devices(mep, mep_scale)
        dev_union = bbox_union([((x, y, z), (x, y, z)) for (x, y, z) in dev_pts]) if dev_pts else None

        print("Lamps found:", len(lamps), dict(sl))
        print("Sockets found:", len(sockets), dict(ss))
        print("Devs union bbox:", dev_union)

        arch_scale = pick_best_arch_scale(arch, arch_unit_scale, dev_union)

        spaces = extract_spaces_raw(arch, arch_scale)

        spaces_union = bbox_union([(tuple(r["bbox_min"]), tuple(r["bbox_max"])) for r in spaces]) if spaces else None

        room = (filter_spaces_by_devices(spaces, dev_union))

        rooms_union = bbox_union([(tuple(r["bbox_min"]), tuple(r["bbox_max"])) for r in room]) if room else None
        print("Rooms union bbox:", rooms_union)

        dx, dy, dz = compute_auto_offset(room, dev_pts)
        apply_offset(lamps, dx, dy, dz)
        apply_offset(sockets, dx, dy, dz)

        al = assign_with_snap(room, lamps, "lamps", MAX_SNAP_DIST_M_LAMPS)
        as_ = assign_with_snap(room, sockets, "sockets", MAX_SNAP_DIST_M_SOCKETS)

        print("Lamps assigned:", al, "/", len(lamps))
        print("Sockets assigned:", as_, "/", len(sockets))
        for each_room in room:
            rooms.append(each_room)
    export_all(rooms)
    print("Exported:", OUT_PREFIX + ".json and .csv")
    print("Done.")


if __name__ == "__main__":
    main()
