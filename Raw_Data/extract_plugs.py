import sys
import ifcopenshell

SOCKET_KEYWORDS = [
    "contactdoos", "wandcontactdoos", "wcd",
    "socket", "outlet", "power outlet", "plug",
    "stopcontact","steckdose", "doos",
]

def safe_str(v):
    try:
        if v is None:
            return ""
        return str(v)
    except Exception:
        return ""

def dump_el(el, idx):
    name = safe_str(getattr(el, "Name", ""))
    otype = safe_str(getattr(el, "ObjectType", ""))
    ptype = safe_str(getattr(el, "PredefinedType", ""))
    tag = safe_str(getattr(el, "Tag", ""))
    desc = safe_str(getattr(el, "Description", ""))
    gid = safe_str(getattr(el, "GlobalId", ""))

    print(f"{idx:04d}) IfcType={el.is_a()}  gid={gid}")
    print(f"     Name         : {name}")
    print(f"     ObjectType   : {otype}")
    print(f"     PredefinedType: {ptype}")
    print(f"     Tag          : {tag}")
    print(f"     Description  : {desc}")

def main():
    ifc_path = r"ifc_files/v22/MEP.ifc"
    limit = int(sys.argv[2]) if len(sys.argv) >= 3 else 300

    f = ifcopenshell.open(ifc_path)


    candidates_types = [
        "IfcFlowTerminal",
        "IfcElectricAppliance",
        "IfcOutlet",
        "IfcSwitchingDevice",
        "IfcDistributionControlElement",
    ]

    hits = []
    seen = set()

    for t in candidates_types:
        try:
            elems = f.by_type(t)
        except Exception:
            continue

        for el in elems:
            gid = safe_str(getattr(el, "GlobalId", ""))
            if gid in seen:
                continue

            name = safe_str(getattr(el, "Name", ""))
            otype = safe_str(getattr(el, "ObjectType", ""))
            ptype = safe_str(getattr(el, "PredefinedType", ""))
            tag = safe_str(getattr(el, "Tag", ""))
            desc = safe_str(getattr(el, "Description", ""))

            hay = (name + " " + otype + " " + ptype + " " + tag + " " + desc).lower()

            if any(k in hay for k in SOCKET_KEYWORDS):
                hits.append(el)
                seen.add(gid)

    print("=== SOCKET CANDIDATES (by keywords) ===")
    print(f"IFC file: {ifc_path}")
    print(f"Keywords: {', '.join(SOCKET_KEYWORDS)}")
    print(f"Found: {len(hits)}\n")

    for i, el in enumerate(hits[:limit], 1):
        dump_el(el, i)

    if len(hits) > limit:
        print(f"\ntruncated, showing first {limit} of {len(hits)}")

if __name__ == "__main__":
    main()
