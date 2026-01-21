# run_extract_locked_onefile.py
import json
import math
import ifcopenshell
import ifcopenshell.geom
from ifcopenshell.util.placement import get_local_placement

IFC_PATH = "/Users/d1feds/PycharmProjects/ExperimentingProject1/Raw_Data/ifc_files/Gebouw E/ARCH.ifc"
OUT_JSON = "rooms_full.json"

MIN_ROOM_AREA_M2 = 0.5
Z_MARGIN = 0.25

WALL_TYPES = ["IfcWall", "IfcWallStandardCase"]
DEVICE_TYPES = [
    "IfcFlowTerminal",
    "IfcOutlet",
    "IfcSwitchingDevice",
    "IfcDistributionControlElement",
    "IfcBuildingElementProxy",
]

LAMP_KEYWORDS = ["lamp", "light", "led", "armatuur", "luminaire"]
SOCKET_KEYWORDS = ["wcd", "contactdoos", "socket", "outlet", "plug"]

def get_scale(ifc):
    try:
        from ifcopenshell.util.unit import calculate_unit_scale
        return float(calculate_unit_scale(ifc))
    except:
        return 1.0


def get_bbox(el, settings, scale):
    try:
        shape = ifcopenshell.geom.create_shape(settings, el)
        v = shape.geometry.verts
        xs = v[0::3]
        ys = v[1::3]
        zs = v[2::3]
        return (
            (min(xs)*scale, min(ys)*scale, min(zs)*scale),
            (max(xs)*scale, max(ys)*scale, max(zs)*scale)
        )
    except:
        return None


def intersects_xy(a, b):
    return not (
        a[1][0] < b[0][0] or b[1][0] < a[0][0] or
        a[1][1] < b[0][1] or b[1][1] < a[0][1]
    )


def intersects_z(a, b):
    return not (
        a[1][2] + Z_MARGIN < b[0][2] or
        b[1][2] + Z_MARGIN < a[0][2]
    )


def wall_surface_2x2(bmin, bmax):
    minx, miny, minz = bmin
    maxx, maxy, maxz = bmax
    dx = maxx - minx
    dy = maxy - miny

    if dx >= dy:
        y = (miny + maxy) / 2
        return [
            [[minx, y, minz], [maxx, y, minz]],
            [[minx, y, maxz], [maxx, y, maxz]]
        ]
    else:
        x = (minx + maxx) / 2
        return [
            [[x, miny, minz], [x, maxy, minz]],
            [[x, miny, maxz], [x, maxy, maxz]]
        ]


def floor_surface(room):
    minx, miny, minz = room["bbox_min"]
    maxx, maxy, _ = room["bbox_max"]
    return [
        [[minx, miny, minz], [maxx, miny, minz]],
        [[minx, maxy, minz], [maxx, maxy, minz]]
    ]


def ceiling_surface(room):
    minx, miny, _ = room["bbox_min"]
    maxx, maxy, maxz = room["bbox_max"]
    return [
        [[minx, miny, maxz], [maxx, miny, maxz]],
        [[minx, maxy, maxz], [maxx, maxy, maxz]]
    ]


def main():
    ifc = ifcopenshell.open(IFC_PATH)
    scale = get_scale(ifc)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    rooms = []
    for sp in ifc.by_type("IfcSpace"):
        bbox = get_bbox(sp, settings, scale)
        if not bbox:
            continue
        (minx, miny, minz), (maxx, maxy, maxz) = bbox
        if (maxx-minx)*(maxy-miny) < MIN_ROOM_AREA_M2:
            continue

        rooms.append({
            "id": sp.GlobalId,
            "name": sp.Name or "",
            "bbox_min": [minx, miny, minz],
            "bbox_max": [maxx, maxy, maxz],
            "walls": [],
            "floor": None,
            "ceiling": None,
            "lamps": [],
            "sockets": []
        })

    # ---------- WALLS ----------
    walls = []
    for wt in WALL_TYPES:
        for w in ifc.by_type(wt):
            bbox = get_bbox(w, settings, scale)
            if not bbox:
                continue
            bmin, bmax = bbox
            walls.append({
                "id": w.GlobalId,
                "name": w.Name or "",
                "bbox": bbox,
                "surface_2x2": wall_surface_2x2(bmin, bmax)
            })

    # assign walls to rooms
    for r in rooms:
        rb = (tuple(r["bbox_min"]), tuple(r["bbox_max"]))
        for w in walls:
            wb = w["bbox"]
            if intersects_xy(rb, wb) and intersects_z(rb, wb):
                r["walls"].append({
                    "id": w["id"],
                    "name": w["name"],
                    "surface_2x2": w["surface_2x2"]
                })

        r["floor"] = {"surface_2x2": floor_surface(r)}
        r["ceiling"] = {"surface_2x2": ceiling_surface(r)}

    # ---------- DEVICES ----------
    for dt in DEVICE_TYPES:
        for d in ifc.by_type(dt):
            text = f"{d.Name} {d.ObjectType}".lower()
            is_lamp = any(k in text for k in LAMP_KEYWORDS)
            is_socket = any(k in text for k in SOCKET_KEYWORDS)
            if not (is_lamp or is_socket):
                continue

            try:
                M = get_local_placement(d.ObjectPlacement)
                x, y, z = M[0][3]*scale, M[1][3]*scale, M[2][3]*scale
            except:
                continue

            for r in rooms:
                minx, miny, minz = r["bbox_min"]
                maxx, maxy, maxz = r["bbox_max"]
                if (minx <= x <= maxx and
                    miny <= y <= maxy and
                    minz-Z_MARGIN <= z <= maxz+Z_MARGIN):
                    entry = {
                        "id": d.GlobalId,
                        "name": d.Name or "",
                        "x": x, "y": y, "z": z
                    }
                    if is_lamp:
                        r["lamps"].append(entry)
                    if is_socket:
                        r["sockets"].append(entry)
                    break

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rooms, f, indent=2, ensure_ascii=False)

    print("DONE:", OUT_JSON)
    print("rooms:", len(rooms))


if __name__ == "__main__":
    main()
