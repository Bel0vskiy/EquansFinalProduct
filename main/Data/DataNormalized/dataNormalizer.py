import os
import json
import argparse
import numpy as np
from pathlib import Path

"""
This script normalizes mesh.obj and component coordinates inside data.json.
- Maps room bbox to [0,1]^3 space
- Normalizes all mesh vertices and component bboxes
- Writes output in a new folder structure (so original files stay untouched)
Usage:
    python3 dataNormalizer.py --input path/to/original --output path/to/normalized
"""

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def normalize_point(pt, room_min, room_max):
    pt = np.array(pt, dtype=np.float32)
    room_min = np.array(room_min, dtype=np.float32)
    room_max = np.array(room_max, dtype=np.float32)
    scale = np.maximum(room_max - room_min, 1e-6)
    return (pt - room_min) / scale

def normalize_bbox(bmin, bmax, room_min, room_max):
    bmin = normalize_point(bmin, room_min, room_max)
    bmax = normalize_point(bmax, room_min, room_max)
    return bmin.tolist(), bmax.tolist()

def process_unit(unit_dir, output_dir):
    mesh_path = os.path.join(unit_dir, "mesh.obj")
    json_path = os.path.join(unit_dir, "data.json")

    if not os.path.isfile(mesh_path) or not os.path.isfile(json_path):
        print(f"⚠️ Skipping {unit_dir}: missing mesh.obj or data.json")
        return

    data = load_json(json_path)
    rmin, rmax = data["min"], data["max"]

    # Normalize mesh vertices
    verts, other_lines = [], []
    with open(mesh_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nx, ny, nz = normalize_point([x, y, z], rmin, rmax)
                verts.append(f"v {nx} {ny} {nz}\n")
            else:
                other_lines.append(line)

    # Prepare output unit folder
    unit_name = os.path.basename(unit_dir)
    out_unit = os.path.join(output_dir, unit_name)
    os.makedirs(out_unit, exist_ok=True)

    # Save normalized mesh
    out_mesh = os.path.join(out_unit, "mesh.obj")
    with open(out_mesh, 'w') as f:
        for v in verts:
            f.write(v)
        for l in other_lines:
            f.write(l)

    # Normalize and simplify data.json (centroid + size)
    for category, objs in data.get("objects", {}).items():
        for obj in objs:
            bmin, bmax = obj["min"], obj["max"]
            nbmin, nbmax = normalize_bbox(bmin, bmax, rmin, rmax)
            obj["min"] = nbmin
            obj["max"] = nbmax

    # Normalize room bbox
    data["min"], data["max"] = [0,0,0], [1,1,1]

    # Save new JSON
    out_json = os.path.join(out_unit, "data.json")
    save_json(out_json, data)

    print(f"✅ Normalized: {unit_dir} -> {out_unit}")

def main():
    # ✅ Define your input and output paths here
    input_folder = "/Users/lucabighignoli/Desktop/uniProject/roomRender/Data/rotterdam"  # path of the building u want to normalize
    output_folder = "/Users/lucabighignoli/Desktop/uniProject/roomRender/DataNormalized"

    input_path = Path(input_folder)
    building_name = input_path.name  # e.g. "opvang"
    output_path = Path(output_folder) / f"{building_name}_normalized"
    os.makedirs(output_path, exist_ok=True)

    # ✅ Process each unit_ folder
    for item in os.listdir(input_path):
        u = input_path / item
        if u.is_dir() and item.startswith("unit_"):
            process_unit(str(u), str(output_path))

    print("\n✅ All units processed successfully.")

if __name__ == '__main__':
    main()