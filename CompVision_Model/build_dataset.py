import os
import glob
import numpy as np
import json
from tqdm import tqdm
import sys
import shutil

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wall_unfolding import UnfoldingEngine

# CONFIG
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "../main/Data/DataOriginal")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "CV_Dataset")
IMG_SIZE = 256
SOCKET_RADIUS_PX = 8


def load_components(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        targets = []
        if 'objects' in data:
            et_list = data['objects'].get('ET', [])
            for comp in et_list:
                name = comp.get('name', '').lower()
                if "wcd" in name or "socket" in name or "outlet" in name:
                    min_p = np.array(comp['min'])
                    max_p = np.array(comp['max'])
                    center = (min_p + max_p) / 2.0
                    comp['center_point'] = center
                    targets.append(comp)
        return targets
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []


# MATH HELPERS
def get_wall_segment(w):
    start = w.origin
    width = w.u_range[1] - w.u_range[0]
    end = start + (w.u_axis * width)
    return start, end


def get_distance_to_wall(point, start, end):
    p = point[:2]
    a = start[:2]
    b = end[:2]
    ab = b - a
    ap = p - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0: return np.linalg.norm(ap)
    t = max(0, min(1, np.dot(ap, ab) / len_sq))
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def draw_socket_on_mask(wall, components):
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    u_min, u_max = wall.u_range
    v_min, v_max = wall.v_range

    drawn_count = 0
    for comp in components:
        center = comp['center_point']
        rel_pos = center - wall.origin

        u_val = np.dot(rel_pos, wall.u_axis)
        v_val = np.dot(rel_pos, wall.v_axis)

        u_norm = (u_val - u_min) / (u_max - u_min)
        v_norm = (v_val - v_min) / (v_max - v_min)

        c_x = int(u_norm * IMG_SIZE)
        c_y = int((1.0 - v_norm) * IMG_SIZE)

        # Check if it actually lands on the image
        if 0 <= c_x < IMG_SIZE and 0 <= c_y < IMG_SIZE:
            y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            dist_sq = (x - c_x) ** 2 + (y - c_y) ** 2
            mask[dist_sq <= SOCKET_RADIUS_PX ** 2] = 1.0
            drawn_count += 1

    return mask, drawn_count


def assign_forcefully(walls, components):
    """
    Assigns EVERY component to its closest wall.
    NO DISTANCE THRESHOLD.
    """
    assignments = {i: [] for i in range(len(walls))}

    for comp in components:
        center = comp['center_point']

        best_dist = float('inf')
        best_wall_idx = -1

        for i, w in enumerate(walls):
            try:
                start, end = get_wall_segment(w)
                dist = get_distance_to_wall(center, start, end)

                if dist < best_dist:
                    best_dist = dist
                    best_wall_idx = i
            except:
                continue

        # FORCE ASSIGNMENT (unless walls list was empty)
        if best_wall_idx != -1:
            assignments[best_wall_idx].append(comp)

    return assignments


# MAIN BUILD LOOP
def build():
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: Data folder not found.")
        return

    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "meta"), exist_ok=True)

    unit_files = glob.glob(os.path.join(DATA_ROOT, "**", "data.json"), recursive=True)
    print(f"Found {len(unit_files)} units.")

    engine = UnfoldingEngine(img_size=IMG_SIZE)

    # STATISTICS
    total_sockets_found = 0
    total_sockets_drawn = 0
    total_walls_generated = 0

    for unit_json_path in tqdm(unit_files):
        unit_dir = os.path.dirname(unit_json_path)
        path_parts = os.path.normpath(unit_dir).split(os.sep)
        base_id = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else os.path.basename(unit_dir)

        comps = load_components(unit_json_path)
        total_sockets_found += len(comps)

        try:
            # 1. Get Walls (Skeleton)
            walls = engine.process_unit(unit_dir, [])
            if not walls: continue

            # 2. Force Assignment
            assignments = assign_forcefully(walls, comps)

            # 3. Draw & Save
            for i, w in enumerate(walls):
                my_sockets = assignments.get(i, [])

                # DRAW
                w.label_mask, drawn = draw_socket_on_mask(w, my_sockets)
                total_sockets_drawn += drawn

                # Save
                np.save(os.path.join(OUTPUT_DIR, "images", f"{base_id}_wall{w.wall_id}.npy"), w.image_tensor)
                np.save(os.path.join(OUTPUT_DIR, "masks", f"{base_id}_wall{w.wall_id}.npy"), w.label_mask)

                meta = {
                    "origin": w.origin.tolist(),
                    "u_axis": w.u_axis.tolist(),
                    "v_axis": w.v_axis.tolist(),
                    "u_range": w.u_range,
                    "v_range": w.v_range
                }
                with open(os.path.join(OUTPUT_DIR, "meta", f"{base_id}_wall{w.wall_id}.json"), 'w') as f:
                    json.dump(meta, f)

                total_walls_generated += 1

        except Exception as e:
            continue

    print(f"\n--- BUILD REPORT ---")
    print(f"Total Walls Generated: {total_walls_generated}")
    print(f"Total Sockets Found in JSON: {total_sockets_found}")
    print(f"Total Sockets Drawn on Masks: {total_sockets_drawn}")
    print(f"Difference (Off-screen/Lost): {total_sockets_found - total_sockets_drawn}")
    print(f"--------------------")
    print(f"Dataset ready in {OUTPUT_DIR}")


if __name__ == "__main__":
    build()