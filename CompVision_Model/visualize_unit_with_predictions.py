import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import numpy as np
import torch
import sys
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
import json
import pyvista as pv

# --- IMPORTS ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from wall_unfolding import UnfoldingEngine
from vision_model import UNet

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "../main/Data/DataOriginal")
DATASET_DIR = os.path.join(SCRIPT_DIR, "CV_Dataset")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
MIN_BLOB_SIZE = 10
HIGH_CONFIDENCE_THRESHOLD = 0.90
CONFIDENCE_THRESHOLD = 0.70
TARGET_UNIT = "rotterdam_0002"


# --- HELPER FUNCTIONS ---

def pixel_to_3d(u_pix, v_pix, wall):
    u_min, u_max = wall.u_range
    v_min, v_max = wall.v_range
    width = u_max - u_min
    height = v_max - v_min
    u_norm = u_pix / IMG_SIZE
    v_norm = (IMG_SIZE - v_pix) / IMG_SIZE
    u_dist = u_min + (u_norm * width)
    v_dist = v_min + (v_norm * height)
    return wall.origin + (u_dist * wall.u_axis) + (v_dist * wall.v_axis)


def get_centroids(probability_map, threshold=0.5, min_size=0):
    binary_mask = probability_map > threshold
    labeled_array, num_features = label(binary_mask)
    centroids = []
    if num_features > 0:
        centers = center_of_mass(binary_mask, labeled_array, range(1, num_features + 1))
        for i, center in enumerate(centers):
            if (labeled_array == (i + 1)).sum() >= min_size:
                centroids.append((center[1], center[0]))
    return centroids


def draw_existing_installations(plotter, unit_path):
    json_files = glob.glob(os.path.join(unit_path, "*.json"))
    if not json_files:
        return False

    has_bim_objects = False
    try:
        with open(json_files[0], 'r') as f:
            data = json.load(f)

        if 'objects' in data and 'ET' in data['objects']:
            for obj in data['objects']['ET']:
                min_p = obj['min']
                max_p = obj['max']

                # Create the 3D Box
                box = pv.Box(bounds=(min_p[0], max_p[0],
                                     min_p[1], max_p[1],
                                     min_p[2], max_p[2]))

                # Add to scene (NO LABEL HERE)
                plotter.add_mesh(box, color='#0099FF', opacity=0.3, style='surface')
                plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
                has_bim_objects = True

    except Exception as e:
        print(f"Could not load BIM objects: {e}")
    return has_bim_objects


# --- MAIN VISUALIZATION FUNCTION ---

def visualize_complete_analytics():
    unit_path, mesh_path = find_unit_paths()
    if not unit_path:
        print(f"Unit {TARGET_UNIT} not found.")
        return

    print(f"Loading Model from {MODEL_PATH}...")
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    except:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    engine = UnfoldingEngine(img_size=IMG_SIZE)

    print(f"Processing walls for {TARGET_UNIT}...")
    try:
        walls = engine.process_unit(unit_path, [])
    except Exception as e:
        print(f"Error processing unit geometry: {e}")
        return

    print("Building 3D Scene...")
    plotter = pv.Plotter()
    plotter.set_background('white')

    if os.path.exists(mesh_path):
        mesh = pv.read(mesh_path)
        # Draw Mesh (NO LABEL HERE)
        plotter.add_mesh(mesh, color='black', opacity=0.15, style='wireframe')

    # Draw BIM Objects and check if any exist
    has_bim_objects = draw_existing_installations(plotter, unit_path)

    gt_points = []
    high_conf_points = []
    med_conf_points = []

    path_parts = os.path.normpath(unit_path).split(os.sep)
    base_id = f"{path_parts[-2]}_{path_parts[-1]}"

    for wall in walls:
        img_filename = f"{base_id}_wall{wall.wall_id}.npy"
        img_path = os.path.join(DATASET_DIR, "images", img_filename)
        mask_path = os.path.join(DATASET_DIR, "masks", img_filename)

        if not os.path.exists(img_path): continue

        img = np.load(img_path)
        mask_gt = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((IMG_SIZE, IMG_SIZE))
        if mask_gt.ndim == 3: mask_gt = mask_gt[0]

        input_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            pred_map = torch.sigmoid(output).cpu().numpy()[0, 0]

        # A. Ground Truth
        gt_centers = get_centroids(mask_gt, threshold=0.5, min_size=1)
        for cx, cy in gt_centers:
            gt_points.append(pixel_to_3d(cx, cy, wall))

        # B. High Conf
        high_centers = get_centroids(pred_map, threshold=HIGH_CONFIDENCE_THRESHOLD, min_size=MIN_BLOB_SIZE)
        for cx, cy in high_centers:
            high_conf_points.append(pixel_to_3d(cx, cy, wall))

        # C. Med Conf
        med_centers = get_centroids(pred_map, threshold=CONFIDENCE_THRESHOLD, min_size=MIN_BLOB_SIZE)
        for cx, cy in med_centers:
            p3d = pixel_to_3d(cx, cy, wall)
            is_new = True
            if high_conf_points:
                dists = cdist([p3d], high_conf_points)
                if dists.min() < 0.1:
                    is_new = False
            if is_new:
                med_conf_points.append(p3d)

    # CALCULATE STATISTICS
    all_predictions = high_conf_points + med_conf_points

    print("\n" + "=" * 40)
    print(f" STATS FOR {TARGET_UNIT}")
    print("=" * 40)
    print(f"Ground Truth Sockets: {len(gt_points)}")
    print(f"High Conf Preds:      {len(high_conf_points)}")
    print(f"Med Conf Preds:       {len(med_conf_points)}")
    print(f"Total Predictions:    {len(all_predictions)}")

    if gt_points and all_predictions:
        gt_arr = np.array(gt_points)
        pred_arr = np.array(all_predictions)
        dists = cdist(pred_arr, gt_arr)
        min_dists_cm = dists.min(axis=1) / 10.0

        print("-" * 40)
        print(f"SPATIAL ACCURACY (All Predictions -> Nearest GT)")
        print(f"Min Error: {np.min(min_dists_cm):.2f} cm")
        print(f"Avg Error: {np.mean(min_dists_cm):.2f} cm")
        print(f"Max Error: {np.max(min_dists_cm):.2f} cm")

        hits = np.sum(min_dists_cm < 30.0)
        print(f"Accurate Hits (<30cm): {hits}/{len(all_predictions)}")
    else:
        print("\nCannot calculate error stats (Missing GT or Predictions).")
    print("=" * 40 + "\n")

    # Draw Points (NO LABELS HERE)
    if gt_points:
        plotter.add_mesh(pv.PolyData(np.array(gt_points)), color='#00CC00',
                         point_size=15, render_points_as_spheres=True)

    if high_conf_points:
        plotter.add_mesh(pv.PolyData(np.array(high_conf_points)), color='#FF0000',
                         point_size=20, render_points_as_spheres=True)

    if med_conf_points:
        plotter.add_mesh(pv.PolyData(np.array(med_conf_points)), color='#FFCC00',
                         point_size=15, render_points_as_spheres=True)

    # --- MANUAL LEGEND CREATION ---
    legend_entries = []
    if gt_points:
        legend_entries.append(('Ground Truth', '#00CC00'))
    if high_conf_points:
        legend_entries.append(('Prediction (High Conf)', '#FF0000'))
    if med_conf_points:
        legend_entries.append(('Prediction (Med Conf)', '#FFCC00'))
    if has_bim_objects:
        legend_entries.append(('BIM Components', '#0099FF'))

    # Add the manual legend if we have entries
    if legend_entries:
        plotter.add_legend(
            labels=legend_entries,
            bcolor='white',
            border=True,
            loc='upper left',
            size=[0.22, 0.18]
        )

    print("Opening Visualization...")
    plotter.show()


def find_unit_paths():
    if not os.path.exists(DATA_ROOT): return None, None
    target_parts = TARGET_UNIT.split("_")
    if len(target_parts) != 2: return None, None

    build_name, unit_name = target_parts[0], target_parts[1]

    for root, dirs, files in os.walk(DATA_ROOT):
        for d in dirs:
            if unit_name.lower() in d.lower() and build_name.lower() in root.lower():
                unit_path = os.path.join(root, d)
                obj_files = glob.glob(os.path.join(unit_path, "*.obj"))
                if obj_files:
                    return unit_path, obj_files[0]
    return None, None


if __name__ == "__main__":
    visualize_complete_analytics()