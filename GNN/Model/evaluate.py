import torch
import os
import argparse
import json
import numpy as np
from torch_geometric.loader import DataLoader
from typing import Optional, Tuple, List, Dict, Any

# Internal imports# Internal imports
from model import ComponentPlacementGNN
from graph_builder import build_room_graph

# --- CONFIGURATION ---
TARGET_COMPONENT_NAME = "wcd enkelvoudig"

# --- GEOMETRY & MATH UTILITIES ---

def _compute_uv_vectors(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes orthogonal u and v vectors for a given surface normal."""
    n = normal / (np.linalg.norm(normal) + 1e-6)
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(np.dot(u, n)) > 0.9:
        u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    u = u - np.dot(u, n) * n
    u /= (np.linalg.norm(u) + 1e-6)
    v = np.cross(n, u)
    return u, v

def get_xyz_from_uv(uv: List[float], centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Computes global (x, y, z) from local (u, v) and surface parameters."""
    u_vec, v_vec = _compute_uv_vectors(normal)
    return centroid + (uv[0] * u_vec) + (uv[1] * v_vec)

def global_denormalize(
    point_norm: np.ndarray,
    room_min: np.ndarray,
    room_max: np.ndarray
) -> np.ndarray:
    """
    Converts a point from Normalized Space ([0,1]^3) to Real-World Space (mm).
    Formula: P_real = P_norm * (Room_Max - Room_Min) + Room_Min
    """
    scale = room_max - room_min
    return point_norm * scale + room_min

def load_real_bounds(unit_path_norm: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derives the path to the original 'data.json' from the normalized unit path
    and loads the room 'min' and 'max' bounds.
    """
    # Assumption: structure is .../DataNormalized/unit_X
    # We want: .../Data/unit_X/data.json
    
    # 1. Replace 'DataNormalized' with 'Data'
    path_real = unit_path_norm.replace("DataNormalized", "Data")
    
    # 2. Append 'data.json'
    json_path = os.path.join(path_real, "data.json")
    print(f"DEBUG: derived real json path: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"⚠️ Warning: Original data.json not found at {json_path}")
        # Fallback: Return 0,0,0 to 1,1,1 (No scaling)
        return np.zeros(3), np.ones(3)
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    rmin = np.array(data.get("min", [0,0,0]), dtype=np.float32)
    rmax = np.array(data.get("max", [1,1,1]), dtype=np.float32)
    print(f"DEBUG: Loaded Bounds: Min={rmin}, Max={rmax}")
    return rmin, rmax

def get_bbox_corners(
    centroid: np.ndarray, # [x, y, z]
    bbox_size: np.ndarray # [width, height, depth]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the 8 corners and min/max points of the Axis-Aligned Bounding Box."""
    cx, cy, cz = centroid
    dx, dy, dz = bbox_size / 2.0
    
    min_pt = np.array([cx - dx, cy - dy, cz - dz])
    max_pt = np.array([cx + dx, cy + dy, cz + dz])
    
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]], [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]], [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]], [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]], [min_pt[0], max_pt[1], max_pt[2]]
    ])
    return corners, min_pt, max_pt

# --- DATA PROCESSING UTILITIES ---

def get_surface_type_from_features(surface_features: torch.Tensor) -> str:
    """Decodes the surface type from its one-hot encoded feature vector."""
    type_one_hot = surface_features[-4:]
    type_idx = torch.argmax(type_one_hot).item()
    surface_types = ["wall", "floor", "ceiling", "other"]
    return surface_types[type_idx] if 0 <= type_idx < len(surface_types) else "unknown"

def apply_mask_to_graph(graph, comp_idx: int):
    """Applies the exact same masking logic as training."""
    masked_graph = graph.clone()
    masked_graph.meta = graph.meta.copy() if hasattr(graph, 'meta') else {}
    masked_graph.meta['masked_component_idx'] = [comp_idx] 
    
    masked_x = masked_graph['component'].x.clone()
    masked_x[comp_idx, 0:3] = 0.0 
    masked_graph['component'].x = masked_x
    return masked_graph

# --- EVALUATION CORE ---

def get_model_prediction(model, data, target_comp_idx: int):
    """Runs inference and extracts specific predictions for the target component."""
    with torch.no_grad():
        classification_logits, regression_output = model(data)

    candidate_edges = data['component', 'candidate_placement', 'surface'].edge_index
    # --- FIX STARTS HERE ---
    # Ensure target_comp_idx is a standard integer (unwrap if it's a 1-element list/tensor)
    if isinstance(target_comp_idx, (list, tuple)):
        target_comp_idx = target_comp_idx[0]
    elif isinstance(target_comp_idx, torch.Tensor):
        target_comp_idx = target_comp_idx.item()

    # Create mask using standard torch comparison
    comp_specific_edge_mask = (candidate_edges[0] == target_comp_idx)
    # --- FIX ENDS HERE ---
    
    if not comp_specific_edge_mask.any():
        return None

    comp_logits = classification_logits[comp_specific_edge_mask]
    comp_regression = regression_output[comp_specific_edge_mask]
    
    predicted_local_idx = torch.argmax(comp_logits)
    pred_global_surf_idx = candidate_edges[1][comp_specific_edge_mask][predicted_local_idx].item()
    predicted_coords = comp_regression[predicted_local_idx].tolist()
    
    return {
        'logits': comp_logits,
        'coords': predicted_coords,
        'global_surf_idx': pred_global_surf_idx,
        'local_idx': predicted_local_idx,
        'edge_mask': comp_specific_edge_mask
    }

def get_ground_truth(data, comp_specific_edge_mask):
    """Extracts ground truth labels from the graph data."""
    y_surface = data['component', 'candidate_placement', 'surface'].y_surface[comp_specific_edge_mask]
    y_pose = data['component', 'candidate_placement', 'surface'].y_pose[comp_specific_edge_mask]
    candidate_edges = data['component', 'candidate_placement', 'surface'].edge_index
    
    true_local_idx = torch.argmax(y_surface).item()
    true_global_surf_idx = candidate_edges[1][comp_specific_edge_mask][true_local_idx].item()
    true_coords = y_pose[true_local_idx].tolist()
    
    return {
        'coords': true_coords,
        'global_surf_idx': true_global_surf_idx,
        'local_idx': true_local_idx
    }

def print_results(unit_name, target_comp_idx, prediction, ground_truth, data, unmasked_comp_x=None, room_min=None, room_max=None):
    """Formats and prints the evaluation results."""
    print(f"\n{'='*60}")
    print(f" EVALUATING: {unit_name}")
    print(f" TARGET COMPONENT INDEX: {target_comp_idx}")
    print(f"{'='*60}")

    # Candidate scores
    print("\n  Candidate Surface Scores:")
    probs = torch.sigmoid(prediction['logits'])
    sorted_indices = torch.argsort(probs, descending=True)
    candidate_edges = data['component', 'candidate_placement', 'surface'].edge_index
    
    for i in sorted_indices:
        idx = i.item()
        surf_idx = candidate_edges[1][prediction['edge_mask']][idx].item()
        score = probs[idx].item()
        surf_type = get_surface_type_from_features(data['surface'].x[surf_idx])
        
        marker = ""
        if idx == ground_truth['local_idx']: marker += "  <-- GROUND TRUTH"
        if idx == prediction['local_idx']: marker += "  <-- PREDICTED"
        print(f"    - Surface {surf_idx:2d} ({surf_type:7s}): {score:.4f}{marker}")

    print("\n  Final Result:")

    # 1. Normalized positions (Unit scale)
    norm_pred_feat = data['surface'].x[prediction['global_surf_idx']]
    norm_true_feat = data['surface'].x[ground_truth['global_surf_idx']]

    # --- ADDED FOR UI ---
    norm_true_centroid = norm_true_feat[0:3].cpu().numpy()
    norm_true_bbox = norm_true_feat[6:9].cpu().numpy()
    corners, _, _ = get_bbox_corners(norm_true_centroid, norm_true_bbox)

    print("\n  True Surface Bounding Box Corners (Normalized):")
    corners_str = "[\n" + ",\n".join([f"    [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]" for c in corners]) + "\n]"
    print(corners_str)
    # --- END ADDED ---

    norm_true_surf_centroid = norm_true_feat[0:3].cpu().numpy()
    
    norm_pred_xyz = get_xyz_from_uv(
        prediction['coords'], 
        norm_pred_feat[0:3].cpu().numpy(), 
        norm_pred_feat[3:6].cpu().numpy()
    )
    norm_true_xyz = get_xyz_from_uv(
        ground_truth['coords'], 
        norm_true_feat[0:3].cpu().numpy(), 
        norm_true_feat[3:6].cpu().numpy()
    )

    # 2. Real-world positions (mm scale)
    if room_min is not None and room_max is not None:
        pred_xyz_mm = global_denormalize(norm_pred_xyz, room_min, room_max)
        true_xyz_mm = global_denormalize(norm_true_xyz, room_min, room_max)
    else:
        pred_xyz_mm, true_xyz_mm = None, None

    if unmasked_comp_x is not None:
        original_centroid = unmasked_comp_x[target_comp_idx, 0:3].cpu().numpy()
        print(f"    Original (Unmasked) Component Centroid: {np.array2string(original_centroid, precision=3, separator=', ')}")

    print(f"    True Surface Centroid (norm):   {np.array2string(norm_true_surf_centroid, precision=3, separator=', ')}")

    correct_wall = "[MATCH]" if prediction['global_surf_idx'] == ground_truth['global_surf_idx'] else "[MISMATCH]"
    print(f"    Wall Selection: {correct_wall}")
    print(f"       Pred Surface: {prediction['global_surf_idx']}")
    print(f"       True Surface: {ground_truth['global_surf_idx']}")
    print(f"    Predicted (u,v): {prediction['coords'][0]:.3f}, {prediction['coords'][1]:.3f}")
    print(f"    True (u,v):      {ground_truth['coords'][0]:.3f}, {ground_truth['coords'][1]:.3f}")
    
    print(f"    True Position (norm):    {np.array2string(norm_true_xyz, precision=3, separator=', ')}")
    print(f"    Pred Position (norm):    {np.array2string(norm_pred_xyz, precision=3, separator=', ')}")

    if pred_xyz_mm is not None:
        print(f"    True Position (mm):      {np.array2string(true_xyz_mm, precision=1, separator=', ')}")
        print(f"    Predicted Position (mm): {np.array2string(pred_xyz_mm, precision=1, separator=', ')}")
        dist = np.linalg.norm(pred_xyz_mm - true_xyz_mm)
        print(f"    Error: {dist:.1f} mm")
    else:
        print("    (Real graph not available for mm conversion)")

def evaluate_graph(model, data, unmasked_comp_x=None, room_min=None, room_max=None):
    """Main function to evaluate a single graph sample."""
    model.eval()

    if 'masked_component_idx' not in data.meta:
        print("Error: Graph is missing 'masked_component_idx'. Cannot evaluate.")
        return

    # Handle meta data extracted via DataLoader (wrapped in lists)
    target_comp_idx = data.meta['masked_component_idx']
    if isinstance(target_comp_idx, list): target_comp_idx = target_comp_idx[0]
    
    unit_name = data.meta.get('unit_dir', ['Unknown Unit'])
    if isinstance(unit_name, list): unit_name = unit_name[0]
    
    prediction = get_model_prediction(model, data, target_comp_idx)
    if prediction is None:
        print("Error: No edges found for this component (Isolated Node).")
        return

    ground_truth = get_ground_truth(data, prediction['edge_mask'])
    print_results(unit_name, target_comp_idx, prediction, ground_truth, data, unmasked_comp_x, room_min, room_max)

# --- EXECUTION MODES ---

def load_model(script_dir):
    """Loads model configuration and weights."""
    config_path = os.path.join(script_dir, 'config.json')
    model_path = os.path.join(script_dir, 'checkpoints', 'best.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = ComponentPlacementGNN(
        config['training_params']['component_in_channels'],
        config['training_params']['surface_in_channels'],
        config_path
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model, config


def run_unit_mode(args, model):
    """Handles evaluation for all target components in a raw unit directory."""
    print(f"\nProcessing Raw Unit: {args.unit_dir}")
    norm_graph = build_room_graph(args.unit_dir)
    if norm_graph is None: return
    
    room_min, room_max = load_real_bounds(args.unit_dir)
    print(f"Refrieved Room Bounds for Denormalization:")
    print(f"  Min: {room_min}")
    print(f"  Max: {room_max}")
    
    comp_names = norm_graph.meta['component_names']
    target_indices = [i for i, name in enumerate(comp_names) if name == TARGET_COMPONENT_NAME]
    print(f"Found {len(target_indices)} targets.")
    
    for idx in target_indices:
        print(f"\n{'-'*48}")
        masked_graph = apply_mask_to_graph(norm_graph, idx)
        data = next(iter(DataLoader([masked_graph], batch_size=1)))
        evaluate_graph(model, data, unmasked_comp_x=norm_graph['component'].x, room_min=room_min, room_max=room_max)

# --- MAIN ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--unit_dir', type=str, help="Path to raw unit directory")
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)
    try:
        model, config = load_model(script_dir)
        if args.unit_dir:
            run_unit_mode(args, model)
    except Exception as e:
        print(f"Error during evaluation: {e}")
