import os
import sys
import json
from typing import Optional, Tuple, List

# Add project root to path to allow absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

from Data.component_types import build_component_types, build_name_to_index

try:
    import torch
    from torch_geometric.data import HeteroData
except Exception:
    torch = None
    HeteroData = object  



def _one_hot(index: Optional[int], size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        vec[index] = 1.0
    return vec


SURFACE_TYPES = ["wall", "floor", "ceiling", "other"]#here only defined 4 surface types

def _classify_surface(normal: np.ndarray, dims: Optional[np.ndarray] = None) -> str:
    if dims is not None and dims[2] > 0.5:
        return "wall"

    nz = abs(float(normal[2]))
    if nz < 0.2:
        return "wall"
    if normal[2] > 0:
        return "ceiling"
    return "floor"


#compute the features of the surface, including centroid, normal, bbox, and surface type.

def _compute_surface_features(poly: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    pts = poly.points
    centroid = pts.mean(axis=0)
    
    # 1. Get Bounding Box Dimensions
    bounds = poly.bounds
    bbox = np.array([
        bounds[1] - bounds[0], # Size X
        bounds[3] - bounds[2], # Size Y
        bounds[5] - bounds[4], # Size Z
    ], dtype=np.float32)

    # 2. FIX: Determine Normal from Geometry (Thinnest Dimension)
    # Instead of averaging normals (which cancels out for closed boxes),
    # we assume the Normal is parallel to the smallest dimension (Thickness).
    
    min_dim_idx = np.argmin(bbox)
    n = np.zeros(3, dtype=np.float32)
    n[min_dim_idx] = 1.0 # e.g. if X is smallest, Normal is [1, 0, 0]

    # Optional: If you really want the sign (+/-) from the mesh, you can probe it.
    # But for graph learning, [1,0,0] vs [-1,0,0] usually doesn't matter 
    # as long as it's consistent. We stick to positive axes for stability.

    # 3. Classify Surface
    surf_type = _classify_surface(n, bbox)
    
    feat = np.concatenate([
        centroid.astype(np.float32),  # 3
        n.astype(np.float32),         # 3 (Perfectly snapped unit vector)
        bbox.astype(np.float32),      # 3
        _one_hot(SURFACE_TYPES.index(surf_type), len(SURFACE_TYPES)),  # 4
    ], axis=0)
    
    return feat, centroid.astype(np.float32)

def _bbox_centroid_and_size(min_pt: List[float], max_pt: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    mn = np.array(min_pt, dtype=np.float32)
    mx = np.array(max_pt, dtype=np.float32)
    center = (mn + mx) / 2.0
    size = (mx - mn)
    return center, size


def _split_mesh_into_patches(mesh: pv.DataSet) -> List[pv.PolyData]:
    # Ensure surface polydata
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().clean()
    # Prefer connectivity-based regions; robust across your meshes
    connectivity = mesh.connectivity()
    if "RegionId" in connectivity.array_names:
        regions = np.unique(connectivity["RegionId"])  # type: ignore
        patches: List[pv.PolyData] = []
        for rid in regions:
            sub = connectivity.threshold([rid - 0.1, rid + 0.1], scalars="RegionId")
            if sub.n_points > 0:
                patches.append(sub.extract_surface().clean())
        return patches
    return [mesh]


def build_room_graph(
    unit_dir: str,
    data_analysis_dir: Optional[str] = None,
    k_surface_neighbors: int = 8,

):
    """
    Build a heterogeneous graph for one unit based on the V2 approach.
    All components will have a '(component, on, surface)' edge for potential supervision.

    Nodes:
      - surface: [centroid(3), normal(3), bbox(3), surface_type_one_hot(4)]
      - component: [centroid(3), bbox(3), component_type_one_hot(|V|)]

    Edges:
      - (surface)-[adjacent]->(surface): kNN on surface centroids.
      - (component)-[near]->(component): kNN on component centroids.
      - (component)-[on]->(surface): Connects each component to its host surface.
        This edge holds the ground truth label for training.

    Labels:
      - y_pose on (component, on, surface) edge: (u, v) coordinates in surface tangent frame.
    """
    if torch is None:
        raise ImportError("PyTorch is required for this function.")

    if data_analysis_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # GNN root
        data_analysis_dir = os.path.join(base_dir, "Data", "DataAnalysis")
    vocab = build_component_types(data_analysis_dir)
    name_to_idx = build_name_to_index(vocab)

    mesh_path = os.path.join(unit_dir, "mesh.obj")
    json_path = os.path.join(unit_dir, "data.json")
    if not (os.path.exists(mesh_path) and os.path.exists(json_path)):
        return None

    mesh = pv.read(mesh_path)
    patches = _split_mesh_into_patches(mesh)

    # 1. Merge the floor and ceiling fragments
    patches = merge_horizontal_zones(patches)
    # 2. Merge coplanar wall fragments (even if separated by a door/window)
    patches = _merge_patches(patches)
    # 3. Filter out small wall patches that are likely not structural
    patches = filter_small_walls(patches)


    surface_feats, surface_centroids = [], []
    for poly in patches:
        f, c = _compute_surface_features(poly)
        surface_feats.append(f)
        surface_centroids.append(c)
    if not surface_feats:
        return None

    surface_feats_arr = np.vstack(surface_feats).astype(np.float32)
    surface_centroids_arr = np.vstack(surface_centroids).astype(np.float32)

    print("\n--- Surface Bounding Boxes ---")
    for idx, poly in enumerate(patches):
        bounds = poly.bounds
        print(f"Surface {idx}: Bounds = [{bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f}, {bounds[4]:.3f}, {bounds[5]:.3f}]")
    print("----------------------------")

    with open(json_path, "r") as f:
        meta = json.load(f)
    et_list = (meta.get("objects", {}).get("ET", []) or [])

    comp_feats, comp_centroids, comp_names = [], [], [] # Capture comp_names
    for obj in et_list:
        raw = (obj.get("name") or "").strip()
        base = raw.split(":")[0].strip()
        mn, mx = obj.get("min"), obj.get("max")
        if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3):
            continue
        center, size = _bbox_centroid_and_size(mn, mx)
        type_idx = name_to_idx.get(base)
        type_one_hot = _one_hot(type_idx, len(vocab))
        feat = np.concatenate([center.astype(np.float32), size.astype(np.float32), type_one_hot], axis=0)
        comp_feats.append(feat)
        comp_centroids.append(center.astype(np.float32))
        comp_names.append(base) # Store component name

    if not comp_feats:
        return None

    comp_feats_arr = np.vstack(comp_feats).astype(np.float32)
    comp_pos_arr = np.vstack(comp_centroids).astype(np.float32)

    data = HeteroData()
    data["surface"].x = torch.from_numpy(surface_feats_arr)
    data["component"].x = torch.from_numpy(comp_feats_arr)

    k_s = min(k_surface_neighbors, len(surface_centroids_arr))
    if k_s > 0:
        knn_s = NearestNeighbors(n_neighbors=k_s).fit(surface_centroids_arr)
        _, nbrs_s = knn_s.kneighbors(surface_centroids_arr)
        s_src, s_dst = [], []
        for i, nbrs in enumerate(nbrs_s):
            for j in nbrs:
                if i != j:
                    s_src.append(i)
                    s_dst.append(j)
        if s_src:
            data["surface", "adjacent", "surface"].edge_index = torch.from_numpy(np.stack([s_src, s_dst])).long()

    # Original k-NN based component-to-component edges (commented out for testing)
    # if len(comp_pos_arr) > 1:
    #     k_c = min(k_component_neighbors, len(comp_pos_arr))
    #     if k_c > 0:
    #         knn_c = NearestNeighbors(n_neighbors=k_c).fit(comp_pos_arr)
    #         _, nbrs_c = knn_c.kneighbors(comp_pos_arr)
    #         c_src, c_dst = [], []
    #         for i, nbrs in enumerate(nbrs_c):
    #             for j in nbrs:
    #                 if i != j:
    #                     c_src.append(i)
    #                     c_dst.append(j)
    #         if c_src:
    #             data["component", "near", "component"].edge_index = torch.from_numpy(np.stack([c_src, c_dst])).long()

    # --- (component)-[near]->(component) edges: Temporarily set to empty ---
    # This is for testing without direct component-to-component connections.
    data["component", "near", "component"].edge_index = torch.empty((2, 0), dtype=torch.long)


    # (component)-[candidate_placement]->(surface) supervision edges
    cs_src, cs_dst, all_y_surface, all_y_pose = [], [], [], []
    num_surfaces = len(surface_feats_arr)

    for i, comp_pos in enumerate(comp_pos_arr):
        # First, find the actual ground truth surface (the nearest one)
        # --- START OF MARGIN FUNCTIONALITY ---
        best_surface_idx = -1
        min_plane_dist = float('inf')
        
        # We also keep a centroid fallback in case AABB check fails
        distances = np.linalg.norm(surface_centroids_arr - comp_pos, axis=1)
        fallback_gt_idx = int(np.argmin(distances))

        for j, poly in enumerate(patches):
            # 1. AABB Check with 0.01 margin (in normalized space)
            bounds = poly.bounds 
            margin = 0.01 
            
            # Check if component position is within the surface's bounding box + margin
            if (comp_pos[0] >= bounds[0] - margin and comp_pos[0] <= bounds[1] + margin and
                comp_pos[1] >= bounds[2] - margin and comp_pos[1] <= bounds[3] + margin and
                comp_pos[2] >= bounds[4] - margin and comp_pos[2] <= bounds[5] + margin):
                
                # 2. Within those candidates, pick the one closest to the infinite plane
                surf_feat = surface_feats_arr[j]
                centroid = surf_feat[0:3]
                normal = surf_feat[3:6]
                plane_dist = abs(np.dot(comp_pos - centroid, normal))
                
                if plane_dist < min_plane_dist:
                    min_plane_dist = plane_dist
                    best_surface_idx = j
        
        # If AABB check found a surface, use it; otherwise, use the nearest centroid
        gt_surface_idx = best_surface_idx if best_surface_idx != -1 else fallback_gt_idx
        # --- END OF MARGIN FUNCTIONALITY ---

        for j in range(num_surfaces):
            # Add an edge from component i to surface j
            cs_src.append(i)
            cs_dst.append(j)

            is_true_surface = (j == gt_surface_idx)
            all_y_surface.append(1.0 if is_true_surface else 0.0)

            # Only compute the detailed pose for the true surface.
            if is_true_surface:
                surf_feat = surface_feats_arr[j]
                n = surf_feat[3:6]
                n /= (np.linalg.norm(n) + 1e-6)
                
                u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                if abs(np.dot(u, n)) > 0.9:
                    u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                
                u = u - np.dot(u, n) * n
                u /= (np.linalg.norm(u) + 1e-6)
                v = np.cross(n, u)
                
                rel_pos = comp_pos - surface_centroids_arr[j]
                u_coord = np.dot(rel_pos, u)
                v_coord = np.dot(rel_pos, v)
                all_y_pose.append([u_coord, v_coord])
                print(f"Component {i} on Surface {j}: u={u_coord:.3f}, v={v_coord:.3f}")
                reconstructed_pos = surface_centroids_arr[j] + (u_coord * u) + (v_coord * v)
                
                # error_dist = np.linalg.norm(reconstructed_pos - comp_pos)
                
                # print(f"  > Original Pos:      {np.array2string(comp_pos, separator=', ', formatter={'float_kind':lambda x: '%.3f' % x})}")
                # print(f"  > Reconstructed Pos: {np.array2string(reconstructed_pos, separator=', ', formatter={'float_kind':lambda x: '%.3f' % x})}")
                # print(f"  > Surface Centroid:  {np.array2string(surface_centroids_arr[j], separator=', ', formatter={'float_kind':lambda x: '%.3f' % x})}")
                # print(f"  > Reconstruction Error: {error_dist:.6f}")
                
                # # If error is large (e.g., > 0.01 in normalized space), something is wrong
                # if error_dist > 1e-4:
                #     print(f"  ❌ WARNING: Large Reconstruction Error! Centroid or Basis Vector mismatch.")
                #     # Optional: Print basis vectors to debug
                #     print(f"     Normal: {np.array2string(n, separator=', ', formatter={'float_kind':lambda x: '%.3f' % x})}")
                #     print(f"     Basis U: {np.array2string(u, separator=', ', formatter={'float_kind':lambda x: '%.3f' % x})}")
                #     print(f"     Basis V: {np.array2string(v, separator=', ', formatter={'float_kind':lambda x: '%.3f' % x})}")
                # else:
                #     print(f"  ✅ Validation Passed: Math and Centroid are consistent.")
      
                
            else:
                # For candidate edges that are not the ground truth, pose is irrelevant.
                # We can use NaNs to signal this. The training script already handles NaNs.
                all_y_pose.append([np.nan, np.nan])

    data["component", "candidate_placement", "surface"].edge_index = torch.from_numpy(np.stack([cs_src, cs_dst])).long()
    data["component", "candidate_placement", "surface"].y_surface = torch.from_numpy(np.array(all_y_surface, dtype=np.float32))
    data["component", "candidate_placement", "surface"].y_pose = torch.from_numpy(np.array(all_y_pose, dtype=np.float32))

    rev_edge_index = torch.from_numpy(np.stack([cs_dst, cs_src])).long()
    data["surface", "rev_candidate_placement", "component"].edge_index = rev_edge_index

    data.meta = {
        "component_vocab_size": len(vocab),
        "unit_dir": unit_dir,
        "component_names": comp_names # Store component names for filtering in train.py
    }

    return data



def inspect_graph(data: HeteroData):
    """
    Logs information about a HeteroData graph object to the console.
    """
    if data is None:
        print("Graph data is None.")
        return
        
    if not isinstance(data, HeteroData):
        print("Object is not a PyTorch Geometric HeteroData instance.")
        return

    print("--- Graph Inspection ---")
    print("Node Types:", list(data.node_types))
    for node_type in data.node_types:
        print(f"  - Node '{node_type}':")
        print(f"    - Number of nodes: {data[node_type].num_nodes}")
        if hasattr(data[node_type], 'x'):
            print(f"    - Feature shape: {data[node_type].x.shape}")

    print("\nEdge Types:", list(data.edge_types))
    for edge_type in data.edge_types:
        print(f"  - Edge '{edge_type}':")
        print(f"    - Number of edges: {data[edge_type].num_edges}")
        if hasattr(data[edge_type], 'edge_index'):
            print(f"    - Edge index shape: {data[edge_type].edge_index.shape}")
        if hasattr(data[edge_type], 'y_pose'):
            print(f"    - y_pose shape: {data[edge_type].y_pose.shape}")
    
    if hasattr(data, 'meta'):
        print("\nMetadata:")
        for key, value in data.meta.items():
                print(f"  - {key}: {value}")
            
    print("--- End of Inspection ---")

def _compute_normal(poly: pv.PolyData) -> np.ndarray:
    poly = poly.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    n = poly.point_normals.mean(axis=0)
    return n / (np.linalg.norm(n) + 1e-6)

def merge_horizontal_zones(patches: List[pv.PolyData]) -> List[pv.PolyData]:
    if not patches: return []
    
    # --- CONSTANTS ---
    # Flatness: How thin is it in Z? 
    # Normalized walls are 1.0 tall, floors are ~0.01 tall.
    FLAT_Z_THICKNESS = 0.05 
    
    # Position: Normalized Z boundaries
    FLOOR_LIMIT = 0.15
    CEILING_LIMIT = 0.85
    
    # Size: Min dimension to be considered a structural floor/ceiling
    # Filters out doormats, shower trays, etc.
    MIN_HORIZONTAL_SPAN = 0.5 
    
    ceiling_candidates = []
    floor_candidates = []
    others = [] # This will now ONLY contain vertical surfaces (Walls)
    
    for p in patches:
        b = p.bounds # [xmin, xmax, ymin, ymax, zmin, zmax]
        
        # Calculate Dimensions
        x_size = b[1] - b[0]
        y_size = b[3] - b[2]
        z_size = b[5] - b[4]
        centroid_z = (b[5] + b[4]) / 2.0
        
        # Check 1: Is it Horizontal (Flat)?
        is_flat_horizontal = z_size < FLAT_Z_THICKNESS
        
        if is_flat_horizontal:
            # It is horizontal. Now decide: Keep or Kill?
            
            # Check 2: Size (Must be large)
            is_large = (x_size > MIN_HORIZONTAL_SPAN) and (y_size > MIN_HORIZONTAL_SPAN)
            
            # Check 3: Position (Must be extrema)
            is_floor_zone = centroid_z < FLOOR_LIMIT
            is_ceiling_zone = centroid_z > CEILING_LIMIT
            
            if is_large and is_floor_zone:
                floor_candidates.append(p)
            elif is_large and is_ceiling_zone:
                ceiling_candidates.append(p)
            else:
                # DISCARD!
                # It's a window sill, a shelf, or a tiny mat.
                # We do nothing, effectively deleting it.
                pass
        else:
            # It has significant Z-height (It is a Wall or slanted surface)
            # We keep all of these for the next step (_merge_patches)
            others.append(p)
            
    # Merge the identified candidates
    final_patches = []
    
    if ceiling_candidates:
        merged = ceiling_candidates[0]
        for c in ceiling_candidates[1:]: merged += c
        final_patches.append(merged)
        
    if floor_candidates:
        merged = floor_candidates[0]
        for f in floor_candidates[1:]: merged += f
        final_patches.append(merged)
        
    # Add back the walls
    final_patches.extend(others)
    
    return final_patches

def _merge_patches(patches: List[pv.PolyData]) -> List[pv.PolyData]:
    merged = True
    while merged:
        merged = False
        n = len(patches)
        if n < 2: break
        
        feats = []
        for i in range(n):
            norm = _compute_normal(patches[i])
            cent = patches[i].points.mean(axis=0)
            d_const = -np.dot(norm, cent)
            feats.append((norm, d_const))
            
        new_patches = []
        skip_indices = set()
        for i in range(n):
            if i in skip_indices: continue
            
            n1, d1 = feats[i]
            best_match = -1
            for j in range(i + 1, n):
                if j in skip_indices: continue
                n2, d2 = feats[j]
                
                # Check if parallel or anti-parallel
                dot_prod = np.dot(n1, n2)
                is_parallel = dot_prod > 0.99 # Stricter for normalized
                is_anti_parallel = dot_prod < -0.99
                
                if not (is_parallel or is_anti_parallel): continue
                
                # Plane Constant Check: Adjusted to 0.01 (1% of room width)
                dist_diff = abs(d1 - d2) if is_parallel else abs(d1 + d2)
                if dist_diff > 0.01: continue 
                
                best_match = j
                break
            
            if best_match != -1:
                new_patches.append(patches[i] + patches[best_match])
                skip_indices.add(best_match)
                merged = True 
            else:
                new_patches.append(patches[i])
        patches = new_patches
    return patches

def filter_small_walls(patches: List[pv.PolyData]) -> List[pv.PolyData]:
    """
    Removes vertical surfaces (walls) that don't have at least one 
    significant dimension (e.g., > 0.6 in normalized space).
    Preserves horizontal surfaces (floors/ceilings) untouched.
    """
    if not patches: return []

    # Rule: A wall must span at least 60% of the room in Height OR Width
    MIN_WALL_DIM = 0.6
    VERTICAL_TOL_Z = 0.2  # Same tolerance used elsewhere

    filtered_patches = []

    for p in patches:
        # 1. Check Orientation
        feat, _ = _compute_surface_features(p)
        normal = feat[3:6]
        is_vertical = abs(normal[2]) < VERTICAL_TOL_Z

        if is_vertical:
            # 2. Check Dimensions
            b = p.bounds
            x_dim = b[1] - b[0]
            y_dim = b[3] - b[2]
            z_dim = b[5] - b[4] # This is Height

            # "At least one dimension is > 0.6"
            # This keeps tall narrow strips (Door Jambs) AND short wide walls (Window aprons)
            # But discards small floating blocks (0.3 x 0.3)
            is_significant = max(x_dim, y_dim, z_dim) > MIN_WALL_DIM
            
            if is_significant:
                filtered_patches.append(p)
            # else: Discard (Too small to be a main wall)
        else:
            # 3. Horizontal Surfaces (Floors/Ceilings)
            # We assume these were already filtered by _merge_vertical_zones
            filtered_patches.append(p)
            
    return filtered_patches

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Build and inspect a room graph from a unit directory.")
    parser.add_argument("--unit_dir", type=str, help="Path to the unit directory containing mesh.obj and data.json")

    args = parser.parse_args()

    print(f"Building graph for unit: {args.unit_dir}")
    graph_data = build_room_graph(args.unit_dir)

    if graph_data:
        inspect_graph(graph_data)
    else:
        print(f"Could not build graph for unit: {args.unit_dir}. Check if the directory and files are correct.")

