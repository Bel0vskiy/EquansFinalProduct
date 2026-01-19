"""
Room Graph Construction Module

This module converts raw room data (3D mesh and JSON metadata) into a heterogeneous graph
suitable for Graph Neural Network (GNN) processing.

Key Functions:
- Parses `mesh.obj` to extract surface patches (walls, floors, ceilings) as 'surface' nodes.
- Parses `data.json` to extract electrical components (sockets, switches) as 'component' nodes.
- Computes geometric features: Centroid, Normal, Bounding Box, and One-Hot encoded types.
- Constructs Edges:
  1. Surface-to-Surface: k-Nearest Neighbors (kNN) links based on centroids.
  2. Component-to-Surface: kNN links from component centroid to surface centroids.
- Assigns Training Labels:
  - Identifies the "Ground Truth" surface for a target component.
  - Generates relative position targets (u, v) for the component on that surface.

Dependencies:
- PyTorch (for graph data structures and kNN operations)
- PyVista (for 3D mesh processing)
- numpy (for numerical operations)
"""
import os
import json
from typing import Optional, Tuple, List

import numpy as np
import pyvista as pv




try:
    import torch
    from torch_geometric.data import HeteroData
except Exception:
    torch = None
    HeteroData = object  

from .component_types import build_component_types, build_name_to_index


def _one_hot(index: Optional[int], size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size:
        vec[index] = 1.0
    return vec


SURFACE_TYPES = ["wall", "floor", "ceiling", "other"]#here only defined 4 surface types

def _classify_surface(normal: np.ndarray, dims: Optional[np.ndarray] = None, centroid: Optional[np.ndarray] = None) -> str:
    # 1. Geometric Override: If it's tall, it's a wall.
    if dims is not None and dims[2] > 0.5:
        return "wall"

    nz = abs(float(normal[2]))
    if nz < 0.2:
        return "wall"
    
    # Vertical Surface (Floor or Ceiling)
    # Use Centroid Height if available (Assuming normalized 0-1 Z range)
    if centroid is not None:
        if centroid[2] > 0.5:
            return "ceiling"
        else:
            return "floor"
            
    # Fallback to Normal direction (Unreliable with PCA)
    if normal[2] > 0:
        return "ceiling" # Outward normal: Up is Ceiling (Top)
    return "floor" # Outward normal: Down is Floor (Bottom)


def _knn_torch(src_points: np.ndarray, dst_points: np.ndarray, k: int):
    """
    Compute k-Nearest Neighbors using PyTorch to avoid sklearn DLL issues on Windows.
    Args:
        src_points: (N, D) array of source points
        dst_points: (M, D) array of destination points (neighbors are picked from here)
        k: Number of neighbors
    Returns:
        dists: (N, k) tensor of distances
        indices: (N, k) tensor of indices in dst_points
    """
    if torch is None:
        raise ImportError("PyTorch is required for graph generation.")
    
    # Ensure inputs are torch tensors
    src_t = torch.from_numpy(src_points).float()
    dst_t = torch.from_numpy(dst_points).float()
    
    # Compute pairwise Euclidean distance
    # cdist handles efficient distance computation
    # shapes: src [N, D], dst [M, D] -> result [N, M]
    dists_matrix = torch.cdist(src_t, dst_t)
    
    # topk returns the k smallest (largest=False) values
    # We need to handle the case where k > M (less candidates than requested neighbors)
    k_valid = min(k, dst_t.shape[0])
    
    knn_dists, knn_indices = torch.topk(dists_matrix, k=k_valid, dim=1, largest=False)
    
    return knn_dists, knn_indices


#compute the features of the surface, including centroid, normal, bbox, and surface type.

def _compute_surface_features(poly: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    pts = poly.points
    centroid = pts.mean(axis=0)
    poly = poly.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    normals = poly.point_normals
    n = normals.mean(axis=0)
    n = n / (np.linalg.norm(n) + 1e-6)
    bounds = poly.bounds
    bbox = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ], dtype=np.float32)
    surf_type = _classify_surface(n, bbox, centroid)
    feat = np.concatenate([
        centroid.astype(np.float32),  # 3
        n.astype(np.float32),         # 3
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
    target_component_name: str,
    data_analysis_dir: Optional[str] = None,
    k_surface: int = 8,
    k_cross: int = 6,
):
    """
    Build a heterogeneous graph for one unit.
    (heterogeneous graph is a graph with different types of nodes and edges)
    here is surface node and component node.

    Nodes:
      - surface: [centroid(3), normal(3), bbox(3), surface_type_one_hot(4)], 
      (centroid is the center of the surface, normal is the normal of the surface, bbox is the bounding box of the surface, 
      surface_type_one_hot is the one-hot encoding of the surface type.)
      - component: [centroid(3), bbox(3), component_type_one_hot(|V|)]

    Edges:
      - (surface)-[near]->(surface): Use kNN on centroids to create edges between surfaces.
      - (component)-[near]->(surface): Use kNN on component centroid to surface centroid, edge_attr=distance to create edges between components and surfaces.

    Labels (if target component exists):
      - y on (component->surface) edges: 1 for the target-to-gt-surface, else 0 (nearest-surface proxy)
      - y_pose: (u, v, yaw) where (u,v) are coordinates in surface tangent frame (yaw=0 placeholder)
    """

    # Build component vocabulary from DataAnalysis CSVs
    if data_analysis_dir is None:
        # Use the DataAnalysis directory in the project root
        data_analysis_dir = "DataAnalysis"
    vocab = build_component_types(data_analysis_dir)
    name_to_idx = build_name_to_index(vocab)
    
    # Debug print
    print(f"Vocabulary size: {len(vocab)}")
    print(f"First 10 vocab items: {vocab[:10]}")
    
    #load the geometry of the unit, including the mesh and the components which is in the json file.
    mesh_path = os.path.join(unit_dir, "mesh.obj")
    json_path = os.path.join(unit_dir, "data.json")
    if not (os.path.exists(mesh_path) and os.path.exists(json_path)):
        return None

    # Load geometry
    mesh = pv.read(mesh_path)
    patches = _split_mesh_into_patches(mesh)
    
    # --- OPTIMIZATION: Merge similar surfaces ---
    # 1. Aggressive vertical merging (Ceiling/Floor layers)
    patches = _merge_vertical_zones(patches)
    # 2. Geometric merging (Coplanar walls)
    patches = _merge_patches(patches)
    print(f"Merged surfaces count: {len(patches)}")
    # --------------------------------------------

    surface_feats: List[np.ndarray] = []
    surface_centroids: List[np.ndarray] = []
    for poly in patches:
        f, c = _compute_surface_features(poly)
        surface_feats.append(f)
        surface_centroids.append(c)
    if not surface_feats:
        return None

    surface_feats_arr = np.vstack(surface_feats).astype(np.float32)
    surface_centroids_arr = np.vstack(surface_centroids).astype(np.float32)

    # Load components from JSON (ET)
    with open(json_path, "r") as f:
        meta = json.load(f)
    et_list = (meta.get("objects", {}).get("ET", []) or [])

    comp_feats: List[np.ndarray] = []
    comp_centroids: List[np.ndarray] = []
    comp_names: List[str] = []
    target_indices: List[int] = []

    for obj in et_list:
        raw = (obj.get("name") or "").strip()
        base = raw.split(":")[0].strip()
        mn = obj.get("min")
        mx = obj.get("max")
        if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3):
            continue
        center, size = _bbox_centroid_and_size(mn, mx)
        type_idx = name_to_idx.get(base, None)
        type_one_hot = _one_hot(type_idx, len(vocab))
        feat = np.concatenate([center.astype(np.float32), size.astype(np.float32), type_one_hot], axis=0)
        
        # Debug print
        print(f"Component: {base}, Type idx: {type_idx}, Feature shape: {feat.shape}")
        
        comp_feats.append(feat)
        comp_centroids.append(center.astype(np.float32))
        comp_names.append(base)
        if base == target_component_name:
            target_indices.append(len(comp_names) - 1)

    if not comp_feats:
        return None

    # Assemble hetero graph
    data = HeteroData()
    data["surface"].x = (torch.from_numpy(surface_feats_arr) if torch else surface_feats_arr)  # type: ignore
    comp_feats_arr = np.vstack(comp_feats).astype(np.float32)
    data["component"].x = (torch.from_numpy(comp_feats_arr) if torch else comp_feats_arr)  # type: ignore

    #use knn to create edges between surfaces.
    # Surface-Surface connections still use Centroid kNN (topology)
    k_s = min(k_surface, len(surface_centroids_arr))
    _, nbrs_s_tensor = _knn_torch(surface_centroids_arr, surface_centroids_arr, k=k_s)
    nbrs_s = nbrs_s_tensor.numpy()
    
    s_src, s_dst = [], []
    for i, nbrs in enumerate(nbrs_s):
        for j in nbrs:
            if i == j:
                continue
            s_src.append(i)
            s_dst.append(j)
    ss_edge = np.stack([np.array(s_src), np.array(s_dst)], axis=0)
    data["surface", "near", "surface"].edge_index = (
        torch.from_numpy(ss_edge) if torch else ss_edge  # type: ignore
    )

    # --- OPTIMIZATION: Component-Surface Edges via Exact Mesh Distance ---
    # Instead of using Centroid-to-Centroid distance (which loses spatial info for large merged surfaces),
    # we compute the exact distance from the component centroid to the nearest point on the surface mesh.
    
    comp_pos = np.vstack(comp_centroids).astype(np.float32)
    cs_src, cs_dst, cs_dist = [], [], []
    
    k_c = min(k_cross, len(patches))
    
    for c_idx, c_pos in enumerate(comp_pos):
        # Calculate distances to all surfaces
        dists = []
        for s_idx, patch in enumerate(patches):
            # Find closest point on the mesh
            try:
                closest_pt_idx = patch.find_closest_point(c_pos)
                closest_pt = patch.points[closest_pt_idx]
                d = np.linalg.norm(c_pos - closest_pt)
            except Exception:
                # Fallback to centroid if mesh query fails
                d = np.linalg.norm(c_pos - surface_centroids_arr[s_idx])
            dists.append((d, s_idx))
        
        # Select Top K closest
        dists.sort(key=lambda x: x[0])
        top_k = dists[:k_c]
        
        for d, s_idx in top_k:
            cs_src.append(c_idx)
            cs_dst.append(s_idx)
            cs_dist.append(d)

    cs_edge = np.stack([np.array(cs_src), np.array(cs_dst)], axis=0)
    data["component", "near", "surface"].edge_index = (
        torch.from_numpy(cs_edge) if torch else cs_edge  # type: ignore
    )
    data["component", "near", "surface"].edge_attr = (
        torch.from_numpy(np.expand_dims(np.array(cs_dist, dtype=np.float32), 1))
        if torch
        else np.expand_dims(np.array(cs_dist, dtype=np.float32), 1)  # type: ignore
    )
    
    # Add reverse edges
    data["surface", "near", "component"].edge_index = (
        torch.from_numpy(cs_edge[[1, 0]]) if torch else cs_edge[[1, 0]]  # type: ignore
    )
    data["surface", "near", "component"].edge_attr = (
        torch.from_numpy(np.expand_dims(np.array(cs_dist, dtype=np.float32), 1))
        if torch
        else np.expand_dims(np.array(cs_dist, dtype=np.float32), 1)  # type: ignore
    )

    # Supervision for the first target instance (if any)
    if target_indices:
        t_idx = target_indices[0]
        
        # Use the already computed geometric connection if possible?
        # For supervision, we want the SINGLE best Ground Truth.
        # We can reuse the logic: AABB check + Plane Distance.
        # OR just use the nearest mesh distance we just computed?
        # Geometric Ground Truth Check is stricter (AABB containment).
        
        # ... (Keep existing GT logic or simplify?)
        # Existing logic is good: Checks AABB containment first.
        
        target_pos_vec = comp_pos[t_idx]
        best_surface_idx = -1
        min_plane_dist = float('inf')
        min_centroid_dist = float('inf')
        closest_centroid_idx = -1

        for i, poly in enumerate(patches):
            # 0. Centroid fallback
            c_dist = np.linalg.norm(surface_centroids_arr[i] - target_pos_vec)
            if c_dist < min_centroid_dist:
                min_centroid_dist = c_dist
                closest_centroid_idx = i

            # 1. AABB Check
            bounds = poly.bounds 
            margin = 0.1 
            
            if (target_pos_vec[0] >= bounds[0] - margin and target_pos_vec[0] <= bounds[1] + margin and
                target_pos_vec[1] >= bounds[2] - margin and target_pos_vec[1] <= bounds[3] + margin and
                target_pos_vec[2] >= bounds[4] - margin and target_pos_vec[2] <= bounds[5] + margin):
                
                # 2. Plane Distance (Using Features)
                surf_feat = surface_feats_arr[i]
                centroid = surf_feat[0:3]
                normal = surf_feat[3:6]
                plane_dist = abs(np.dot(target_pos_vec - centroid, normal))
                
                if plane_dist < min_plane_dist:
                    min_plane_dist = plane_dist
                    best_surface_idx = i
        
        if best_surface_idx != -1:
            gt_surface = int(best_surface_idx)
        else:
            # Fallback: Closest Point on Mesh (Better than centroid)
            min_mesh_dist = float('inf')
            best_mesh_idx = -1
            for s_idx, patch in enumerate(patches):
                 try:
                    closest_pt_idx = patch.find_closest_point(target_pos_vec)
                    closest_pt = patch.points[closest_pt_idx]
                    d = np.linalg.norm(target_pos_vec - closest_pt)
                    if d < min_mesh_dist:
                        min_mesh_dist = d
                        best_mesh_idx = s_idx
                 except: pass
            gt_surface = int(best_mesh_idx if best_mesh_idx != -1 else closest_centroid_idx)

        e_idx = data["component", "near", "surface"].edge_index
        if torch:
            y = torch.zeros(e_idx.shape[1], dtype=torch.float32)
            mask = (e_idx[0] == t_idx) & (e_idx[1] == gt_surface)
            y[mask.nonzero(as_tuple=False).view(-1)] = 1.0
        else:
            y = np.zeros(e_idx.shape[1], dtype=np.float32)
            for k in range(e_idx.shape[1]):
                if int(e_idx[0, k]) == t_idx and int(e_idx[1, k]) == gt_surface:
                    y[k] = 1.0
        data["component", "near", "surface"].y = y

        # Pose target (u, v)
        surf_feat = surface_feats_arr[gt_surface]
        n = surf_feat[3:6]
        # ... (rest of pose code) ...
        n = n / (np.linalg.norm(n) + 1e-6)
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(u, n))) > 0.9:
            u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = u - np.dot(u, n) * n
        u = u / (np.linalg.norm(u) + 1e-6)
        v = np.cross(n, u)
        rel = comp_pos[t_idx] - surface_centroids_arr[gt_surface]
        u_coord = float(np.dot(rel, u))
        v_coord = float(np.dot(rel, v))
        y_pose = np.array([u_coord, v_coord], dtype=np.float32)
        data.y_pose = torch.from_numpy(y_pose) if torch else y_pose  # type: ignore
        data.target_component_index = int(t_idx)
        data.target_surface_index = int(gt_surface)

    data.meta = {
        "component_vocab_size": len(vocab),
        "target_component_name": target_component_name,
        "unit_dir": unit_dir,
    }

    return data



try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

def _compute_normal(poly: pv.PolyData) -> np.ndarray:
    """
    Computes the normal of the surface. 
    Uses PCA to find the direction of least variance (thickness), 
    which is robust for double-sided thin walls.
    Fallback to mean point normal if PCA fails or scikit-learn is missing.
    """
    if PCA is not None and poly.n_points >= 3:
        try:
            pts = poly.points
            pca = PCA(n_components=3)
            pca.fit(pts)
            # The component with the smallest variance (index 2) is the normal direction
            n = pca.components_[2]
            return n / (np.linalg.norm(n) + 1e-6)
        except Exception:
            pass  # Fallback
            
    # Fallback to PyVista mean normal
    poly = poly.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    n = poly.point_normals.mean(axis=0)
    return n / (np.linalg.norm(n) + 1e-6)


def _merge_vertical_zones(patches: List[pv.PolyData]) -> List[pv.PolyData]:
    """
    Heuristically merges surfaces that are likely 'Ceiling' into one node,
    and 'Floor' into another, based on Z-position and vertical bounds.
    This handles double-sided/noisy geometry often found in architectural meshes.
    """
    if not patches:
        return []

    # 1. Compute Global Bounds
    # Bbox of all patches
    global_min = np.array([float('inf'), float('inf'), float('inf')])
    global_max = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    for p in patches:
        b = p.bounds
        global_min = np.minimum(global_min, [b[0], b[2], b[4]])
        global_max = np.maximum(global_max, [b[1], b[3], b[5]])
        
    z_min = global_min[2]
    z_max = global_max[2]
    z_height = z_max - z_min
    if z_height < 0.1: # Flat graph? Check X/Y? Assuming Z is up.
        z_height = 1.0 
        
    # Thresholds (Top 20%, Bottom 10%)
    ceiling_thresh = z_max - 0.2 * z_height
    floor_thresh = z_min + 0.1 * z_height
    
    ceiling_candidates = []
    floor_candidates = []
    others = []
    
    for p in patches:
        # Check simple vertical bounds similarity (Horizontal Plate check)
        b = p.bounds
        z_thickness = b[5] - b[4]
        centroid_z = (b[5] + b[4]) / 2.0
        
        # Criteria:
        # 1. Must be reasonably flat (<20cm thick) OR < 5cm thick
        # 2. Must be in the zone
        
        is_flat = z_thickness < 0.2
        
        if is_flat and centroid_z > ceiling_thresh:
            ceiling_candidates.append(p)
        elif is_flat and centroid_z < floor_thresh:
            floor_candidates.append(p)
        else:
            others.append(p)
            
    # Merge candidates
    final_patches = []
    
    if ceiling_candidates:
        print(f"Aggressive Merge: Squeezing {len(ceiling_candidates)} ceiling surfaces.")
        merged_c = ceiling_candidates[0]
        for c in ceiling_candidates[1:]:
            merged_c = merged_c + c
        final_patches.append(merged_c)
        
    if floor_candidates:
        print(f"Aggressive Merge: Squeezing {len(floor_candidates)} floor surfaces.")
        merged_f = floor_candidates[0]
        for f in floor_candidates[1:]:
            merged_f = merged_f + f
        final_patches.append(merged_f)
        
    final_patches.extend(others)
    
    return final_patches


def _merge_patches(patches: List[pv.PolyData]) -> List[pv.PolyData]:
    """
    Iteratively merges patches that are coplanar. 
    Adjacency check is REMOVED to allow merging of fragmented wall segments (e.g. separated by windows).
    """
    merged = True
    pass_num = 0
    while merged:
        merged = False
        pass_num += 1
        n = len(patches)
        if n < 2:
            break
            
        print(f"Merge Pass {pass_num}: {n} surfaces")
        
        # Precompute features for current set
        feats = []
        for i in range(n):
            norm = _compute_normal(patches[i])
            cent = patches[i].points.mean(axis=0)
            # Plane constant d: n.x + d = 0  => d = -dot(n, p)
            # We use the centroid as the point p.
            d_const = -np.dot(norm, cent)
            feats.append((norm, d_const))
            
        new_patches = []
        skip_indices = set()
        
        for i in range(n):
            if i in skip_indices:
                continue
                
            current_patch = patches[i]
            n1, d1 = feats[i]
            
            # Look for a merge candidate
            best_match = -1
            
            for j in range(i + 1, n):
                if j in skip_indices:
                    continue
                    
                n2, d2 = feats[j]
                
                # 1. Normal Check (Coplanar orientation)
                # Check BOTH directions (normal can be flipped)
                dot_prod = np.dot(n1, n2)
                
                # If normals are opposite, plane constant d changes sign?
                # n.x + d = 0
                # -n.x - d = 0
                # So if n2 ~ -n1, then d2 should be ~ -d1.
                
                is_parallel = dot_prod > 0.95
                is_anti_parallel = dot_prod < -0.95
                
                if not (is_parallel or is_anti_parallel):
                    continue
                    
                # 2. Plane Constant Check (Same infinite plane)
                # If parallel: d1 should be close to d2
                # If antiparallel: d1 should be close to -d2
                
                if is_parallel:
                    dist_diff = abs(d1 - d2)
                else:
                    dist_diff = abs(d1 - (-d2))
                    
                if dist_diff > 0.15: # 15cm tolerance
                    continue
                    
                # 3. Adjacency Check (Re-added)
                # Ensure surfaces are spatially close using AABB overlap with margin
                b1 = current_patch.bounds
                b2 = patches[j].bounds
                margin = 0.2 # 20cm gap tolerance
                
                # Check for intersection of expanded bounding boxes
                # (min1 - margin) <= (max2 + margin)  AND  (min2 - margin) <= (max1 + margin)
                # Rewritten: max1 + margin >= min2 - margin  => max1 - min2 >= -2*margin
                # Overlap condition: max1 >= min2 - 2*margin AND max2 >= min1 - 2*margin
                
                gap_x = max(b1[0], b2[0]) - min(b1[1], b2[1]) # Positive means gap, Negative means overlap
                gap_y = max(b1[2], b2[2]) - min(b1[3], b2[3])
                gap_z = max(b1[4], b2[4]) - min(b1[5], b2[5])
                
                # Check if gap is smaller than margin (allowing small separation)
                # Wait, gap calculation above is for Intersection = [max(min), min(max)]
                # Len = min(max) - max(min). If len > 0 -> Overlap.
                # If len < 0 -> Gap is -len.
                
                # Easier: Check if distance between intervals is > margin
                def distinct_intervals(min1, max1, min2, max2, tol):
                    # Returns True if intervals are separated by MORE than tol
                    return (min1 > max2 + tol) or (min2 > max1 + tol)

                dx = distinct_intervals(b1[0], b1[1], b2[0], b2[1], margin)
                dy = distinct_intervals(b1[2], b1[3], b2[2], b2[3], margin)
                dz = distinct_intervals(b1[4], b1[5], b2[4], b2[5], margin)
                
                if dx or dy or dz:
                     # Disjoint in at least one dimension
                     continue
                
                # Refined Adjacency: Precise Mesh Distance Check
                # AABB overlap is not enough for diagonal walls.
                # We check the distance from points on P1 to Mesh P2.
                # If the minimum distance is large, they are disjoint.
                
                # Sampling points for speed (e.g., up to 50 points)
                n_pts = current_patch.n_points
                step = max(1, n_pts // 50)
                sample_indices = range(0, n_pts, step)
                
                min_dist = float('inf')
                
                # Check distance from P1 samples to P2
                for idx in sample_indices:
                    pt = current_patch.points[idx]
                    try:
                        # find_closest_point returns index of closest vertex in P2
                        closest_idx = patches[j].find_closest_point(pt)
                        closest_pt = patches[j].points[closest_idx]
                        d = np.linalg.norm(pt - closest_pt)
                        if d < min_dist:
                            min_dist = d
                        if min_dist < margin:
                            break # Found a connection!
                    except: pass
                
                if min_dist > margin:
                    # Double check P2 to P1 (asymmetric sampling)
                    n_pts2 = patches[j].n_points
                    step2 = max(1, n_pts2 // 50)
                    sample_indices2 = range(0, n_pts2, step2)
                    min_dist2 = float('inf')
                    
                    for idx in sample_indices2:
                        pt = patches[j].points[idx]
                        try:
                            closest_idx = current_patch.find_closest_point(pt)
                            closest_pt = current_patch.points[closest_idx]
                            d = np.linalg.norm(pt - closest_pt)
                            if d < min_dist2:
                                min_dist2 = d
                            if min_dist2 < margin:
                                break
                        except: pass
                    
                    if min_dist2 > margin:
                        continue # Truly disjoint
                
                print(f"  MATCH FOUND: {i} + {j} (Dot: {dot_prod:.3f}, D_diff: {dist_diff:.3f})")
                best_match = j
                break
            
            if best_match != -1:
                # Merge
                merged_mesh = current_patch + patches[best_match]
                new_patches.append(merged_mesh)
                skip_indices.add(best_match)
                merged = True 
            else:
                new_patches.append(current_patch)
                
        patches = new_patches
        
    return patches
