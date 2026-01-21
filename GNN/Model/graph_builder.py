import os
import sys
import json
from typing import Optional, Tuple, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

from Data.component_types import build_component_types, build_name_to_index

    #nodes- surface (floor/wall/ceiling), component (MEP object).
    #edges- adjacent (surface-surface), near (component-component), on (component-surface).

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


SURFACE_TYPES = ["wall", "floor", "ceiling", "other"]

def _classify_surface(normal: np.ndarray, dims: Optional[np.ndarray] = None) -> str:
    if dims is not None and dims[2] > 0.5:
        return "wall"

    nz = abs(float(normal[2]))
    if nz < 0.2:
        return "wall"
    if normal[2] > 0:
        return "ceiling"
    return "floor"


def _compute_surface_features(poly: pv.PolyData) -> Tuple[np.ndarray, np.ndarray]:
    pts = poly.points
    centroid = pts.mean(axis=0)
    
    bounds = poly.bounds
    bbox = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ], dtype=np.float32)

    
    min_dim_idx = np.argmin(bbox)
    n = np.zeros(3, dtype=np.float32)
    n[min_dim_idx] = 1.0

    surf_type = _classify_surface(n, bbox)
    
    feat = np.concatenate([
        centroid.astype(np.float32),
        n.astype(np.float32),
        bbox.astype(np.float32),
        _one_hot(SURFACE_TYPES.index(surf_type), len(SURFACE_TYPES)),
    ], axis=0)
    
    return feat, centroid.astype(np.float32)

def _bbox_centroid_and_size(min_pt: List[float], max_pt: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    mn = np.array(min_pt, dtype=np.float32)
    mx = np.array(max_pt, dtype=np.float32)
    center = (mn + mx) / 2.0
    size = (mx - mn)
    return center, size


def _split_mesh_into_patches(mesh: pv.DataSet) -> List[pv.PolyData]:
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().clean()
    connectivity = mesh.connectivity()
    if "RegionId" in connectivity.array_names:
        regions = np.unique(connectivity["RegionId"])
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

    if torch is None:
        raise ImportError("PyTorch is required for this function.")

    if data_analysis_dir is None:
        data_analysis_dir = "Data/DataAnalysis"
    vocab = build_component_types(data_analysis_dir)
    name_to_idx = build_name_to_index(vocab)

    mesh_path = os.path.join(unit_dir, "mesh.obj")
    json_path = os.path.join(unit_dir, "data.json")
    if not (os.path.exists(mesh_path) and os.path.exists(json_path)):
        return None

    mesh = pv.read(mesh_path)
    patches = _split_mesh_into_patches(mesh)


    patches = merge_horizontal_zones(patches)
    patches = _merge_patches(patches)
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

    comp_feats, comp_centroids, comp_names = [], [], []
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
        comp_names.append(base)

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

    data["component", "near", "component"].edge_index = torch.empty((2, 0), dtype=torch.long)

    cs_src, cs_dst, all_y_surface, all_y_pose = [], [], [], []
    num_surfaces = len(surface_feats_arr)

    for i, comp_pos in enumerate(comp_pos_arr):
        best_surface_idx = -1
        min_plane_dist = float('inf')
        
        distances = np.linalg.norm(surface_centroids_arr - comp_pos, axis=1)
        fallback_gt_idx = int(np.argmin(distances))

        for j, poly in enumerate(patches):
            bounds = poly.bounds 
            margin = 0.01 
            
            if (comp_pos[0] >= bounds[0] - margin and comp_pos[0] <= bounds[1] + margin and
                comp_pos[1] >= bounds[2] - margin and comp_pos[1] <= bounds[3] + margin and
                comp_pos[2] >= bounds[4] - margin and comp_pos[2] <= bounds[5] + margin):
                
                if plane_dist < min_plane_dist:
                    min_plane_dist = plane_dist
                    best_surface_idx = j
        
        gt_surface_idx = best_surface_idx if best_surface_idx != -1 else fallback_gt_idx

        for j in range(num_surfaces):
            cs_src.append(i)
            cs_dst.append(j)

            is_true_surface = (j == gt_surface_idx)
            all_y_surface.append(1.0 if is_true_surface else 0.0)

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
            else:
                all_y_pose.append([np.nan, np.nan])

    data["component", "candidate_placement", "surface"].edge_index = torch.from_numpy(np.stack([cs_src, cs_dst])).long()
    data["component", "candidate_placement", "surface"].y_surface = torch.from_numpy(np.array(all_y_surface, dtype=np.float32))
    data["component", "candidate_placement", "surface"].y_pose = torch.from_numpy(np.array(all_y_pose, dtype=np.float32))

    rev_edge_index = torch.from_numpy(np.stack([cs_dst, cs_src])).long()
    data["surface", "rev_candidate_placement", "component"].edge_index = rev_edge_index

    data.meta = {
        "component_vocab_size": len(vocab),
        "unit_dir": unit_dir,
        "component_names": comp_names
    }

    return data



def inspect_graph(data: HeteroData):
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
    
    FLAT_Z_THICKNESS = 0.05 
    
    FLOOR_LIMIT = 0.15
    CEILING_LIMIT = 0.85
    
    MIN_HORIZONTAL_SPAN = 0.5 
    
    ceiling_candidates = []
    floor_candidates = []
    others = []
    
    for p in patches:
        b = p.bounds
        
        x_size = b[1] - b[0]
        y_size = b[3] - b[2]
        z_size = b[5] - b[4]
        centroid_z = (b[5] + b[4]) / 2.0
        
        is_flat_horizontal = z_size < FLAT_Z_THICKNESS
        
        if is_flat_horizontal:
            is_large = (x_size > MIN_HORIZONTAL_SPAN) and (y_size > MIN_HORIZONTAL_SPAN)
            is_floor_zone = centroid_z < FLOOR_LIMIT
            is_ceiling_zone = centroid_z > CEILING_LIMIT
            
            if is_large and is_floor_zone:
                floor_candidates.append(p)
            elif is_large and is_ceiling_zone:
                ceiling_candidates.append(p)
            else:
                pass
        else:
            others.append(p)
            
    final_patches = []
    
    if ceiling_candidates:
        merged = ceiling_candidates[0]
        for c in ceiling_candidates[1:]: merged += c
        final_patches.append(merged)
        
    if floor_candidates:
        merged = floor_candidates[0]
        for f in floor_candidates[1:]: merged += f
        final_patches.append(merged)
        
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
                
                dot_prod = np.dot(n1, n2)
                is_parallel = dot_prod > 0.99
                is_anti_parallel = dot_prod < -0.99
                
                if not (is_parallel or is_anti_parallel): continue
                
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
    # remove walls that don't have at least one significant dimension

    if not patches: return []

    MIN_WALL_DIM = 0.6
    VERTICAL_TOL_Z = 0.2

    filtered_patches = []

    for p in patches:
        feat, _ = _compute_surface_features(p)
        normal = feat[3:6]
        is_vertical = abs(normal[2]) < VERTICAL_TOL_Z

        if is_vertical:
            b = p.bounds
            x_dim = b[1] - b[0]
            y_dim = b[3] - b[2]
            z_dim = b[5] - b[4] 


            is_significant = max(x_dim, y_dim, z_dim) > MIN_WALL_DIM
            
            if is_significant:
                filtered_patches.append(p)
        else:
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

