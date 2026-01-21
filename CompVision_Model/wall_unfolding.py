import os
import numpy as np
import pyvista as pv
import cv2
from sklearn.cluster import DBSCAN
from typing import List, Dict, Tuple, Optional


class WallView:
    def __init__(self, wall_id: int, origin: np.ndarray,
                 u_axis: np.ndarray, v_axis: np.ndarray, normal: np.ndarray,
                 u_range: Tuple[float, float], v_range: Tuple[float, float],
                 meshes: List[pv.PolyData]):
        self.wall_id = wall_id
        self.origin = origin
        self.u_axis = u_axis
        self.v_axis = v_axis
        self.normal = normal
        self.u_range = u_range
        self.v_range = v_range
        self.meshes = meshes
        self.image_tensor = None
        self.label_mask = None

    def world_to_uv(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.origin
        u = np.dot(rel, self.u_axis)
        v = np.dot(rel, self.v_axis)
        return np.column_stack((u, v))

    def uv_to_pixel(self, uv_points: np.ndarray, img_size: int = 256) -> np.ndarray:
        u = uv_points[:, 0]
        v = uv_points[:, 1]
        u_min, u_max = self.u_range
        v_min, v_max = self.v_range

        width = max(u_max - u_min, 1.0)
        height = max(v_max - v_min, 1.0)

        u_norm = (u - u_min) / width
        v_norm = (v - v_min) / height

        x = u_norm * (img_size - 1)
        y = (1 - v_norm) * (img_size - 1)
        return np.column_stack((x, y)).astype(int)


class UnfoldingEngine:
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.global_up = np.array([0, 0, 1], dtype=np.float32)

    def _get_dominant_normal(self, mesh: pv.DataSet) -> np.ndarray:
        if mesh.n_cells == 0: return np.array([0, 0, 1])
        sized = mesh.compute_cell_sizes(length=False, volume=False)
        areas = sized.cell_data["Area"]
        max_idx = np.argmax(areas)
        return mesh.cell_data["Normals"][max_idx]

    def _cluster_walls_robust(self, mesh: pv.DataSet) -> List[WallView]:
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()

        # UNIT NORMALIZATION
        # Detect if Meters or MM. If max bound < 500, it's Meters.
        bounds = mesh.bounds
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        scale_factor = 1.0
        if max_dim < 500.0:
            print(f"  [Info] Detected Meters (Max Dim: {max_dim:.2f}). Scaling to MM.")
            scale_factor = 1000.0
            mesh.points *= scale_factor

        # PRE-PROCESSING
        # Merge vertices close to each other (1cm tolerance)
        mesh = mesh.clean(point_merging=True, tolerance=10.0)
        mesh = mesh.compute_normals(cell_normals=True, point_normals=False,
                                    auto_orient_normals=True, consistent_normals=True)

        # Filter Vertical Faces
        normals = mesh.cell_data["Normals"]
        is_vertical = np.abs(normals[:, 2]) < 0.2  # Strict vertical check

        # Add array for filtering
        mesh.cell_data["IsVertical"] = np.where(is_vertical, 1, 0)
        walls_only = mesh.threshold(value=[0.9, 1.1], scalars="IsVertical", preference="cell")

        if walls_only.n_cells == 0: return []

        # CLUSTER BY ORIENTATION
        # DBSCAN on Normal Vectors (X, Y)
        vertical_normals = walls_only.cell_data["Normals"][:, 0:2]
        clustering_dir = DBSCAN(eps=0.1, min_samples=1).fit(vertical_normals)

        # Assign labels to the walls_only subset
        walls_only.cell_data["DirID"] = clustering_dir.labels_
        unique_dirs = np.unique(clustering_dir.labels_)

        wall_views = []
        wall_id_counter = 0

        for dir_id in unique_dirs:
            if dir_id == -1: continue  # Skip noise

            # Get all faces facing this way
            dir_group = walls_only.threshold([dir_id - 0.1, dir_id + 0.1], scalars="DirID", preference="cell")
            if dir_group.n_cells == 0: continue

            # Reference Normal
            ref_normal = self._get_dominant_normal(dir_group)

            # CLUSTER BY DEPTH (THICKNESS)
            # Project centers onto the normal vector
            centers = dir_group.cell_centers().points
            depths = np.dot(centers, ref_normal).reshape(-1, 1)

            # Use loose epsilon (300mm = 30cm) to group front/back/insulation layers
            clustering_depth = DBSCAN(eps=300.0, min_samples=1).fit(depths)

            dir_group.cell_data["DepthID"] = clustering_depth.labels_
            unique_depths = np.unique(clustering_depth.labels_)

            for depth_id in unique_depths:
                if depth_id == -1: continue

                # Get the candidate wall geometry
                plane_group = dir_group.threshold([depth_id - 0.1, depth_id + 0.1], scalars="DepthID",
                                                  preference="cell")

                # THE AREA FILTER
                # Calculate total surface area
                sized = plane_group.compute_cell_sizes(length=False, volume=False)
                total_area = np.sum(sized.cell_data["Area"])

                # Minimum Wall Area: 1.0 m^2 (1,000,000 mm^2)
                # If area is smaller, it's noise/trim/window frame,Skip it
                if total_area < 1000000.0:
                    continue

                # BUILD WALL VIEW
                # Calculate Basis
                n = ref_normal
                g_dot_n = np.dot(self.global_up, n)
                v = self.global_up - g_dot_n * n
                v = v / (np.linalg.norm(v) + 1e-6)
                u = np.cross(v, n)
                u = u / (np.linalg.norm(u) + 1e-6)

                all_pts = np.array(plane_group.points)
                origin = np.mean(all_pts, axis=0)

                # Project all points to 2D to find bounds
                rel = all_pts - origin
                u_c = np.dot(rel, u)
                v_c = np.dot(rel, v)

                u_range = (float(np.min(u_c)), float(np.max(u_c)))
                v_range = (float(np.min(v_c)), float(np.max(v_c)))

                # Extract clean meshes for rasterization
                parts = plane_group.connectivity(extraction_mode="all")
                mesh_list = []
                if "RegionId" in parts.cell_data:
                    for rid in np.unique(parts.cell_data["RegionId"]):
                        sub = parts.threshold([rid - 0.1, rid + 0.1], scalars="RegionId", preference="cell")
                        sub = sub.extract_surface()
                        if sub.n_points > 0: mesh_list.append(sub)
                else:
                    mesh_list.append(parts.extract_surface())

                wv = WallView(wall_id_counter, origin, u, v, n, u_range, v_range, mesh_list)
                wall_views.append(wv)
                wall_id_counter += 1

        return wall_views

    def generate_input_channels(self, wall: WallView) -> np.ndarray:
        MAX_WIDTH_MM = 10000.0
        MAX_HEIGHT_MM = 4000.0

        width_mm = wall.u_range[1] - wall.u_range[0]
        height_mm = wall.v_range[1] - wall.v_range[0]

        u_max = min(width_mm / MAX_WIDTH_MM, 1.0)
        v_max = min(height_mm / MAX_HEIGHT_MM, 1.0)

        x = np.linspace(0, u_max, self.img_size)
        y = np.linspace(0, v_max, self.img_size)
        xv, yv = np.meshgrid(x, y)
        yv = np.flipud(yv)

        # Rasterize Mask
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        for mesh_part in wall.meshes:
            if not mesh_part.is_all_triangles:
                tri_mesh = mesh_part.triangulate()
            else:
                tri_mesh = mesh_part

            uv_points = wall.world_to_uv(tri_mesh.points)
            pixel_points = wall.uv_to_pixel(uv_points, self.img_size)
            if pixel_points.shape[0] == 0: continue

            faces = tri_mesh.faces.reshape(-1, 4)[:, 1:]
            polygons = pixel_points[faces]
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2)) for pts in polygons], color=1.0)

        # Close gaps
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return np.stack([xv, yv, mask], axis=0).astype(np.float32)

    def process_unit(self, unit_path: str, component_list: List[Dict]) -> List[WallView]:
        mesh_path = os.path.join(unit_path, "mesh.obj")
        if not os.path.exists(mesh_path): return []

        full_mesh = pv.read(mesh_path)
        processed_walls = self._cluster_walls_robust(full_mesh)

        for w in processed_walls:
            w.image_tensor = self.generate_input_channels(w)
            w.label_mask = np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        # SMART SCALE DETECTION
        # We test 3 common scenarios:
        # 1. Meters (needs x1000)
        # 2. Millimeters (needs x1)
        # 3. Centimeters (needs x10)
        scale_factor = 1.0

        if len(component_list) > 0 and len(processed_walls) > 0:
            # Pick the first component to test
            c = component_list[0]
            raw_center = (np.array(c['min']) + np.array(c['max'])) / 2.0

            best_dist = float('inf')
            best_scale = 1.0

            # Test candidates
            for candidate_scale in [1.0, 1000.0, 10.0, 0.001]:
                test_point = raw_center * candidate_scale
                # Find distance to NEAREST wall with this scale
                min_wall_dist = float('inf')
                for w in processed_walls:
                    d = abs(np.dot(test_point - w.origin, w.normal))
                    min_wall_dist = min(min_wall_dist, d)

                if min_wall_dist < best_dist:
                    best_dist = min_wall_dist
                    best_scale = candidate_scale

            # If the best scale brings us reasonably close (< 500mm), use it
            if best_dist < 500.0:
                scale_factor = best_scale
                # print(f"  [Auto-Scale] Detected scale factor: {scale_factor} (Dist: {best_dist:.1f}mm)")

        # ASSIGN COMPONENTS
        for comp in component_list:
            min_pt = np.array(comp['min'])
            max_pt = np.array(comp['max'])
            center = ((min_pt + max_pt) / 2.0) * scale_factor  # Apply detected scale

            best_wall = None
            min_dist = float('inf')

            for w in processed_walls:
                dist = abs(np.dot(center - w.origin, w.normal))

                # Check UV bounds with margin
                uv = w.world_to_uv(center.reshape(1, 3))
                u, v = uv[0]

                if (w.u_range[0] - 500 <= u <= w.u_range[1] + 500) and \
                        (w.v_range[0] - 500 <= v <= w.v_range[1] + 500):
                    if dist < min_dist:
                        min_dist = dist
                        best_wall = w

            # Threshold: 300mm
            if best_wall and min_dist < 300.0:
                uv = best_wall.world_to_uv(center.reshape(1, 3))
                pixels = best_wall.uv_to_pixel(uv, self.img_size)
                px, py = pixels[0]

                if 0 <= px < self.img_size and 0 <= py < self.img_size:
                    r = 2
                    y_min, y_max = max(0, py - r), min(self.img_size, py + r + 1)
                    x_min, x_max = max(0, px - r), min(self.img_size, px + r + 1)
                    best_wall.label_mask[0, y_min:y_max, x_min:x_max] = 1.0

        return processed_walls