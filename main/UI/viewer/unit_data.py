import os
import json
import pyvista as pv

class UnitData:

    def __init__(self, unit_id: int, data_path: str):
        self.id = unit_id
        self.data_path = data_path
        self.mesh_path = os.path.join(data_path, "mesh.obj")
        self.json_path = os.path.join(data_path, "data.json")
        self.mesh = None
        self.data = None
        self.loaded = False

    def load_data(self) -> bool:
        try:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"[ERROR] Loading JSON for unit {self.id}: {e}")
            return False

    def load_mesh(self) -> bool:
        try:
            if os.path.exists(self.mesh_path):
                self.mesh = pv.read(self.mesh_path)
                self.loaded = True
                return True
            print(f"[WARN] Mesh file not found for unit {self.id}: {self.mesh_path}")
            return False
        except Exception as e:
            print(f"[ERROR] Loading mesh for unit {self.id}: {e}")
            return False
