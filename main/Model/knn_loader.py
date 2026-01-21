import os
import sys
import joblib
import pandas as pd
from typing import Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from app.knn_lamp_placer import KnnLampPlacer
    from app.knn_socket_placer import KnnSocketPlacer
except ImportError:
    print("Warning: Could not import placers from 'app'. Ensure project root is in PYTHONPATH.")
    pass

class KnnLoader:
    _instance = None
    
    def __init__(self):
        self.count_model = None
        self.socket_count_model = None
        self.lamp_placer = None
        self.socket_placer = None
        
        # Paths relative to project root
        self.base_path = project_root
        self.COUNT_MODEL_PATH = os.path.join(self.base_path, "app", "models", "count_model.pkl")
        self.SOCKET_COUNT_MODEL_PATH = os.path.join(self.base_path, "app", "models", "socket_count_model.pkl")
        self.CSV_1_PATH = os.path.join(self.base_path, "app", "models", "rooms_lamps.csv")
        self.CSV_2_PATH = os.path.join(self.base_path, "app", "models", "rooms_lamps_V22.csv")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = KnnLoader()
        return cls._instance

    def load_count_model(self):
        if self.count_model is None:
            if not os.path.exists(self.COUNT_MODEL_PATH):
                raise FileNotFoundError(f"Model not found: {self.COUNT_MODEL_PATH}")
            self.count_model = joblib.load(self.COUNT_MODEL_PATH)
        return self.count_model

    def load_socket_count_model(self):
        if self.socket_count_model is None:
            if not os.path.exists(self.SOCKET_COUNT_MODEL_PATH):
                raise FileNotFoundError(f"Model not found: {self.SOCKET_COUNT_MODEL_PATH}")
            self.socket_count_model = joblib.load(self.SOCKET_COUNT_MODEL_PATH)
        return self.socket_count_model

    def load_placer(self, k: int = 7, use_count_in_knn: bool = True) -> 'KnnLampPlacer':
        
        df = self._load_combined_df()
        
        df_lamps = df[["room_id", "room_length", "room_height", "room_width", "x_norm", "y_norm"]].copy()
        placer = KnnLampPlacer(k=k, use_count_in_knn=use_count_in_knn).fit(df_lamps)
        return placer

    def load_socket_placer(self, k: int = 7, use_count_in_knn: bool = True) -> 'KnnSocketPlacer':
        df = self._load_combined_df()
        
        if "component_type" in df.columns:
            df_sockets = df[df["component_type"].str.lower().eq("socket")].copy()
        else:
            df_sockets = df.copy()
            
        df_sockets = df_sockets[["room_id", "room_length", "room_height", "room_width", "x_norm", "y_norm", "z_norm"]].dropna()
        
        placer = KnnSocketPlacer(k=k, use_count_in_knn=use_count_in_knn).fit(df_sockets)
        return placer

    def _load_combined_df(self) -> pd.DataFrame:
        if hasattr(self, '_cached_df'):
            return self._cached_df
            
        if not os.path.exists(self.CSV_1_PATH) or not os.path.exists(self.CSV_2_PATH):
             raise FileNotFoundError(f"CSV files not found in {os.path.dirname(self.CSV_1_PATH)}")

        df1 = pd.read_csv(self.CSV_1_PATH, sep=";")
        df2 = pd.read_csv(self.CSV_2_PATH, sep=";")
        df = pd.concat([df1, df2], ignore_index=True)

        if "room_id" not in df.columns and "room_name" in df.columns:
            df = df.rename(columns={"room_name": "room_id"})
            
        self._cached_df = df
        return df
