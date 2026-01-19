import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class KnnSocketPlacer:
    def __init__(self, k=7, use_count_in_knn=True, sort_cols=("z_norm","y_norm","x_norm")):
        self.k = int(k)
        self.use_count_in_knn = bool(use_count_in_knn)
        self.sort_cols = list(sort_cols)
        self.df_ord = None
        self.room_table = None
        self.nn = None

    def fit(self, df: pd.DataFrame):
        needed = {"room_id","room_length","room_width","room_height","x_norm","y_norm","z_norm"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"df missing columns: {missing}")

        df_ord = df.sort_values(["room_id"] + self.sort_cols).copy()
        self.df_ord = df_ord[["room_id","room_length","room_width","room_height","x_norm","y_norm","z_norm"]].copy()

        meta = df_ord.groupby("room_id")[["room_length","room_width","room_height"]].first().reset_index()
        counts = df_ord.groupby("room_id").size().rename("n_sockets").reset_index()
        self.room_table = meta.merge(counts, on="room_id")

        feat_cols = ["room_length","room_width","room_height"] + (["n_sockets"] if self.use_count_in_knn else [])
        X = self.room_table[feat_cols].to_numpy(float)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self.room_table)), metric="euclidean")
        self.nn.fit(X)
        return self

    def set_k(self, k: int):
        self.k = int(k)
        if self.room_table is None:
            return
        feat_cols = ["room_length","room_width","room_height"] + (["n_sockets"] if self.use_count_in_knn else [])
        X = self.room_table[feat_cols].to_numpy(float)
        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(self.room_table)), metric="euclidean")
        self.nn.fit(X)

    def _layout_norm(self, room_id: str) -> np.ndarray:
        return self.df_ord[self.df_ord["room_id"] == room_id][["x_norm","y_norm","z_norm"]].to_numpy(float)

    def predict_room(self, room_length, room_width, room_height, count_model, require_at_least=True):
        n = int(np.rint(count_model.predict([[room_length, room_width, room_height]])[0]))
        n = max(n, 0)
        if n == 0:
            return np.empty((0, 3)), 0

        if self.use_count_in_knn:
            query = np.array([[room_length, room_width, room_height, n]], float)
            feat_cols = ["room_length","room_width","room_height","n_sockets"]
        else:
            query = np.array([[room_length, room_width, room_height]], float)
            feat_cols = ["room_length","room_width","room_height"]

        _, idxs = self.nn.kneighbors(query, n_neighbors=min(self.k, len(self.room_table)))
        neighbor_rooms = self.room_table.iloc[idxs[0]]["room_id"].tolist()

        # clamp n so we don't average all-NaN columns
        max_supported = max((len(self._layout_norm(r)) for r in neighbor_rooms), default=0)
        n = min(n, max_supported)
        if n == 0:
            return np.empty((0, 3)), 0

        layouts = []
        for rid in neighbor_rooms:
            lay = self._layout_norm(rid)
            if require_at_least:
                if len(lay) >= n:
                    layouts.append(lay[:n])
            else:
                if len(lay) == n:
                    layouts.append(lay)

        layouts = np.stack(layouts, axis=0)
        pred_norm = np.nanmean(layouts, axis=0)

        x_m = pred_norm[:, 0] * room_width
        y_m = pred_norm[:, 1] * room_length
        z_m = pred_norm[:, 2] * room_height
        return np.column_stack([x_m, y_m, z_m]), n
