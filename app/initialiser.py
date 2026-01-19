import joblib
import streamlit as st
import pandas as pd

from knn_lamp_placer import KnnLampPlacer
from knn_socket_placer import KnnSocketPlacer

class AssetRepository:
    COUNT_MODEL_PATH = "app/models/count_model.pkl"
    SOCKET_COUNT_MODEL_PATH = "app/models/socket_count_model.pkl"
    CSV_1_PATH = "app/models/rooms_lamps.csv"
    CSV_2_PATH = "app/models/rooms_lamps_V22.csv"

    @staticmethod
    @st.cache_resource
    def load_count_model():
        return joblib.load(AssetRepository.COUNT_MODEL_PATH)

    @staticmethod
    @st.cache_resource
    def load_placer(k: int = 7, use_count_in_knn: bool = True) -> KnnLampPlacer:
        df1 = pd.read_csv(AssetRepository.CSV_1_PATH, sep=";")
        df2 = pd.read_csv(AssetRepository.CSV_2_PATH, sep=";")
        df = pd.concat([df1, df2], ignore_index=True)

        if "room_id" not in df.columns and "room_name" in df.columns:
            df = df.rename(columns={"room_name": "room_id"})

        df = df[["room_id", "room_length", "room_height", "room_width", "x_norm", "y_norm"]]
        placer = KnnLampPlacer(k=k, use_count_in_knn=use_count_in_knn).fit(df)
        return placer

    @staticmethod
    @st.cache_resource
    def load_socket_count_model():
        return joblib.load(AssetRepository.SOCKET_COUNT_MODEL_PATH)

    @staticmethod
    @st.cache_resource
    def load_socket_placer(k: int = 7, use_count_in_knn: bool = True) -> KnnSocketPlacer:
        df1 = pd.read_csv(AssetRepository.CSV_1_PATH, sep=";")
        df2 = pd.read_csv(AssetRepository.CSV_2_PATH, sep=";")
        df = pd.concat([df1, df2], ignore_index=True)

        if "room_id" not in df.columns and "room_name" in df.columns:
            df = df.rename(columns={"room_name": "room_id"})

        if "component_type" in df.columns:
            df = df[df["component_type"].str.lower().eq("socket")]

        df = df[["room_id", "room_length", "room_height", "room_width", "x_norm", "y_norm", "z_norm"]].dropna()
        return KnnSocketPlacer(k=k, use_count_in_knn=use_count_in_knn).fit(df)