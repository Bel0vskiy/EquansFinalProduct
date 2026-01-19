import joblib
import streamlit as st
import pandas as pd

from knn_lamp_placer import KnnLampPlacer


class AssetRepository:
    COUNT_MODEL_PATH = "app/models/count_model.pkl"
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

        # normalize column name if needed
        if "room_id" not in df.columns and "room_name" in df.columns:
            df = df.rename(columns={"room_name": "room_id"})

        df = df[["room_id", "room_length", "room_height", "room_width", "x_norm", "y_norm"]]
        placer = KnnLampPlacer(k=k, use_count_in_knn=use_count_in_knn).fit(df)
        return placer
