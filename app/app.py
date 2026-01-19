import streamlit as st

from initialiser import AssetRepository
from visualiser import RoomVisualizer


def main():
    st.title("EQUANS MEP Generator")

    count_model = AssetRepository.load_count_model()
    placer = AssetRepository.load_placer(k=7, use_count_in_knn=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        room_length = st.number_input("Room length (m)", min_value=0.1, value=6.0, step=0.1)
    with c2:
        room_width = st.number_input("Room width (m)", min_value=0.1, value=4.0, step=0.1)
    with c3:
        room_height = st.number_input("Room height (m)", min_value=0.1, value=2.7, step=0.1)

    k = st.slider("Set Nearest Rooms", min_value=1, max_value=30, value=7, step=1)
    placer.set_k(k)

    if st.button("Generate components"):
        lamps_m, n = placer.predict_room(
            room_length=room_length,
            room_width=room_width,
            room_height=room_height,
            count_model=count_model,
        )

        st.write(f"Predicted lamps: **{n}**")

        fig = RoomVisualizer.build_figure(room_length, room_width, room_height, lamps_m)
        st.plotly_chart(fig, use_container_width=True)

        if len(lamps_m):
            st.subheader("Component Positions Meters (m)")
            st.write(
                [
                    {"lamp": i + 1, "x_m": float(p[0]), "y_m": float(p[1]), "z_m": float(p[2])}
                    for i, p in enumerate(lamps_m)
                ]
            )


if __name__ == "__main__":
    main()
