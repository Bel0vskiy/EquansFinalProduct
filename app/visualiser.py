import numpy as np
import plotly.graph_objects as go


class RoomVisualizer:
    @staticmethod
    def room_wireframe_traces(L, W, H):
        corners = np.array(
            [
                [0, 0, 0],
                [W, 0, 0],
                [W, L, 0],
                [0, L, 0],
                [0, 0, H],
                [W, 0, H],
                [W, L, H],
                [0, L, H],
            ],
            dtype=float,
        )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        traces = []
        for a, b in edges:
            traces.append(
                go.Scatter3d(
                    x=[corners[a, 0], corners[b, 0]],
                    y=[corners[a, 1], corners[b, 1]],
                    z=[corners[a, 2], corners[b, 2]],
                    mode="lines",
                    showlegend=False,
                )
            )
        return traces

    @staticmethod
    def build_figure(room_length, room_width, room_height, lamps_m: np.ndarray):
        fig = go.Figure()

        for tr in RoomVisualizer.room_wireframe_traces(room_length, room_width, room_height):
            fig.add_trace(tr)

        if lamps_m is not None and len(lamps_m):
            fig.add_trace(
                go.Scatter3d(
                    x=lamps_m[:, 0],
                    y=lamps_m[:, 1],
                    z=lamps_m[:, 2],
                    mode="markers+text",
                    text=[str(i + 1) for i in range(len(lamps_m))],
                    textposition="top center",
                    name="lamps",
                    marker=dict(
                        symbol="square",
                        size=6,
                        color="yellow",
                        line=dict(width=1, color="black"),
                    ),
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="Width (m)",
                yaxis_title="Length (m)",
                zaxis_title="Height (m)",
                xaxis=dict(range=[0, room_width]),
                yaxis=dict(range=[0, room_length]),
                zaxis=dict(range=[0, room_height]),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=650,
        )
        return fig
