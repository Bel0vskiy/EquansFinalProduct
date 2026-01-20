from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QFrame, QPushButton, QButtonGroup
)
from PySide6.QtCore import Qt

from .knn_window import KNNViewerWidget
from .main_window import GNNViewerWidget


class ModelSwitchWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EQUANS MEP Generator")
        self.resize(1400, 900)

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # ===== Segmented Toggle (in one frame) =====
        toggle_frame = QFrame()
        toggle_frame.setObjectName("ModeToggleFrame")
        toggle_frame.setFixedHeight(38)

        tlay = QHBoxLayout(toggle_frame)
        tlay.setContentsMargins(3, 3, 3, 3)
        tlay.setSpacing(0)  # слеплены

        self.btn_knn = QPushButton("KNN")
        self.btn_gnn = QPushButton("GNN")

        for b in (self.btn_knn, self.btn_gnn):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            # ключевой момент: классами управляет QSS
            b.setProperty("class", "mode_toggle")

        self.btn_knn.setProperty("class", "mode_toggle mode_toggle_left")
        self.btn_gnn.setProperty("class", "mode_toggle mode_toggle_right")

        group = QButtonGroup(toggle_frame)
        group.setExclusive(True)
        group.addButton(self.btn_knn)
        group.addButton(self.btn_gnn)

        tlay.addWidget(self.btn_knn)
        tlay.addWidget(self.btn_gnn)

        # Центрируем toggle по ширине сайдбара
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(0)

        SIDEBAR_WIDTH = 300  # под твой UI (у GNN scroll min width 300, у KNN 280)
        toggle_frame.setFixedWidth(SIDEBAR_WIDTH - 20)  # чуть меньше, чтобы красиво влезло

        toggle_row.addStretch(1)
        toggle_row.addWidget(toggle_frame)
        toggle_row.addStretch(1)

        # ===== Pages stack =====
        self.stack = QStackedWidget()
        self.gnn_page = GNNViewerWidget()
        self.knn_page = KNNViewerWidget()

        self.stack.addWidget(self.gnn_page)
        self.stack.addWidget(self.knn_page)

        # Default = GNN
        self.btn_gnn.setChecked(True)
        self.stack.setCurrentWidget(self.gnn_page)

        # Wiring
        self.btn_gnn.clicked.connect(lambda: self.stack.setCurrentWidget(self.gnn_page))
        self.btn_knn.clicked.connect(lambda: self.stack.setCurrentWidget(self.knn_page))

        # Layout order: toggle row, then stack
        root_layout.addLayout(toggle_row)
        root_layout.addWidget(self.stack, 1)
