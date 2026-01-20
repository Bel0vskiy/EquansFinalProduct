from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

class IntroWidget(QWidget):
    """
    Landing page widget to select between GNN and KNN models.
    """
    request_gnn_mode = Signal()
    request_knn_mode = Signal()

    def __init__(self):
        super().__init__()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(40)

        # Title
        title_label = QLabel("EQUANS MEP Generator")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Select Model to Use")
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label)

        # Buttons Container
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(40)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # KNN Button
        knn_btn = QPushButton("KNN Model Generator")
        knn_btn.setMinimumSize(250, 150)
        knn_btn.clicked.connect(self.request_knn_mode)
        # Custom style for big button
        knn_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                font-weight: bold; 
                padding: 20px;
                background-color: #2e7d32; /* Greenish hint */
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
        """)
        buttons_layout.addWidget(knn_btn)

        # GNN Button
        gnn_btn = QPushButton("GNN Model Generator")
        gnn_btn.setMinimumSize(250, 150)
        gnn_btn.clicked.connect(self.request_gnn_mode)
        gnn_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                font-weight: bold; 
                padding: 20px;
                background-color: #1565c0; /* Blue hint */
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        buttons_layout.addWidget(gnn_btn)

        layout.addLayout(buttons_layout)
        
        # Spacer at bottom to visually center things a bit higher
        layout.addItem(QSpacerItem(20, 100, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
