import sys
import traceback
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QMessageBox, QStatusBar, QLabel,
    QPushButton, QButtonGroup
)

"""
Main file to run latest version of UI with Landing Page and Multi-Model Support
"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from viewer.main_window import GNNViewerWidget
from viewer.knn_window import KNNViewerWidget

DARK_STYLESHEET = """
/* Global */
QMainWindow, QDialog, QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #4a4a4a;
    border-radius: 6px;
    margin-top: 24px;
    padding-top: 10px;
    font-weight: bold;
    color: #ffffff;
    background-color: #333333;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    background-color: #2b2b2b; /* Match window bg for title mask effect */
}

/* Buttons */
QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    color: #ffffff;
    padding: 6px 12px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #505050;
    border-color: #666666;
}
QPushButton:pressed {
    background-color: #323232;
    border-color: #444444;
}
QPushButton:disabled {
    background-color: #2b2b2b;
    border-color: #3d3d3d;
    color: #666666;
}

/* Specific Button Classes */
QPushButton[class="primary"] {
    background-color: #2e7d32; /* Green */
    border: 1px solid #2e7d32;
}
QPushButton[class="primary"]:hover {
    background-color: #388e3c;
}

QPushButton[class="danger"] {
    background-color: #c62828; /* Red */
    border: 1px solid #c62828;
}
QPushButton[class="danger"]:hover {
    background-color: #d32f2f;
}

/* Panel Toggles */
QPushButton[class="panel_toggle"] {
    text-align: left;
    background-color: transparent;
    border: none;
    border-bottom: 1px solid #3d3d3d;
    color: #aaaaaa;
    padding: 8px 0px;
    font-weight: bold;
}
QPushButton[class="panel_toggle"]:hover {
    color: #ffffff;
    background-color: #323232;
}

/* Input Fields */
QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox, QDoubleSpinBox {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    color: #ffffff;
    padding: 4px;
}
QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #5c9eff;
}

/* ComboBox */
QComboBox {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 4px;
    color: #ffffff;
}
QComboBox::drop-down {
    border: none;
    background: transparent;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    selection-background-color: #3d4b5c;
}

/* List Widget */
QListWidget {
    background-color: #1e1e1e;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 4px;
}
QListWidget::item {
    padding: 4px;
}
QListWidget::item:selected {
    background-color: #3d4b5c;
    color: #ffffff;
}
QListWidget::item:hover {
    background-color: #2a2a2a;
}

/* Slider */
QSlider::groove:horizontal {
    border: 1px solid #3d3d3d;
    height: 8px;
    background: #1e1e1e;
    margin: 2px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #5c9eff;
    border: 1px solid #5c9eff;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background: #2b2b2b;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #555555;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QPushButton[mode="toggle"] {
    background-color: #2b2b2b;
    border: 1px solid #555555;
    color: #cfcfcf;
    padding: 6px 10px;
    min-height: 24px;
}

QPushButton[mode="toggle"]:checked {
    background-color: #708090;
    border-color: #778899;
    color: #ffffff;
    font-weight: bold;
}

QPushButton[side="left"] {
    border-top-left-radius: 6px;
    border-bottom-left-radius: 6px;
}

QPushButton[side="right"] {
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

QLabel {
    background: transparent;
}
QGroupBox::title {
    background: transparent; /* или оставь как было, если нравится "mask" */
}
"""

class CombinedMainWindow(QMainWindow):
    """
    Top-level application window managing navigation between Home, GNN, and KNN views.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EQUANS MEP Generator")
        self.setGeometry(100, 100, 1400, 900)

        self._init_ui()

    def _init_ui(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)
        self.setCentralWidget(root)

        toggle_bar = QWidget()
        toggle_layout = QHBoxLayout(toggle_bar)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(2)

        self.btn_knn = QPushButton("KNN")
        self.btn_gnn = QPushButton("GNN")

        for b in (self.btn_knn, self.btn_gnn):
            b.setCheckable(True)

        self.btn_knn.setProperty("mode", "toggle")
        self.btn_gnn.setProperty("mode", "toggle")

        self.btn_knn.setProperty("side", "left")
        self.btn_gnn.setProperty("side", "right")

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_knn)
        self.mode_group.addButton(self.btn_gnn)

        toggle_layout.addStretch(1)
        toggle_layout.addWidget(self.btn_knn)
        toggle_layout.addWidget(self.btn_gnn)
        toggle_layout.addStretch(1)

        root_layout.addWidget(toggle_bar)

        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, 1)

        self.gnn_page = GNNViewerWidget()
        self.gnn_page.status_message.connect(self.update_status)
        self.stack.addWidget(self.gnn_page)

        self.knn_page = KNNViewerWidget()
        self.knn_page.status_message.connect(self.update_status)
        self.stack.addWidget(self.knn_page)

        self.btn_gnn.setChecked(True)
        self.stack.setCurrentWidget(self.gnn_page)
        self.lbl_mode = None  # toolbar label больше не нужен
        self.status_bar.showMessage("GNN Generator Ready.")

        self.btn_gnn.clicked.connect(self.show_gnn_page)
        self.btn_knn.clicked.connect(self.show_knn_page)

    def show_gnn_page(self):
        self.stack.setCurrentWidget(self.gnn_page)
        self.btn_gnn.setChecked(True)
        self.status_bar.showMessage("GNN Generator Ready.")

    def show_knn_page(self):
        self.stack.setCurrentWidget(self.knn_page)
        self.btn_knn.setChecked(True)
        self.status_bar.showMessage("KNN Generator Ready.")

    def update_status(self, msg: str):
        self.status_bar.showMessage(msg)

    def propagate_reset_camera(self):
        current = self.stack.currentWidget()
        if hasattr(current, 'plotter') and current.plotter:
             current.plotter.reset_camera()
             current.plotter.render()
        elif hasattr(current, 'reset_camera'):
             current.reset_camera()

    def show_about(self):
        QMessageBox.about(self, "About",
                          "EQUANS MEP Generator\n\n"
                          "Features:\n"
                          "- GNN-based placement on 3D scans\n"
                          "- KNN-based placement on manual rooms")

    def closeEvent(self, event):
        if hasattr(self.gnn_page, 'cleanup'):
            self.gnn_page.cleanup()
        if hasattr(self.knn_page, 'cleanup'):
            self.knn_page.cleanup()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    
    window = CombinedMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)