# main.py

import sys
import traceback
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, 
    QVBoxLayout, QMessageBox, QStatusBar, QToolBar, QLabel
)
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtCore import Qt

"""
Main file to run latest version of UI with Landing Page and Multi-Model Support
"""

# Add GNN root to path to allow importing Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Widgets
from viewer.intro_window import IntroWidget
from viewer.main_window import GNNViewerWidget
from viewer.knn_window import KNNViewerWidget

# --- MODERN DARK THEME QSS ---
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
        self._init_menu()
        self._init_toolbar()

    def _init_ui(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Stacked Widget for Pages
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 1. Intro Page
        self.intro_page = IntroWidget()
        self.intro_page.request_gnn_mode.connect(self.show_gnn_page)
        self.intro_page.request_knn_mode.connect(self.show_knn_page)
        self.stack.addWidget(self.intro_page)

        # 2. GNN Viewer Page
        self.gnn_page = GNNViewerWidget()
        self.gnn_page.status_message.connect(self.update_status)
        self.stack.addWidget(self.gnn_page)
        
        # 3. KNN Generator Page
        self.knn_page = KNNViewerWidget()
        self.knn_page.status_message.connect(self.update_status)
        self.stack.addWidget(self.knn_page)

    def _init_menu(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')
        
        home_action = QAction('&Home', self)
        home_action.setShortcut(QKeySequence("Ctrl+H"))
        home_action.triggered.connect(self.show_home_page)
        file_menu.addAction(home_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu (Common)
        view_menu = menubar.addMenu('&View')
        reset_cam_action = QAction('&Reset Camera', self)
        reset_cam_action.triggered.connect(self.propagate_reset_camera)
        view_menu.addAction(reset_cam_action)

        # Help
        help_menu = menubar.addMenu('&Help')
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _init_toolbar(self):
        toolbar = QToolBar("Navigation")
        self.addToolBar(toolbar)
        
        home_btn = QAction("Home", self)
        home_btn.triggered.connect(self.show_home_page)
        toolbar.addAction(home_btn)
        
        toolbar.addSeparator()
        
        self.lbl_mode = QLabel("  Mode: Selection  ")
        toolbar.addWidget(self.lbl_mode)

    def show_home_page(self):
        self.stack.setCurrentWidget(self.intro_page)
        self.lbl_mode.setText("  Mode: Selection  ")
        self.status_bar.showMessage("Select a model to start.")

    def show_gnn_page(self):
        self.stack.setCurrentWidget(self.gnn_page)
        self.lbl_mode.setText("  Mode: GNN Generator  ")
        self.status_bar.showMessage("GNN Generator Ready.")

    def show_knn_page(self):
        self.stack.setCurrentWidget(self.knn_page)
        self.lbl_mode.setText("  Mode: KNN Generator  ")
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
        # Cleanup child widgets
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