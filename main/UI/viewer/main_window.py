
# main_window.py
import sys
import os
import json
from pathlib import Path
from typing import Dict, Optional

import pyvista as pv
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QGroupBox,
    QTextEdit, QComboBox, QMessageBox, QFileDialog, QFrame,
    QScrollArea, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

# Import the refactored classes
from .unit_data import UnitData
from .scene_manager import SceneManager
from .ui_panels import PropertiesPanel, VisibilityPanel, MarkerPanel, PredictionPanel


class GNNViewerWidget(QWidget):
    """
    Main GNN Viewer component, refactored as a QWidget to be embedded in a main window.
    """

    status_message = Signal(str)

    def __init__(self, default_data_path: Optional[str] = None):
        super().__init__()
        self.scene_manager = SceneManager()
        self.units_data: Dict[int, str] = {}
        self.current_building_path: Optional[str] = None

        self._init_ui()
        self._init_context_menu()

        print("MEP Generator initialized.")

    def _init_ui(self):
        """Initialize the main UI layout and panels."""
        main_layout = QHBoxLayout(self) 

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_panel_content = QWidget()
        left_layout = QVBoxLayout(left_panel_content)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel_content)
        scroll_area.setMinimumWidth(300)

        data_group = QGroupBox("Data Source")
        data_layout = QVBoxLayout()
        data_layout.setContentsMargins(5, 5, 5, 5)
        

        building_row = QVBoxLayout()
        building_row.setSpacing(2)
        building_row.addWidget(QLabel("Building Folder:"))
        self.path_label = QLabel("Not Selected")
        self.path_label.setStyleSheet("color: gray; font-style: italic;")
        self.path_label.setWordWrap(True)
        building_row.addWidget(self.path_label)
        
        self.browse_btn = QPushButton("Select Folder...")
        self.browse_btn.clicked.connect(self.select_building_folder)
        building_row.addWidget(self.browse_btn)
        
        data_layout.addLayout(building_row)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        data_layout.addWidget(line)

        unit_row = QVBoxLayout()
        unit_row.setSpacing(2)
        unit_row.addWidget(QLabel("Unit Selection:"))
        self.unit_combo = QComboBox()
        unit_row.addWidget(self.unit_combo)
        
        btns_layout = QHBoxLayout()
        self.load_unit_btn = QPushButton("Load")
        self.load_unit_btn.clicked.connect(self.load_selected_unit)
        self.load_unit_btn.setProperty("class", "primary")

        self.load_all_btn = QPushButton("Load All")
        self.load_all_btn.clicked.connect(self.load_all_units)
        
        self.remove_units_btn = QPushButton("Clear")
        self.remove_units_btn.clicked.connect(lambda: self.clear_loaded_units(confirm=True))
        self.remove_units_btn.setProperty("class", "danger")
        
        btns_layout.addWidget(self.load_unit_btn)
        btns_layout.addWidget(self.load_all_btn)
        btns_layout.addWidget(self.remove_units_btn)
        unit_row.addLayout(btns_layout)
        
        data_layout.addLayout(unit_row)
        
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)


        self.visibility_panel = VisibilityPanel(self.scene_manager)
        left_layout.addWidget(self.visibility_panel)


        self.properties_panel = PropertiesPanel(self.scene_manager)
        left_layout.addWidget(self.properties_panel)

        self.prediction_panel = PredictionPanel(self.scene_manager)
        left_layout.addWidget(self.prediction_panel)


        self.marker_panel = MarkerPanel(self.scene_manager)
        left_layout.addWidget(self.marker_panel)

        left_layout.addStretch()
        
        splitter.addWidget(scroll_area)

        self.viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(self.viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = self.scene_manager.initialize_plotter(self.viewer_frame)
        viewer_layout.addWidget(self.plotter.interactor)
        splitter.addWidget(self.viewer_frame)

        splitter.setSizes([350, 1050])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _init_context_menu(self):
        """Initialize the right-click context menu."""

        self.context_menu = QMenu(self)
        self.context_menu.addAction("Reset Camera", self.reset_camera)
        self.context_menu.addSeparator()
        self.context_menu.addAction("Clear Loaded Units", lambda: self.clear_loaded_units(confirm=True))

    def show_message(self, msg: str):
        self.status_message.emit(msg)



    def select_building_folder(self):
        """Opens a dialog to select the building folder."""
        start_dir = self.current_building_path or str(Path.home())
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Building Folder (containing unit_* subfolders)", start_dir
        )
        if folder_path:
            self.set_building_folder(folder_path)

    def set_building_folder(self, folder_path: str):
        """Sets the current building folder and refreshes the unit list."""
        resolved_path = str(Path(folder_path).resolve())
        if not os.path.isdir(resolved_path):
            QMessageBox.warning(self, "Invalid Path", f"Directory not found:\n{resolved_path}")
            return

        self.current_building_path = resolved_path
        self.path_label.setText(f"Path: {self.current_building_path}")
        print(f"Set building folder: {self.current_building_path}")

        self.clear_loaded_units(confirm=False)

        self.load_units_from_folder(self.current_building_path)

    def load_units_from_folder(self, building_path: str):
        """Scans the folder for valid unit subdirectories and populates the dropdown."""
        self.unit_combo.clear()
        self.units_data.clear()
        print(f"Scanning for units in: {building_path}")

        try:
            path_obj = Path(building_path)
            if not path_obj.is_dir():
                raise FileNotFoundError("Path is not a directory")

            unit_dirs = sorted([p for p in path_obj.glob("unit_*") if p.is_dir()])
        except Exception as e:
            msg = f"Error scanning directory:\n{building_path}\n\n{e}"
            QMessageBox.critical(self, "Directory Scan Error", msg)
            self.show_message("Error scanning directory.")
            print(f"Error scanning: {e}")
            return

        print(f"Found potential unit directories: {[p.name for p in unit_dirs]}")
        found_count = 0
        for unit_dir in unit_dirs:
            unit_name = unit_dir.name
            mesh_file = unit_dir / "mesh.obj"
            json_file = unit_dir / "data.json"

            if mesh_file.exists() and json_file.exists():
                try:
                    unit_id = int(unit_name.split('_')[1])
                    self.units_data[unit_id] = str(unit_dir.resolve())
                    self.unit_combo.addItem(unit_name)
                    found_count += 1
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse ID from directory name '{unit_name}'")
            else:
                print(f"Skipping {unit_name}: Missing mesh.obj or data.json")

        self.show_message(f"Found {found_count} valid units in {path_obj.name}")
        if found_count == 0:
            QMessageBox.information(self, "No Units", f"No valid 'unit_*' folders found in:\n{building_path}")

    def load_selected_unit(self):
        """Loads the unit currently selected in the combo box."""
        unit_name = self.unit_combo.currentText()
        if not unit_name:
            self.show_message("No unit selected.")
            return

        try:
            unit_id = int(unit_name.split('_')[1])
            if unit_id in self.scene_manager.units:
                self.show_message(f"Unit {unit_id} is already loaded.")
            elif unit_id in self.units_data:
                self.load_unit(unit_id)
            else:
                QMessageBox.warning(self, "Error", f"Unit '{unit_name}' data path not found.")
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Error", f"Cannot parse ID from '{unit_name}'")

    def load_all_units(self):
        """Loads all units listed in the combo box."""
        if self.unit_combo.count() == 0:
            QMessageBox.information(self, "No Units", "No units available to load. Please select a data folder.")
            return

        self.show_message("Loading all units...")
        QApplication.processEvents()

        loaded_count = 0
        for i in range(self.unit_combo.count()):
            unit_name = self.unit_combo.itemText(i)
            try:
                unit_id = int(unit_name.split('_')[1])
                if unit_id not in self.scene_manager.units:
                    if self.load_unit(unit_id):
                        loaded_count += 1
            except (ValueError, IndexError):
                print(f"Skipping invalid unit name in combo box: {unit_name}")

        self.show_message(f"Finished loading. Total units in scene: {len(self.scene_manager.units)}")
        if self.scene_manager.units:
            self.scene_manager.plotter.reset_camera()

    def load_unit(self, unit_id: int) -> bool:
        """Loads a single unit by ID. Returns True on success, False on failure."""
        if unit_id not in self.units_data:
            print(f"Error: Path for Unit {unit_id} not found.")
            return False
        if unit_id in self.scene_manager.units:
            print(f"Info: Unit {unit_id} already loaded.")
            return True

        unit_path = self.units_data[unit_id]
        unit = UnitData(unit_id, unit_path)

        print(f"Attempting to load Unit {unit_id} from {unit_path}...")
        self.show_message(f"Loading Unit {unit_id}...")
        QApplication.processEvents()

        if not unit.load_data():
            msg = f"Failed to load data.json for unit {unit_id}"
            QMessageBox.warning(self, "Load Error", msg)
            self.show_message(msg)
            return False
        if not unit.load_mesh():
            msg = f"Failed to load mesh.obj for unit {unit_id}"
            QMessageBox.warning(self, "Load Error", msg)
            self.show_message(msg)
            return False

        self.scene_manager.add_unit(unit)  
        self.visibility_panel.add_unit_checkbox(unit_id, f"Unit {unit_id}")
        self.properties_panel.update_unit_info(unit)  
        self.prediction_panel.update_targets(unit_path)

        self.show_message(f"Loaded Unit {unit_id}")
        print(f"Successfully loaded Unit {unit_id}")
        return True

    def clear_loaded_units(self, confirm=True):
        """Clears all currently loaded units from the scene."""
        if not self.scene_manager.units:
            self.show_message("Scene is already empty.")
            return

        do_clear = False
        if confirm:
            reply = QMessageBox.question(self, "Confirm Clear",
                                         "Remove all loaded units from the scene?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                do_clear = True
        else:
            do_clear = True

        if do_clear:
            print("Clearing loaded units...")
            self.scene_manager.remove_all_units()
            self.visibility_panel.clear_unit_checkboxes()
            self.properties_panel.update_unit_info(None)
            self.show_message("Cleared loaded units.")
            print("Cleared loaded units.")

    def reset_camera(self):
        """Resets the PyVista camera."""
        if self.scene_manager.plotter:
            print("Resetting camera...")
            self.scene_manager.plotter.reset_camera()
            self.scene_manager.plotter.render()

    def show_about(self):
        """Shows the About dialog."""
        QMessageBox.about(self, "About",
                          "Advanced 3D Room Viewer\nVersion 1.0\n\n"
                          "Built using PySide6 and PyVista.")

    def contextMenuEvent(self, event):
        """Shows the context menu on right-click."""
        self.context_menu.exec(event.globalPos())

    def cleanup(self):
        """Explicit cleanup method to handle resource disposal."""
        print("Cleaning up MEP Generator...")
        if self.scene_manager.plotter:
            self.scene_manager.plotter.close()