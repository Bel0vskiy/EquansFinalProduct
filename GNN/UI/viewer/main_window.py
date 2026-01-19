# main_window.py

import sys
import os
import json
from pathlib import Path
from typing import Dict, Optional

import pyvista as pv  # Keep pyvista import for type hints if needed
# from pyvistaqt import QtInteractor # Imported within SceneManager
import numpy as np  # Used in panels/scene manager, maybe not directly here
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QMenuBar, QMenu, QGroupBox,
    QTextEdit, QComboBox, QMessageBox, QFileDialog, QStatusBar, QFrame,
    QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence

# Import the refactored classes
from .unit_data import UnitData
from .scene_manager import SceneManager
from .ui_panels import PropertiesPanel, VisibilityPanel, MarkerPanel, BoundingBoxPanel


class AdvancedRoomViewer(QMainWindow):
    """Main application window integrating all UI components."""

    def __init__(self, default_data_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Advanced 3D Room Viewer")
        self.setGeometry(100, 100, 1400, 900)  # x, y, width, height

        # Core components
        self.scene_manager = SceneManager()
        self.units_data: Dict[int, str] = {}  # Maps unit ID to its directory path
        self.current_building_path: Optional[str] = None

        # Build UI
        self._init_ui()
        self._init_menu()
        self._init_status_bar()
        self._init_context_menu()

        # Load initial data path


        print("Main window initialized.")

    def _init_ui(self):
        """Initialize the main UI layout and panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)  # Layout for central widget

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # --- Left Panel ---
        # This will be the content widget for the scroll area
        left_panel_content = QWidget()
        left_layout = QVBoxLayout(left_panel_content)  # Layout for the content

        # Create a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(left_panel_content)
        scroll_area.setMaximumWidth(350) # Set max width for the scroll area itself

        # Building Selection Group
        building_group = QGroupBox("Building Data Source")
        building_layout = QVBoxLayout()
        self.path_label = QLabel("Path: Not Selected")
        self.path_label.setWordWrap(True)
        self.browse_btn = QPushButton("Select Building Folder...")
        self.browse_btn.clicked.connect(self.select_building_folder)
        building_layout.addWidget(QLabel("Current Building Folder:"))
        building_layout.addWidget(self.path_label)
        building_layout.addWidget(self.browse_btn)
        building_group.setLayout(building_layout)
        left_layout.addWidget(building_group)

        # Unit Selection Group
        unit_group = QGroupBox("Unit Selection")
        unit_layout = QVBoxLayout()
        self.unit_combo = QComboBox()
        self.load_unit_btn = QPushButton("Load Selected Unit")
        self.load_all_btn = QPushButton("Load All Units")
        self.remove_units_btn = QPushButton("Clear Loaded Units")
        self.load_unit_btn.clicked.connect(self.load_selected_unit)
        self.load_all_btn.clicked.connect(self.load_all_units)
        self.remove_units_btn.clicked.connect(lambda: self.clear_loaded_units(confirm=True))
        unit_layout.addWidget(QLabel("Select Unit:"))
        unit_layout.addWidget(self.unit_combo)
        unit_layout.addWidget(self.load_unit_btn)
        unit_layout.addWidget(self.load_all_btn)
        unit_layout.addWidget(self.remove_units_btn)
        unit_group.setLayout(unit_layout)
        left_layout.addWidget(unit_group)

        # Visibility Panel (Pass SceneManager)
        self.visibility_panel = VisibilityPanel(self.scene_manager)
        left_layout.addWidget(self.visibility_panel)

        # Properties Panel (Pass SceneManager)
        self.properties_panel = PropertiesPanel(self.scene_manager)
        left_layout.addWidget(self.properties_panel)

        # Marker Panel (Pass SceneManager)
        self.marker_panel = MarkerPanel(self.scene_manager)
        left_layout.addWidget(self.marker_panel)

        # Bounding Box Panel (Pass SceneManager)
        self.bounding_box_panel = BoundingBoxPanel(self.scene_manager)
        left_layout.addWidget(self.bounding_box_panel)

        left_layout.addStretch()  # Push controls towards the top
        
        # Add the scroll area (which contains the left_panel_content) to the splitter
        splitter.addWidget(scroll_area)

        # --- Right Panel (3D Viewer) ---
        # Use a QFrame for better embedding if needed, or directly QWidget
        self.viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(self.viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)  # Remove padding around plotter
        # Initialize plotter via SceneManager, passing the frame/widget
        self.plotter = self.scene_manager.initialize_plotter(self.viewer_frame)
        viewer_layout.addWidget(self.plotter.interactor)  # Add the plotter's Qt widget
        splitter.addWidget(self.viewer_frame)  # Add viewer frame to splitter

        # Set initial splitter sizes (adjust as needed)
        splitter.setSizes([350, 1050])

    def _init_menu(self):
        """Initialize the main menu bar."""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')
        browse_action = QAction("&Select Building Folder...", self)
        browse_action.triggered.connect(self.select_building_folder)
        file_menu.addAction(browse_action)
        file_menu.addSeparator()
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)  # Use standard key
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = menubar.addMenu('&View')
        reset_camera_action = QAction('&Reset Camera', self)
        reset_camera_action.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_camera_action)
        # fit_action = QAction('&Fit to Screen', self) # Often same as reset
        # fit_action.triggered.connect(self.fit_to_screen)
        # view_menu.addAction(fit_action)

        # Tools Menu
        tools_menu = menubar.addMenu('&Tools')
        clear_action = QAction('&Clear Loaded Units', self)
        clear_action.triggered.connect(lambda: self.clear_loaded_units(confirm=True))
        tools_menu.addAction(clear_action)

        # Help Menu
        help_menu = menubar.addMenu('&Help')
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _init_status_bar(self):
        """Initialize the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _init_context_menu(self):
        """Initialize the right-click context menu."""
        # For simplicity, attached to the main window for now
        self.context_menu = QMenu(self)
        self.context_menu.addAction("Reset Camera", self.reset_camera)
        # self.context_menu.addAction("Fit to Screen", self.fit_to_screen)
        self.context_menu.addSeparator()
        self.context_menu.addAction("Clear Loaded Units", lambda: self.clear_loaded_units(confirm=True))

    # --- Action Methods / Slots ---

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

        # Clear existing scene before loading new unit list
        self.clear_loaded_units(confirm=False)  # Clear without prompt

        # Load available units from the new path
        self.load_units_from_folder(self.current_building_path)

    def load_units_from_folder(self, building_path: str):
        """Scans the folder for valid unit subdirectories and populates the dropdown."""
        self.unit_combo.clear()
        self.units_data.clear()  # Clear the map of available units
        print(f"Scanning for units in: {building_path}")

        try:
            # Use Pathlib for better path handling
            path_obj = Path(building_path)
            if not path_obj.is_dir():
                raise FileNotFoundError("Path is not a directory")

            unit_dirs = sorted([p for p in path_obj.glob("unit_*") if p.is_dir()])
        except Exception as e:
            msg = f"Error scanning directory:\n{building_path}\n\n{e}"
            QMessageBox.critical(self, "Directory Scan Error", msg)
            self.status_bar.showMessage("Error scanning directory.")
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
                    self.units_data[unit_id] = str(unit_dir.resolve())  # Store full path
                    self.unit_combo.addItem(unit_name)  # Add name to dropdown
                    found_count += 1
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse ID from directory name '{unit_name}'")
            else:
                print(f"Skipping {unit_name}: Missing mesh.obj or data.json")

        self.status_bar.showMessage(f"Found {found_count} valid units in {path_obj.name}")
        if found_count == 0:
            QMessageBox.information(self, "No Units", f"No valid 'unit_*' folders found in:\n{building_path}")

    def load_selected_unit(self):
        """Loads the unit currently selected in the combo box."""
        unit_name = self.unit_combo.currentText()
        if not unit_name:
            self.status_bar.showMessage("No unit selected.")
            return

        try:
            unit_id = int(unit_name.split('_')[1])
            if unit_id in self.scene_manager.units:
                self.status_bar.showMessage(f"Unit {unit_id} is already loaded.")
                # Optionally focus on it or bring to front if needed
            elif unit_id in self.units_data:
                self.load_unit(unit_id)
            else:
                # Should not happen if combo box is populated correctly
                QMessageBox.warning(self, "Error", f"Unit '{unit_name}' data path not found.")
        except (ValueError, IndexError):
            QMessageBox.warning(self, "Error", f"Cannot parse ID from '{unit_name}'")

    def load_all_units(self):
        """Loads all units listed in the combo box."""
        if self.unit_combo.count() == 0:
            QMessageBox.information(self, "No Units", "No units available to load. Please select a data folder.")
            return

        self.status_bar.showMessage("Loading all units...")
        QApplication.processEvents()  # Update UI

        loaded_count = 0
        for i in range(self.unit_combo.count()):
            unit_name = self.unit_combo.itemText(i)
            try:
                unit_id = int(unit_name.split('_')[1])
                # Check if already loaded before attempting to load again
                if unit_id not in self.scene_manager.units:
                    if self.load_unit(unit_id):  # Check return value of load_unit
                        loaded_count += 1
            except (ValueError, IndexError):
                print(f"Skipping invalid unit name in combo box: {unit_name}")

        self.status_bar.showMessage(f"Finished loading. Total units in scene: {len(self.scene_manager.units)}")
        if self.scene_manager.units:
            self.scene_manager.plotter.reset_camera()  # Fit camera after loading all

    def load_unit(self, unit_id: int) -> bool:
        """Loads a single unit by ID. Returns True on success, False on failure."""
        if unit_id not in self.units_data:
            print(f"Error: Path for Unit {unit_id} not found.")
            return False
        if unit_id in self.scene_manager.units:
            print(f"Info: Unit {unit_id} already loaded.")
            return True  # Consider already loaded as success

        unit_path = self.units_data[unit_id]
        unit = UnitData(unit_id, unit_path)

        print(f"Attempting to load Unit {unit_id} from {unit_path}...")
        self.status_bar.showMessage(f"Loading Unit {unit_id}...")
        QApplication.processEvents()

        if not unit.load_data():
            msg = f"Failed to load data.json for unit {unit_id}"
            QMessageBox.warning(self, "Load Error", msg)
            self.status_bar.showMessage(msg)
            return False
        if not unit.load_mesh():
            msg = f"Failed to load mesh.obj for unit {unit_id}"
            QMessageBox.warning(self, "Load Error", msg)
            self.status_bar.showMessage(msg)
            return False

        # Add to scene and UI
        self.scene_manager.add_unit(unit)  # This triggers scene update and camera reset
        self.visibility_panel.add_unit_checkbox(unit_id, f"Unit {unit_id}")
        self.properties_panel.update_unit_info(unit)  # Show info for the new unit

        self.status_bar.showMessage(f"Loaded Unit {unit_id}")
        print(f"Successfully loaded Unit {unit_id}")
        return True

    def clear_loaded_units(self, confirm=True):
        """Clears all currently loaded units from the scene."""
        if not self.scene_manager.units:  # Check if there's anything to clear
            self.status_bar.showMessage("Scene is already empty.")
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
            do_clear = True  # Clear without asking (e.g., when changing folder)

        if do_clear:
            print("Clearing loaded units...")
            self.scene_manager.remove_all_units()  # Clears SceneManager state and plotter actors
            self.visibility_panel.clear_unit_checkboxes()  # Clear UI checkboxes
            self.properties_panel.update_unit_info(None)  # Clear properties display
            self.status_bar.showMessage("Cleared loaded units.")
            print("Cleared loaded units.")

    def reset_camera(self):
        """Resets the PyVista camera."""
        if self.scene_manager.plotter:
            print("Resetting camera...")
            # self.scene_manager.plotter.camera_position = 'iso' # Can be too far out
            self.scene_manager.plotter.reset_camera()
            self.scene_manager.plotter.render()

    def fit_to_screen(self):
        """Fits the current scene contents to the screen (same as reset)."""
        self.reset_camera()

    def show_about(self):
        """Shows the About dialog."""
        QMessageBox.about(self, "About",
                          "Advanced 3D Room Viewer\nVersion 1.0\n\n"
                          "Built using PySide6 and PyVista.")

    def contextMenuEvent(self, event):
        """Shows the context menu on right-click."""
        # Optionally check if click is within plotter area
        # widget = self.childAt(event.pos())
        # if widget is self.plotter.interactor: # Check if click is on plotter
        self.context_menu.exec(event.globalPos())

    def closeEvent(self, event):
        """Handles window close event."""
        print("Closing application...")
        # Optional: Add confirmation dialog?
        # reply = QMessageBox.question(self, 'Confirm Exit',
        #                              "Are you sure you want to exit?",
        #                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        #                              QMessageBox.StandardButton.No)
        # if reply == QMessageBox.StandardButton.Yes:
        #     if self.scene_manager.plotter:
        #         self.scene_manager.plotter.close() # Close plotter window cleanly
        #     event.accept() # Proceed with closing
        # else:
        #     event.ignore() # Cancel closing

        # Simpler close without confirmation:
        if self.scene_manager.plotter:
            self.scene_manager.plotter.close()  # Close plotter window cleanly
        event.accept()