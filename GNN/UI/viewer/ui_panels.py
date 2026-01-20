# ui_panels.py
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QListWidget, QTextEdit,
    QCheckBox, QPushButton, QListWidgetItem, QHBoxLayout, QLineEdit,
    QComboBox, QMessageBox, QApplication
)
from PySide6.QtGui import QDoubleValidator
from PySide6.QtCore import Qt, QLocale
from typing import Optional, Dict

# Import SceneManager for type hinting and interaction
# Assuming scene_manager.py is in the same directory or accessible
from .scene_manager import SceneManager
from .unit_data import UnitData # Import UnitData for type hinting
from Model import evaluate # Import GNN evaluation logic

class PropertiesPanel(QWidget):
    """Information properties panel to display unit and object details."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self.scene_manager = scene_manager # Store reference for highlighting
        self.current_unit: Optional[UnitData] = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI elements of the panel."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # Compact main layout

        # --- Unit Information Group (Collapsible) ---
        self.unit_info_group = QGroupBox("Unit Information")
        self.unit_info_layout = QVBoxLayout()
        
        self.unit_id_label = QLabel("Unit ID: None")
        self.unit_bounds_label = QLabel("Bounds: None")
        self.unit_objects_label = QLabel("Objects: None")
        
        self.unit_info_layout.addWidget(self.unit_id_label)
        self.unit_info_layout.addWidget(self.unit_bounds_label)
        self.unit_info_layout.addWidget(self.unit_objects_label)
        self.unit_info_group.setLayout(self.unit_info_layout)
        
        # Wrapper for collapse
        self.unit_info_toggle = QPushButton("Hide Unit Information")
        self.unit_info_toggle.setCheckable(True)
        self.unit_info_toggle.setChecked(False) # Not checked = Expanded (logic below)
        self.unit_info_toggle.clicked.connect(lambda: self.toggle_group(self.unit_info_group, self.unit_info_toggle))
        self.unit_info_toggle.setProperty("class", "panel_toggle")
        
        layout.addWidget(self.unit_info_toggle)
        layout.addWidget(self.unit_info_group)

        # --- Object Details Group (Collapsible) ---
        self.object_details_group = QGroupBox("Object Details")
        object_layout = QVBoxLayout()
        
        self.object_list = QListWidget()
        self.object_list.setMaximumHeight(100) # Limit list height
        self.object_list.itemClicked.connect(self.on_object_selected)
        object_layout.addWidget(self.object_list)
        
        self.object_details_text = QTextEdit()
        self.object_details_text.setMaximumHeight(100) # Limit details height
        self.object_details_text.setReadOnly(True)
        object_layout.addWidget(self.object_details_text)
        
        self.object_details_group.setLayout(object_layout)
        
        # Wrapper for collapse
        self.object_details_toggle = QPushButton("Show Object Details")
        self.object_details_toggle.setCheckable(True)
        self.object_details_toggle.setChecked(True) # Checked = Collapsed (logic below)
        self.object_details_toggle.clicked.connect(lambda: self.toggle_group(self.object_details_group, self.object_details_toggle))
        self.object_details_toggle.setProperty("class", "panel_toggle")
        
        # Start collapsed
        self.object_details_group.setVisible(False) 
        
        layout.addWidget(self.object_details_toggle)
        layout.addWidget(self.object_details_group)

        layout.addStretch()

    def toggle_group(self, group_box: QGroupBox, button: QPushButton):
        """Toggles visibility of a group box."""
        is_visible = group_box.isVisible()
        group_box.setVisible(not is_visible)
        
        # Update button text based on new state
        base_text = group_box.title()
        if not is_visible: # It was hidden, now shown
            button.setText(f"Hide {base_text}")
        else:
            button.setText(f"Show {base_text}")

    # ... (rest of methods like update_unit_info keep same logic, just ensure widget refs exist)

    def update_unit_info(self, unit: Optional[UnitData]):
        """Update the panel based on the selected unit (or None to clear)."""
        self.current_unit = unit
        self.object_list.clear() # Clear previous items
        self.object_details_text.clear() # Clear details text

        if unit and unit.data: # Check if unit and its data exist
            self.unit_id_label.setText(f"Unit ID: {unit.data.get('id', 'N/A')}")

            # Format bounds correctly
            min_coords = unit.data.get('min')
            max_coords = unit.data.get('max')
            if isinstance(min_coords, list) and len(min_coords) == 3 and \
               isinstance(max_coords, list) and len(max_coords) == 3:
                self.unit_bounds_label.setText(
                    f"Bounds:\n"
                    f"  X: ({min_coords[0]:.1f} to {max_coords[0]:.1f})\n"
                    f"  Y: ({min_coords[1]:.1f} to {max_coords[1]:.1f})\n"
                    f"  Z: ({min_coords[2]:.1f} to {max_coords[2]:.1f})"
                )
            else:
                 self.unit_bounds_label.setText("Bounds: N/A")

            # Count and list objects
            objects = unit.data.get('objects', {})
            total_objects = sum(len(obj_list) for obj_list in objects.values())
            self.unit_objects_label.setText(f"Total Objects: {total_objects}")

            for category, obj_list in objects.items():
                if obj_list: # Only process if list is not empty
                    for i, obj in enumerate(obj_list):
                        obj_name_json = obj.get('name')
                        # Use a unique fallback name if JSON name missing/duplicate
                        obj_name_display = obj_name_json if obj_name_json else f"obj_{i}"
                        actor_name = f"unit_{unit.id}_{category}_{obj_name_display}"

                        item = QListWidgetItem(f"{category}: {obj_name_display}")
                        # Store category dict, object dict, and unique actor name
                        item.setData(Qt.ItemDataRole.UserRole, (category, obj, actor_name))
                        self.object_list.addItem(item)
        else:
            # Clear fields if no unit is selected or data is missing
            self.unit_id_label.setText("Unit ID: None")
            self.unit_bounds_label.setText("Bounds: None")
            self.unit_objects_label.setText("Objects: None")

    def on_object_selected(self, item: QListWidgetItem):
        """Display details of the selected object and highlight it."""
        if item is None: return
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None or len(data) != 3: return # Basic validation

        category, obj, actor_name = data

        # Update details text
        details = f"Category: {category}\n"
        details += f"Name: {obj.get('name', 'N/A')}\n"
        min_coords = obj.get('min')
        max_coords = obj.get('max')
        if isinstance(min_coords, list) and len(min_coords) == 3 and \
           isinstance(max_coords, list) and len(max_coords) == 3:
            details += f"Min: {tuple(round(x, 1) for x in min_coords)}\n"
            details += f"Max: {tuple(round(x, 1) for x in max_coords)}\n"
            try:
                size = np.subtract(max_coords, min_coords)
                details += f"Dimensions: {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f}\n"
            except Exception:
                details += "Dimensions: Error calculating\n" # Handle potential errors
        else:
            details += "Coordinates: N/A\n"
        self.object_details_text.setText(details)

        # Trigger highlighting in SceneManager
        self.scene_manager.highlight_object(actor_name)


class VisibilityPanel(QWidget):
    """Panel for controlling unit and object category visibility."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self.scene_manager = scene_manager
        self.unit_checkboxes: Dict[int, QCheckBox] = {}
        self.category_checkboxes: Dict[str, QCheckBox] = {}
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI elements of the panel."""
        layout = QVBoxLayout(self) # Apply layout directly
        layout.setContentsMargins(0, 5, 0, 5)

        # --- Unit Visibility Group ---
        # Use ScrollArea or just compact layout? For few units, VBox is fine.
        # Let's use a group box with a tight layout.
        self.units_group = QGroupBox("Unit Visibility")
        self.units_layout = QVBoxLayout()
        self.units_layout.setContentsMargins(5, 5, 5, 5)
        self.units_layout.setSpacing(2)
        self.units_group.setLayout(self.units_layout)
        layout.addWidget(self.units_group)

        # --- Object Category Visibility Group ---
        self.categories_group = QGroupBox("Object Categories")
        categories_layout = QVBoxLayout()
        categories_layout.setContentsMargins(5, 5, 5, 5)
        categories_layout.setSpacing(2)
        
        for category, settings in self.scene_manager.object_categories.items():
            checkbox = QCheckBox(category)
            checkbox.setChecked(settings.get('visible', True)) # Default to True if key missing
            # Use lambda to correctly capture category in loop
            checkbox.toggled.connect(lambda checked, cat=category: self.toggle_category(cat, checked))
            self.category_checkboxes[category] = checkbox
            categories_layout.addWidget(checkbox)
        self.categories_group.setLayout(categories_layout)
        layout.addWidget(self.categories_group)

        # --- Scene Controls Group ---
        self.scene_group = QGroupBox("Scene Controls")
        scene_layout = QVBoxLayout()
        scene_layout.setContentsMargins(5, 5, 5, 5)
        
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        scene_layout.addWidget(self.reset_view_btn)
        
        self.scene_group.setLayout(scene_layout)
        layout.addWidget(self.scene_group)

        layout.addStretch() # Push controls up

    def add_unit_checkbox(self, unit_id: int, unit_name: str):
        """Add a checkbox for controlling a unit's visibility."""
        if unit_id in self.unit_checkboxes:
            print(f"Checkbox for Unit {unit_id} already exists.")
            return # Avoid adding duplicates

        checkbox = QCheckBox(f"Unit {unit_id}") # Label the checkbox
        checkbox.setChecked(True) # Assume visible when first added
        # Use lambda to capture the correct unit_id for the slot
        checkbox.toggled.connect(lambda checked, uid=unit_id: self.toggle_unit(uid, checked))

        self.unit_checkboxes[unit_id] = checkbox
        self.units_layout.addWidget(checkbox) # Add to the layout

    def remove_unit_checkbox(self, unit_id: int):
        """Remove the checkbox associated with a unit."""
        if unit_id in self.unit_checkboxes:
            checkbox = self.unit_checkboxes.pop(unit_id) # Remove from dict
            self.units_layout.removeWidget(checkbox) # Remove from layout
            checkbox.deleteLater() # Schedule for deletion

    def clear_unit_checkboxes(self):
        """Remove all unit checkboxes from the panel."""
        # Iterate safely while removing items
        for unit_id in list(self.unit_checkboxes.keys()):
            self.remove_unit_checkbox(unit_id)
        # Sanity check: ensure dictionary is empty
        self.unit_checkboxes.clear()

    # --- Slots to connect signals to SceneManager actions ---
    def toggle_unit(self, unit_id: int, visible: bool):
        """Slot to react to unit checkbox changes."""
        print(f"UI: Toggling Unit {unit_id} visibility to {visible}")
        self.scene_manager.toggle_unit_visibility(unit_id, visible)

    def toggle_category(self, category: str, visible: bool):
        """Slot to react to category checkbox changes."""
        print(f"UI: Toggling Category '{category}' visibility to {visible}")
        self.scene_manager.toggle_category_visibility(category, visible)

    def reset_view(self):
        """Slot to reset the camera view."""
        if self.scene_manager.plotter:
            print("UI: Resetting camera view")
            # self.scene_manager.plotter.camera_position = 'iso' # Often too far
            self.scene_manager.plotter.reset_camera()
            self.scene_manager.plotter.render()

    def fit_to_screen(self):
        """Slot to fit the scene to the view (same as reset)."""
        self.reset_view()

class MarkerPanel(QWidget):
    """Panel for placing coordinate markers in the scene."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self.scene_manager = scene_manager
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI elements of the panel."""
        layout = QVBoxLayout(self)
        
        group = QGroupBox("Place Marker")
        group_layout = QVBoxLayout()

        # Create a locale that uses a period for decimals
        c_locale = QLocale(QLocale.Language.C)

        # Coordinate Input
        coords_layout = QHBoxLayout()
        self.coords_input = QLineEdit("0.0, 0.0, 0.0")
        self.coords_input.setPlaceholderText("e.g., 100.5, 200.0, 50.0")
        coords_layout.addWidget(QLabel("Coords [x,y,z]:"))
        coords_layout.addWidget(self.coords_input)
        group_layout.addLayout(coords_layout)

        # Color and Radius Selection
        options_layout = QHBoxLayout()
        self.color_combo = QComboBox()
        self.color_combo.addItems(['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'White'])
        options_layout.addWidget(QLabel("Color:"))
        options_layout.addWidget(self.color_combo)
        
        self.radius_input = QLineEdit("15.0")
        radius_validator = QDoubleValidator(0.001, 1000.0, 4)
        radius_validator.setLocale(c_locale)
        self.radius_input.setValidator(radius_validator)
        options_layout.addWidget(QLabel("Radius:"))
        options_layout.addWidget(self.radius_input)
        group_layout.addLayout(options_layout)

        # Action Buttons
        buttons_layout = QHBoxLayout()
        self.place_btn = QPushButton("Place Marker")
        self.place_btn.clicked.connect(self._on_place_marker)
        self.place_btn.setProperty("class", "primary")
        
        self.clear_btn = QPushButton("Clear All Markers")
        self.clear_btn.clicked.connect(self._on_clear_markers)
        self.clear_btn.setProperty("class", "danger")
        
        buttons_layout.addWidget(self.place_btn)
        buttons_layout.addWidget(self.clear_btn)
        group_layout.addLayout(buttons_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)

    def _on_place_marker(self):
        """Handle the 'Place Marker' button click."""
        try:
            coord_text = self.coords_input.text()
            # Clean the string: remove brackets and strip whitespace
            cleaned_str = coord_text.strip().strip('[]')
            parts = cleaned_str.split(',')

            if len(parts) != 3:
                raise ValueError("Input must contain three comma-separated coordinates.")

            # Convert each part to float
            position = np.array([float(p.strip()) for p in parts])
            
            color = self.color_combo.currentText().lower()
            
            radius = float(self.radius_input.text())
            
            self.scene_manager.add_marker_sphere(position, color, radius)
            
            
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                                "Please enter coordinates in the format 'x, y, z' and a valid radius.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def _on_clear_markers(self):
        """Handle the 'Clear All Markers' button click."""
        self.scene_manager.clear_all_markers()




class PredictionPanel(QWidget):
    """Panel for running GNN predictions on the selected unit."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self.scene_manager = scene_manager
        self.current_unit_path: Optional[str] = None
        self.target_indices: Dict[str, int] = {} # "Name (Index)" -> Index
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        group = QGroupBox("GNN Prediction (wcd enkelvoudig)")
        group_layout = QVBoxLayout()
        
        # Target Selection
        target_label = QLabel("Select Target Object:")
        target_label.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(target_label)
        self.target_combo = QComboBox()
        self.target_combo.setStyleSheet("padding: 5px;")
        group_layout.addWidget(self.target_combo)
        
        group_layout.addSpacing(10)
        
        # Predict Button
        btns_layout = QHBoxLayout()
        self.predict_btn = QPushButton("Predict Position")
        self.predict_btn.clicked.connect(self._on_predict)
        self.predict_btn.setProperty("class", "primary") # Use theme class
        btns_layout.addWidget(self.predict_btn)
        
        # Clear Button
        self.clear_btn = QPushButton("Clear Prediction")
        self.clear_btn.clicked.connect(self._on_clear_prediction)
        self.clear_btn.setProperty("class", "danger") # Use theme class
        btns_layout.addWidget(self.clear_btn)
        
        group_layout.addLayout(btns_layout)
        
        group_layout.addSpacing(10)

        # Results Display
        result_label = QLabel("Results:")
        result_label.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(result_label)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(120)
        self.result_text.setPlaceholderText("Prediction results will appear here...")
        group_layout.addWidget(self.result_text)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()

    def update_targets(self, unit_path: str):
        """Populates the target dropdown for the new unit."""
        self.current_unit_path = unit_path
        self.target_combo.clear()
        self.result_text.clear()
        
        if not unit_path:
            return
            
        try:
            # Get targets from evaluate module (which reads existing wcd enkelvoudig)
            # targets is dict {index: name_str}
            targets = evaluate.get_available_targets(unit_path, "wcd enkelvoudig")
            
            self.target_indices.clear()
            for idx, name_display in targets.items():
                self.target_combo.addItem(name_display)
                self.target_indices[name_display] = idx
                
            if not targets:
                self.target_combo.addItem("No valid targets found")
                self.predict_btn.setEnabled(False)
            else:
                self.predict_btn.setEnabled(True)
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to list targets: {e}")
            self.predict_btn.setEnabled(False)

    def _on_predict(self):
        """Runs prediction for the selected target."""
        if not self.current_unit_path:
            return
            
        target_name = self.target_combo.currentText()
        if target_name not in self.target_indices:
            return
            
        target_idx = self.target_indices[target_name]
        
        self.result_text.setText("Running prediction...")
        QApplication.processEvents()
        
        # Call evaluate.predict_component
        result = evaluate.predict_component(self.current_unit_path, target_idx)
        
        if result['status'] == 'error':
            self.result_text.setText(f"Error: {result.get('message')}")
            QMessageBox.critical(self, "Prediction Error", result.get('message'))
            return
            
        # Success
        pred_mm = np.array(result['pred_pos_mm'])
        true_mm = np.array(result['true_pos_mm'])
        error = result['error_mm']
        
        # Display Text
        msg = (f"Target Index: {target_idx}\n"
               f"Error: {error:.1f} mm\n"
               f"True Surface: {result['true_surface_id']}\n"
               f"Pred Surface: {result['pred_surface_id']}\n"
               f"True Pos: {true_mm[0]:.0f}, {true_mm[1]:.0f}, {true_mm[2]:.0f}\n"
               f"Pred Pos: {pred_mm[0]:.0f}, {pred_mm[1]:.0f}, {pred_mm[2]:.0f}")
        self.result_text.setText(msg)
        
        # Visual markers
        # Clear old prediction markers if any? Or keep accumulating? User might want to compare.
        # Maybe clear just "pred" markers? existing MarkerPanel clears *all* "marker_*" actors.
        # We can use a specific prefix to manage them if needed, but for now simple markers.
        
        # Add True Marker (Green)
        self.scene_manager.add_marker_sphere(true_mm, color='green', radius=40.0)
        self.scene_manager.add_marker_label(true_mm, "True Position", color='green')
        
        # Add Pred Marker (Red/Orange)
        self.scene_manager.add_marker_sphere(pred_mm, color='orange', radius=40.0)
        self.scene_manager.add_marker_label(pred_mm, "Predicted Position", color='orange')
        
        # Force render
        if self.scene_manager.plotter:
            self.scene_manager.plotter.render()

    def _on_clear_prediction(self):
        """Clears all prediction markers and text."""
        self.scene_manager.clear_all_markers()
        self.result_text.clear()