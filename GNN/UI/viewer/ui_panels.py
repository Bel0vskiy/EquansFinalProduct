# ui_panels.py
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QListWidget, QTextEdit,
    QCheckBox, QPushButton, QListWidgetItem, QHBoxLayout, QLineEdit,
    QComboBox, QMessageBox
)
from PySide6.QtGui import QDoubleValidator
from PySide6.QtCore import Qt, QLocale
from typing import Optional, Dict

# Import SceneManager for type hinting and interaction
# Assuming scene_manager.py is in the same directory or accessible
from .scene_manager import SceneManager
from .unit_data import UnitData # Import UnitData for type hinting

class PropertiesPanel(QWidget):
    """Information properties panel to display unit and object details."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self.scene_manager = scene_manager # Store reference for highlighting
        self.current_unit: Optional[UnitData] = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI elements of the panel."""
        layout = QVBoxLayout(self) # Apply layout directly to self

        # --- Unit Information Group ---
        self.unit_info_group = QGroupBox("Unit Information")
        unit_layout = QVBoxLayout()
        self.unit_id_label = QLabel("Unit ID: None")
        self.unit_bounds_label = QLabel("Bounds: None")
        self.unit_objects_label = QLabel("Objects: None")
        unit_layout.addWidget(self.unit_id_label)
        unit_layout.addWidget(self.unit_bounds_label)
        unit_layout.addWidget(self.unit_objects_label)
        self.unit_info_group.setLayout(unit_layout)
        layout.addWidget(self.unit_info_group)

        # --- Object Details Group ---
        self.object_details_group = QGroupBox("Object Details")
        object_layout = QVBoxLayout()
        self.object_list = QListWidget()
        self.object_list.itemClicked.connect(self.on_object_selected)
        object_layout.addWidget(self.object_list)
        self.object_details_text = QTextEdit()
        self.object_details_text.setMaximumHeight(150) # Limit height
        self.object_details_text.setReadOnly(True) # Make read-only
        object_layout.addWidget(self.object_details_text)
        self.object_details_group.setLayout(object_layout)
        layout.addWidget(self.object_details_group)

        layout.addStretch() # Push content upwards

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

        # --- Unit Visibility Group ---
        self.units_group = QGroupBox("Unit Visibility")
        self.units_layout = QVBoxLayout() # We'll add checkboxes here later
        self.units_group.setLayout(self.units_layout)
        layout.addWidget(self.units_group)

        # --- Object Category Visibility Group ---
        self.categories_group = QGroupBox("Object Categories")
        categories_layout = QVBoxLayout()
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
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        scene_layout.addWidget(self.reset_view_btn)
        # self.fit_to_screen_btn = QPushButton("Fit to Screen") # Often redundant
        # self.fit_to_screen_btn.clicked.connect(self.fit_to_screen)
        # scene_layout.addWidget(self.fit_to_screen_btn)
        # Add Toggle Axes button if needed
        # self.toggle_axes_btn = QPushButton("Toggle Axes")
        # self.toggle_axes_btn.clicked.connect(self.toggle_axes)
        # scene_layout.addWidget(self.toggle_axes_btn)
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
        self.clear_btn = QPushButton("Clear All Markers")
        self.clear_btn.clicked.connect(self._on_clear_markers)
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


class BoundingBoxPanel(QWidget):
    """Panel for drawing a bounding box from a list of coordinates."""

    def __init__(self, scene_manager: SceneManager):
        super().__init__()
        self.scene_manager = scene_manager
        self._init_ui()

    def _init_ui(self):
        """Initialize the UI elements of the panel."""
        layout = QVBoxLayout(self)

        group = QGroupBox("Draw Bounding Box")
        group_layout = QVBoxLayout()

        # Coordinate Input
        self.coords_input = QTextEdit()
        self.coords_input.setPlaceholderText(
            "Paste 8 corner coordinates, e.g.,\n"
            "[x1, y1, z1],\n"
            "[x2, y2, z2],\n"
            "..."
        )
        self.coords_input.setMinimumHeight(150) # Taller input area
        group_layout.addWidget(QLabel("8 Bounding Box Corners:"))
        group_layout.addWidget(self.coords_input)

        # Color Selection
        color_layout = QHBoxLayout()
        self.color_combo = QComboBox()
        self.color_combo.addItems(['Cyan', 'Magenta', 'Green', 'Blue', 'Red'])
        color_layout.addWidget(QLabel("Color:"))
        color_layout.addWidget(self.color_combo)
        group_layout.addLayout(color_layout)

        # Action Buttons
        buttons_layout = QHBoxLayout()
        self.draw_btn = QPushButton("Draw Box")
        self.draw_btn.clicked.connect(self._on_draw_box)
        self.clear_btn = QPushButton("Clear Box")
        self.clear_btn.clicked.connect(self._on_clear_box)
        buttons_layout.addWidget(self.draw_btn)
        buttons_layout.addWidget(self.clear_btn)
        group_layout.addLayout(buttons_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _on_draw_box(self):
        """Handle the 'Draw Box' button click."""
        try:
            text = self.coords_input.toPlainText()
            
            # --- Robust Parsing ---
            # 1. Remove brackets and newlines
            cleaned_text = text.replace('[', '').replace(']', '').replace('\n', '')
            # 2. Split by comma
            parts = cleaned_text.split(',')
            # 3. Filter out empty strings that result from trailing commas etc.
            nums = [p.strip() for p in parts if p.strip()]

            if len(nums) != 24: # 8 corners * 3 coords
                raise ValueError(f"Expected 24 numbers (8 vertices of 3 coordinates), but found {len(nums)}.")

            # Convert to a numpy array and reshape
            vertices = np.array([float(n) for n in nums]).reshape(8, 3)
            
            color = self.color_combo.currentText().lower()

            # Call SceneManager to draw the box
            self.scene_manager.draw_bounding_box(vertices, color=color)

        except ValueError as ve:
            QMessageBox.warning(self, "Invalid Input", 
                                f"Could not parse coordinates. Please ensure you provide 8 vertices.\n\nError: {ve}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {e}")

    def _on_clear_box(self):
        """Handle the 'Clear Box' button click."""
        self.scene_manager.clear_bounding_box()

    # def toggle_axes(self): # Implement if needed
    #     if self.scene_manager.plotter:
    #         # Example: toggle the axes widget
    #         pass