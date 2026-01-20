# scene_manager.py

import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
from typing import Dict, Set, Optional, Any
import traceback  # For error printing

# Assuming unit_data.py is in the same directory or accessible via PYTHONPATH
from .unit_data import UnitData


class SceneManager:
    """Manages the PyVista 3D scene, loaded units, and object visibility."""

    def __init__(self):
        self.plotter: Optional[QtInteractor] = None
        self.units: Dict[int, UnitData] = {}
        self.visible_units: Set[int] = set()
        self.object_categories: Dict[str, Dict[str, Any]] = {
            # Define categories, visibility, and colors
            'ET': {'visible': True, 'color': 'blue'},
            # Add other categories when needed, e.g.:
            # 'CV_GKW': {'visible': True, 'color': 'red'},
            # 'lucht': {'visible': True, 'color': 'green'},
            # 'Riolering': {'visible': True, 'color': 'brown'},
            # 'Sanitair': {'visible': True, 'color': 'orange'}
        }
        # --- Highlight state ---
        self.highlighted_actor_name: Optional[str] = None
        self.highlighted_actor_ref = None  # Store direct ref to actor
        self.original_color: Optional[str] = None
        self.highlight_color: str = 'yellow'  # Highlight color
        # ------------------------
        self.marker_count = 0
        self.bounding_box_actor_name = "drawn_bounding_box"

    def initialize_plotter(self, parent_widget) -> QtInteractor:
        """Initialize the PyVista QtInteractor plotter."""
        if self.plotter is None:
            self.plotter = QtInteractor(parent_widget)
            self.plotter.set_background('lightgray')  # Use light gray instead of white?
            self.plotter.add_axes()
            # self.plotter.enable_anti_aliasing('fxaa') # FXAA is generally safe
            print("Plotter initialized.")
        return self.plotter

    def add_unit(self, unit: UnitData):
        """Load a unit's data and add its actors to the scene."""
        if unit.id in self.units:
            print(f"Unit {unit.id} is already loaded.")
            return
        if not unit.loaded or unit.mesh is None:
            print(f"Cannot add Unit {unit.id}: Mesh not loaded.")
            return

        print(f"Adding Unit {unit.id} to scene...")
        self.units[unit.id] = unit
        self.visible_units.add(unit.id)  # Assume visible when added

        # Add actors for the new unit
        is_unit_visible = True  # Initially visible
        try:
            # Add main mesh actor
            actor = self.plotter.add_mesh(
                unit.mesh,
                color='darkgrey',  # Darker grey for room?
                style='wireframe',
                opacity=0.2,  # More transparent?
                line_width=1,
                name=f"unit_{unit.id}_mesh",
                pickable=False  # Room mesh usually not pickable
            )
            actor.SetVisibility(is_unit_visible)

            # Add object categories for this unit
            if unit.data and 'objects' in unit.data:
                self._add_object_categories_for_unit(unit, is_unit_visible)

            self.plotter.reset_camera()  # Adjust camera to fit new unit
            self.plotter.render()
            print(f"Unit {unit.id} added and camera reset.")

        except Exception as e:
            print(f"Error adding unit {unit.id} actors: {e}")
            traceback.print_exc()
            # Clean up if adding failed partially
            self.remove_unit(unit.id)

    def remove_unit(self, unit_id: int):
        """Remove a unit and its actors from the scene."""
        if unit_id not in self.units:
            return

        print(f"Removing Unit {unit_id}...")
        # Remove actors associated with this unit
        actor_name_prefix = f"unit_{unit_id}_"
        actors_to_remove = [name for name in self.plotter.actors if name.startswith(actor_name_prefix)]
        for name in actors_to_remove:
            self.plotter.remove_actor(name, render=False)  # Remove actor without immediate render

        # Remove from internal tracking
        del self.units[unit_id]
        self.visible_units.discard(unit_id)

        # Reset highlight if the removed unit contained the highlighted object
        if self.highlighted_actor_name and self.highlighted_actor_name.startswith(actor_name_prefix):
            self.highlighted_actor_name = None
            self.highlighted_actor_ref = None
            self.original_color = None

        self.plotter.render()  # Render after removing all actors for this unit
        print(f"Unit {unit_id} removed.")
        if self.units:  # Only reset camera if something is left
            self.plotter.reset_camera()

    def remove_all_units(self):
        """Remove all units and clear the plotter."""
        print("Removing all units...")
        self.units.clear()
        self.visible_units.clear()
        self.highlighted_actor_name = None
        self.highlighted_actor_ref = None
        self.original_color = None
        if self.plotter:
            self.plotter.clear_actors()  # Efficiently remove all actors
            self.plotter.render()
        print("All units removed.")

    def toggle_unit_visibility(self, unit_id: int, visible: bool):
        """Toggle visibility of a specific unit's actors."""
        if unit_id not in self.units: return
        print(f"Setting Unit {unit_id} visibility to {visible}")

        actor_name_prefix = f"unit_{unit_id}_"
        unit_actors_found = False
        for name, actor in self.plotter.actors.items():
            if name.startswith(actor_name_prefix):
                unit_actors_found = True
                # Check category visibility before showing object actors
                is_object_actor = "_mesh" not in name  # Simple check if it's an object box
                should_be_visible = visible
                if is_object_actor:
                    try:
                        # Extract category name (assuming format unit_ID_Category_...)
                        category = name.split('_')[2]
                        if category in self.object_categories:
                            should_be_visible = visible and self.object_categories[category]['visible']
                    except IndexError:
                        pass  # Should not happen if naming convention is followed

                actor.SetVisibility(should_be_visible)

                # Reset highlight if the now hidden actor was highlighted
                if not should_be_visible and self.highlighted_actor_name == name:
                    self.highlighted_actor_name = None
                    self.highlighted_actor_ref = None
                    self.original_color = None  # Color will be correct when re-shown

        if unit_actors_found:
            if visible:
                self.visible_units.add(unit_id)
            else:
                self.visible_units.discard(unit_id)
            self.plotter.render()
        else:
            print(f"Warning: No actors found for unit {unit_id} to toggle visibility.")

    def toggle_category_visibility(self, category: str, visible: bool):
        """Toggle visibility of an object category across all visible units."""
        if category not in self.object_categories: return
        print(f"Setting Category '{category}' visibility to {visible}")

        self.object_categories[category]['visible'] = visible
        actor_name_suffix = f"_{category}_"  # Part of the actor name structure

        category_actors_found = False
        for name, actor in self.plotter.actors.items():
            if actor_name_suffix in name:  # Check if actor belongs to this category
                category_actors_found = True
                try:
                    # Only change visibility if its unit is supposed to be visible
                    unit_id_str = name.split('_')[1]
                    unit_id = int(unit_id_str)
                    if unit_id in self.visible_units:
                        actor.SetVisibility(visible)
                        # Reset highlight if the now hidden actor was highlighted
                        if not visible and self.highlighted_actor_name == name:
                            self.highlighted_actor_name = None
                            self.highlighted_actor_ref = None
                            self.original_color = None
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse unit ID from actor name '{name}'")

        if category_actors_found:
            self.plotter.render()
        else:
            print(f"Info: No actors found for category '{category}'.")

    def _add_object_categories_for_unit(self, unit: UnitData, is_unit_visible: bool):
        """Helper to add object category actors for a specific unit."""
        if not unit.data or 'objects' not in unit.data:
            return

        for category, settings in self.object_categories.items():
            is_category_visible = settings['visible']

            if category in unit.data['objects']:
                objects = unit.data['objects'][category]
                for i, obj in enumerate(objects):  # Use enumerate for unique fallback names
                    # Ensure a unique name even if JSON name is missing or duplicated
                    obj_name_json = obj.get('name')
                    obj_name_unique = obj_name_json if obj_name_json else f"obj_{i}"
                    actor_name = f"unit_{unit.id}_{category}_{obj_name_unique}"

                    if 'min' in obj and 'max' in obj:
                        try:
                            min_coords = np.array(obj['min'], dtype=float)
                            max_coords = np.array(obj['max'], dtype=float)

                            # Validate coordinates
                            if min_coords.shape != (3,) or max_coords.shape != (3,):
                                print(f"Warning: Skipping object '{actor_name}': Invalid coordinate dimensions.")
                                continue

                            size = max_coords - min_coords
                            if np.any(size <= 1e-6):  # Use tolerance for size check
                                print(f"Warning: Skipping object '{actor_name}' due to near-zero size: {size}")
                                continue

                            box = pv.Box(bounds=[min_coords[0], max_coords[0],
                                                 min_coords[1], max_coords[1],
                                                 min_coords[2], max_coords[2]])
                            actor = self.plotter.add_mesh(
                                box,
                                color=settings['color'],
                                opacity=0.7,  # Slightly less transparent?
                                name=actor_name,
                                pickable=True  # Make object boxes pickable
                            )
                            # Set initial visibility based on both unit and category
                            actor.SetVisibility(is_unit_visible and is_category_visible)
                        except Exception as e:
                            print(f"Error creating box for '{actor_name}': {e}")
                            traceback.print_exc()
                    else:
                        print(f"Warning: Skipping object '{actor_name}': Missing 'min' or 'max' coordinates.")

    def highlight_object(self, actor_name: Optional[str]):
        """Highlights the specified actor, resetting the previous one."""
        if not self.plotter: return

        # --- Reset previous highlight ---
        # Use stored reference if available and valid
        if self.highlighted_actor_ref and self.original_color:
            try:
                # Check if actor still exists in plotter before accessing prop
                if self.highlighted_actor_name in self.plotter.actors:
                    self.highlighted_actor_ref.prop.color = self.original_color
                else:
                    # Actor was removed, just clear state
                    pass
            except Exception as e:
                print(f"Info: Could not reset color for previous actor {self.highlighted_actor_name}: {e}")

        # Clear current highlight state regardless
        self.highlighted_actor_name = None
        self.highlighted_actor_ref = None
        self.original_color = None
        # --------------------------------

        # --- Find and highlight the new actor (if provided) ---
        if actor_name:
            try:
                actor = self.plotter.actors.get(actor_name)
                if actor:
                    # Check visibility before highlighting
                    if not actor.GetVisibility():
                        print(f"Info: Cannot highlight '{actor_name}', it is currently hidden.")
                        self.plotter.render()  # Render to ensure previous highlight is reset
                        return

                    self.highlighted_actor_name = actor_name
                    self.highlighted_actor_ref = actor
                    if hasattr(actor, 'prop') and hasattr(actor.prop, 'color'):
                        # Use pv.Color to handle potential color name/tuple/list issues
                        self.original_color = pv.Color(actor.prop.color)
                        actor.prop.color = self.highlight_color
                        print(f"Highlighted: {actor_name}")
                    else:
                        print(f"Warning: Actor '{actor_name}' prop has no color attribute.")
                else:
                    print(f"Warning: Actor '{actor_name}' not found for highlighting.")
            except Exception as e:
                print(f"Error highlighting actor '{actor_name}': {e}")
                traceback.print_exc()
        # -----------------------------------------

        self.plotter.render()  # Update render after changing colors or just resetting

    def add_marker_sphere(self, position: np.ndarray, color: str = 'red', radius: float = 10.0):
        """Adds a sphere marker to the scene at a given position."""
        if self.plotter is None:
            return
        
        try:
            sphere = pv.Sphere(radius=radius, center=position)
            marker_name = f"marker_{self.marker_count}"
            self.plotter.add_mesh(sphere, color=color, name=marker_name)
            self.marker_count += 1
            print(f"Added marker {marker_name} at {position}")
        except Exception as e:
            print(f"Error adding marker sphere: {e}")

    def add_marker_label(self, position: np.ndarray, text: str, color: str = 'black'):
        """Adds a text label at the given position."""
        if self.plotter is None:
            return
        
        try:
            # Shift label slightly so it doesn't overlap exactly with the center
            # label_pos = position + np.array([0, 0, radius * 1.5]) 
            
            marker_name = f"marker_label_{self.marker_count}"
            
            self.plotter.add_point_labels(
                [position], 
                [text],
                point_size=0,
                name=marker_name,
                always_visible=True,
                show_points=False,
                text_color=color,
                shape_opacity=0.5
            )
            self.marker_count += 1
            print(f"Added label '{text}' at {position}")
        except Exception as e:
            print(f"Error adding marker label: {e}")
            traceback.print_exc()

    def clear_all_markers(self):
        """Removes all sphere markers from the scene."""
        if self.plotter is None:
            return
        
        # Find all actor names that start with 'marker_' (covers 'marker_label_' too)
        actors_to_remove = [name for name in self.plotter.actors if name.startswith("marker_")]
        
        if not actors_to_remove:
            print("No markers to clear.")
            return

        for name in actors_to_remove:
            self.plotter.remove_actor(name, render=False)
        
        self.plotter.render()
        print(f"Cleared {len(actors_to_remove)} markers.")

    def draw_bounding_box(self, vertices: np.ndarray, color: str = 'cyan'):
        """Draws a bounding box from 8 vertices."""
        if self.plotter is None:
            return

        # Vertices must be in a specific order for faces to form correctly.
        # This order matches the output of the get_bbox_corners function.
        # 0: min_x, min_y, min_z
        # 1: max_x, min_y, min_z
        # 2: max_x, max_y, min_z
        # 3: min_x, max_y, min_z
        # 4: min_x, min_y, max_z
        # 5: max_x, min_y, max_z
        # 6: max_x, max_y, max_z
        # 7: min_x, max_y, max_z
        
        # Define the 6 faces of the cube using the vertex indices
        faces = np.hstack([
            [4, 0, 1, 2, 3],  # Front face
            [4, 4, 5, 6, 7],  # Back face
            [4, 0, 4, 7, 3],  # Left face
            [4, 1, 5, 6, 2],  # Right face
            [4, 0, 1, 5, 4],  # Bottom face
            [4, 3, 2, 6, 7]   # Top face
        ])

        box_surface = pv.PolyData(vertices, faces=faces)

        # Remove any previously drawn box without immediate rendering
        self.clear_bounding_box(render=False)

        self.plotter.add_mesh(
            box_surface,
            color=color,
            style='surface',
            opacity=0.6,
            name=self.bounding_box_actor_name
        )
        print(f"Added bounding box with color {color}")
        self.plotter.render()

    def clear_bounding_box(self, render: bool = True):
        """Removes the drawn bounding box from the scene."""
        if self.plotter and self.bounding_box_actor_name in self.plotter.actors:
            self.plotter.remove_actor(self.bounding_box_actor_name, render=render)
            print("Cleared drawn bounding box.")