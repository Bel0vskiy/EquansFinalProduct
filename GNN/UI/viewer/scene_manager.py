import pyvista as pv
from pyvistaqt import QtInteractor
import numpy as np
from typing import Dict, Set, Optional, Any
import traceback

from .unit_data import UnitData


class SceneManager:

    def __init__(self):
        self.plotter: Optional[QtInteractor] = None
        self.units: Dict[int, UnitData] = {}
        self.visible_units: Set[int] = set()
        self.object_categories: Dict[str, Dict[str, Any]] = {
            'ET': {'visible': True, 'color': 'blue'},
        }
        self.highlighted_actor_name: Optional[str] = None
        self.highlighted_actor_ref = None
        self.original_color: Optional[str] = None
        self.highlight_color: str = 'yellow'
        self.marker_count = 0
        self.bounding_box_actor_name = "drawn_bounding_box"

    def initialize_plotter(self, parent_widget) -> QtInteractor:
        if self.plotter is None:
            self.plotter = QtInteractor(parent_widget)
            self.plotter.set_background('lightgray')
            self.plotter.add_axes()
            print("Plotter initialized.")
        return self.plotter

    def add_unit(self, unit: UnitData):
        if unit.id in self.units:
            print(f"Unit {unit.id} is already loaded.")
            return
        if not unit.loaded or unit.mesh is None:
            print(f"Cannot add Unit {unit.id}: Mesh not loaded.")
            return

        print(f"Adding Unit {unit.id} to scene...")
        self.units[unit.id] = unit
        self.visible_units.add(unit.id)

        is_unit_visible = True
        try:
            actor = self.plotter.add_mesh(
                unit.mesh,
                color='darkgrey',
                style='wireframe',
                opacity=0.2,
                line_width=1,
                name=f"unit_{unit.id}_mesh",
                pickable=False
            )
            actor.SetVisibility(is_unit_visible)

            if unit.data and 'objects' in unit.data:
                self._add_object_categories_for_unit(unit, is_unit_visible)

            self.plotter.reset_camera()
            self.plotter.render()
            print(f"Unit {unit.id} added and camera reset.")

        except Exception as e:
            print(f"Error adding unit {unit.id} actors: {e}")
            traceback.print_exc()
            self.remove_unit(unit.id)

    def remove_unit(self, unit_id: int):
        if unit_id not in self.units:
            return

        print(f"Removing Unit {unit_id}...")
        actor_name_prefix = f"unit_{unit_id}_"
        actors_to_remove = [name for name in self.plotter.actors if name.startswith(actor_name_prefix)]
        for name in actors_to_remove:
            self.plotter.remove_actor(name, render=False)

        del self.units[unit_id]
        self.visible_units.discard(unit_id)

        if self.highlighted_actor_name and self.highlighted_actor_name.startswith(actor_name_prefix):
            self.highlighted_actor_name = None
            self.highlighted_actor_ref = None
            self.original_color = None

        self.plotter.render()
        print(f"Unit {unit_id} removed.")
        if self.units:
            self.plotter.reset_camera()

    def remove_all_units(self):
        print("Removing all units...")
        self.units.clear()
        self.visible_units.clear()
        self.highlighted_actor_name = None
        self.highlighted_actor_ref = None
        self.original_color = None
        if self.plotter:
            self.plotter.clear_actors()
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
                is_object_actor = "_mesh" not in name
                should_be_visible = visible
                if is_object_actor:
                    try:
                        category = name.split('_')[2]
                        if category in self.object_categories:
                            should_be_visible = visible and self.object_categories[category]['visible']
                    except IndexError:
                        pass

                actor.SetVisibility(should_be_visible)

                if not should_be_visible and self.highlighted_actor_name == name:
                    self.highlighted_actor_name = None
                    self.highlighted_actor_ref = None
                    self.original_color = None

        if unit_actors_found:
            if visible:
                self.visible_units.add(unit_id)
            else:
                self.visible_units.discard(unit_id)
            self.plotter.render()
        else:
            print(f"Warning: No actors found for unit {unit_id} to toggle visibility.")

    def toggle_category_visibility(self, category: str, visible: bool):
        if category not in self.object_categories: return
        print(f"Setting Category '{category}' visibility to {visible}")

        self.object_categories[category]['visible'] = visible
        actor_name_suffix = f"_{category}_"

        category_actors_found = False
        for name, actor in self.plotter.actors.items():
            if actor_name_suffix in name:
                category_actors_found = True
                try:
                    unit_id_str = name.split('_')[1]
                    unit_id = int(unit_id_str)
                    if unit_id in self.visible_units:
                        actor.SetVisibility(visible)
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
                for i, obj in enumerate(objects):
                    obj_name_json = obj.get('name')
                    obj_name_unique = obj_name_json if obj_name_json else f"obj_{i}"
                    actor_name = f"unit_{unit.id}_{category}_{obj_name_unique}"

                    if 'min' in obj and 'max' in obj:
                        try:
                            min_coords = np.array(obj['min'], dtype=float)
                            max_coords = np.array(obj['max'], dtype=float)
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
                                opacity=0.7,
                                name=actor_name,
                                pickable=True
                            )
                            actor.SetVisibility(is_unit_visible and is_category_visible)
                        except Exception as e:
                            print(f"Error creating box for '{actor_name}': {e}")
                            traceback.print_exc()
                    else:
                        print(f"Warning: Skipping object '{actor_name}': Missing 'min' or 'max' coordinates.")

    def highlight_object(self, actor_name: Optional[str]):
        """Highlights the specified actor, resetting the previous one."""
        if not self.plotter: return

        if self.highlighted_actor_ref and self.original_color:
            try:
                if self.highlighted_actor_name in self.plotter.actors:
                    self.highlighted_actor_ref.prop.color = self.original_color
                else:
                    pass
            except Exception as e:
                print(f"Info: Could not reset color for previous actor {self.highlighted_actor_name}: {e}")

        self.highlighted_actor_name = None
        self.highlighted_actor_ref = None
        self.original_color = None

        if actor_name:
            try:
                actor = self.plotter.actors.get(actor_name)
                if actor:
                    if not actor.GetVisibility():
                        print(f"Info: Cannot highlight '{actor_name}', it is currently hidden.")
                        self.plotter.render()
                        return

                    self.highlighted_actor_name = actor_name
                    self.highlighted_actor_ref = actor
                    if hasattr(actor, 'prop') and hasattr(actor.prop, 'color'):
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

        self.plotter.render()

    def add_marker_sphere(self, position: np.ndarray, color: str = 'red', radius: float = 10.0):
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

    def clear_all_markers(self):
        if self.plotter is None:
            return
        
        # Find all actor names that start with 'marker_'
        actors_to_remove = [name for name in self.plotter.actors if name.startswith("marker_")]
        
        if not actors_to_remove:
            print("No markers to clear.")
            return

        for name in actors_to_remove:
            self.plotter.remove_actor(name, render=False)
        
        self.plotter.render()
        print(f"Cleared {len(actors_to_remove)} markers.")

    def draw_bounding_box(self, vertices: np.ndarray, color: str = 'cyan'):
        if self.plotter is None:
            return

        faces = np.hstack([
            [4, 0, 1, 2, 3],
            [4, 4, 5, 6, 7],
            [4, 0, 4, 7, 3],
            [4, 1, 5, 6, 2],
            [4, 0, 1, 5, 4],
            [4, 3, 2, 6, 7]
        ])

        box_surface = pv.PolyData(vertices, faces=faces)

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
        if self.plotter and self.bounding_box_actor_name in self.plotter.actors:
            self.plotter.remove_actor(self.bounding_box_actor_name, render=render)
            print("Cleared drawn bounding box.")