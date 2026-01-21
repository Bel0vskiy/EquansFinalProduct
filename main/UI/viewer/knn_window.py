
import sys
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QDoubleSpinBox, QSlider, QPushButton, QFrame, 
    QSplitter, QGroupBox, QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal
import pyvista as pv

# Import SceneManager for 3D visualization
from .scene_manager import SceneManager
try:
    from ...Model.knn_loader import KnnLoader
except ImportError:
    from Model.knn_loader import KnnLoader

class KNNViewerWidget(QWidget):
    """
    Widget for manual room generation and asset prediction using the KNN model.
    Replicates the functionality of the Streamlit app.
    """
    status_message = Signal(str)

    def __init__(self):
        super().__init__()
        
        self.scene_manager = SceneManager()
        self.loader = KnnLoader.get_instance()
    
        try:
            self.loader.load_count_model()
            self.loader.load_socket_count_model()
        except Exception as e:
            print(f"Error loading models on init: {e}")

        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        

        dim_group = QGroupBox("Room Dimensions")
        dim_layout = QVBoxLayout()
        

        dim_layout.addWidget(QLabel("Length (m):"))
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(0.1, 50.0)
        self.len_spin.setValue(6.0)
        self.len_spin.setSingleStep(0.1)
        dim_layout.addWidget(self.len_spin)
        

        dim_layout.addWidget(QLabel("Width (m):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 50.0)
        self.width_spin.setValue(4.0)
        self.width_spin.setSingleStep(0.1)
        dim_layout.addWidget(self.width_spin)

        dim_layout.addWidget(QLabel("Height (m):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 20.0)
        self.height_spin.setValue(2.7)
        self.height_spin.setSingleStep(0.1)
        dim_layout.addWidget(self.height_spin)
        
        dim_group.setLayout(dim_layout)
        left_layout.addWidget(dim_group)
        

        settings_group = QGroupBox("Model Settings")
        settings_layout = QVBoxLayout()
        
        settings_layout.addWidget(QLabel("Nearest Neighbors (K):"))
        k_layout = QHBoxLayout()
        self.k_slider = QSlider(Qt.Orientation.Horizontal)
        self.k_slider.setRange(1, 30)
        self.k_slider.setValue(7)
        self.k_label = QLabel("7")
        self.k_slider.valueChanged.connect(lambda v: self.k_label.setText(str(v)))
        k_layout.addWidget(self.k_slider)
        k_layout.addWidget(self.k_label)
        settings_layout.addLayout(k_layout)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        

        self.gen_btn = QPushButton("Generate Components")
        self.gen_btn.clicked.connect(self.generate_room)
        self.gen_btn.setProperty("class", "primary")
        self.gen_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        left_layout.addWidget(self.gen_btn)
        
        self.clear_btn = QPushButton("Clear Scene")
        self.clear_btn.clicked.connect(self.clear_scene)
        self.clear_btn.setProperty("class", "danger")
        left_layout.addWidget(self.clear_btn)
        

        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)

        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_panel)
        scroll.setMinimumWidth(280)
        
        splitter.addWidget(scroll)


        self.viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(self.viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        

        self.plotter = self.scene_manager.initialize_plotter(self.viewer_frame)
        viewer_layout.addWidget(self.plotter.interactor)
        
        splitter.addWidget(self.viewer_frame)
        
        splitter.setSizes([300, 900])
        splitter.setStretchFactor(1, 1)

    def generate_room(self):
        """Run prediction and visualize results."""
        self.result_text.clear()
        self.status_message.emit("Generating room...")
        
        L = self.len_spin.value()
        W = self.width_spin.value()
        H = self.height_spin.value()
        k = self.k_slider.value()
        
        try:
            count_model = self.loader.load_count_model()
            lamp_placer = self.loader.load_placer(k=k, use_count_in_knn=True)
            
            socket_count_model = self.loader.load_socket_count_model()
            socket_placer = self.loader.load_socket_placer(k=k, use_count_in_knn=True)

            lamps_m, n_lamps = lamp_placer.predict_room(L, W, H, count_model)
            sockets_m, n_sockets = socket_placer.predict_room(L, W, H, socket_count_model)

            msg = f"Room: {L} x {W} x {H} m\n"
            msg += f"Predicted Lamps: {n_lamps}\n"
            msg += f"Predicted Sockets: {n_sockets}\n\n"
            
            if len(lamps_m) > 0:
                msg += "Lamps (x, y, z):\n"
                for i, p in enumerate(lamps_m):
                    msg += f"  {i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\n"
            
            if len(sockets_m) > 0:
                msg += "\nSockets (x, y, z):\n"
                for i, p in enumerate(sockets_m):
                    msg += f"  {i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\n"
            
            self.result_text.setText(msg)

            self.visualize_prediction(L, W, H, lamps_m, sockets_m)
            self.status_message.emit(f"Generated {n_lamps} lamps and {n_sockets} sockets.")
            
        except Exception as e:
            err_msg = f"Error during generation: {e}"
            self.result_text.setText(err_msg)
            self.status_message.emit("Generation failed.")
            print(err_msg)

    def visualize_prediction(self, L, W, H, lamps_m, sockets_m):
        """Update 3D scene with wireframe room and asset markers."""
        self.scene_manager.remove_all_units()
        
        plotter = self.scene_manager.plotter
        plotter.clear()

        bounds = [0, W, 0, L, 0, H] # xmin, xmax, ymin, ymax, zmin, zmax
        plotter.add_box_axes()
        plotter.show_grid()

        box = pv.Box(bounds=bounds)
        plotter.add_mesh(box, style='wireframe', color='white', line_width=2)

        if len(lamps_m) > 0:
            for i, p in enumerate(lamps_m):
                marker = pv.Cube(center=p, x_length=0.2, y_length=0.2, z_length=0.2)
                plotter.add_mesh(marker, color='yellow', label='Lamps' if i==0 else None)
                plotter.add_point_labels([p], [f"L{i+1}"], font_size=10, text_color='white', point_color='yellow')

        if len(sockets_m) > 0:
            for i, p in enumerate(sockets_m):
                marker = pv.Sphere(radius=0.1, center=p)
                plotter.add_mesh(marker, color='cyan', label='Sockets' if i==0 else None)
                plotter.add_point_labels([p], [f"S{i+1}"], font_size=10, text_color='cyan', point_color='cyan')

        plotter.reset_camera()
        plotter.render()

    def clear_scene(self):
        """Clear the 3D view."""
        self.scene_manager.plotter.clear()
        self.result_text.clear()
        self.status_message.emit("Scene cleared.")

    def cleanup(self):
        if self.scene_manager.plotter:
            self.scene_manager.plotter.close()
