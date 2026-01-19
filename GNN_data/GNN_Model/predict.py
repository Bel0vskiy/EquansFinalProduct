import os
import json
import torch
import numpy as np
import pyvista as pv
from typing import Dict, List, Tuple, Optional, Union
import logging

from .socket_gnn import SocketGNN
from .room_graph import build_room_graph
from .component_types import build_component_types, build_name_to_index

logger = logging.getLogger(__name__)


class SocketPredictor:
    """
    Predictor for socket placement using trained GNN model
    """
    
    def __init__(
        self,
        model_path: str,
        data_analysis_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model checkpoint
            data_analysis_dir: Directory containing component CSV files
            device: Device to run inference on
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        
        # Build component vocabulary
        if data_analysis_dir is None:
            # Try to infer from model path
            model_dir = os.path.dirname(model_path)
            project_root = os.path.dirname(os.path.dirname(model_dir))
            data_analysis_dir = os.path.join(project_root, "DataAnalysis")
        
        self.vocab = build_component_types(data_analysis_dir)
        self.name_to_idx = build_name_to_index(self.vocab)
        
        # Initialize model
        self.model = SocketGNN(
            component_vocab_size=len(self.vocab),
            hidden_dim=self.config.get('hidden_dim', 128),
            num_heads=self.config.get('num_heads', 4),
            num_layers=self.config.get('num_layers', 3),
            dropout=self.config.get('dropout', 0.2),
        )
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Component vocabulary size: {len(self.vocab)}")
    
    def predict_socket_placement(
        self,
        unit_dir: str,
        target_component: str,
        k_surface: int = 8,
        k_cross: int = 6,
        return_all_predictions: bool = False,
    ) -> Dict:
        """
        Predict socket placement for a unit
        
        Args:
            unit_dir: Path to unit directory containing mesh.obj and data.json
            target_component: Name of the target component to predict
            k_surface: Number of nearest surface neighbors
            k_cross: Number of nearest component-surface neighbors
            return_all_predictions: Whether to return all predictions or just the best one
            
        Returns:
            Dictionary with prediction results
        """
        # Build graph for the unit
        graph_data = build_room_graph(
            unit_dir=unit_dir,
            target_component_name=target_component,
            data_analysis_dir=None,  # Use default
            k_surface=k_surface,
            k_cross=k_cross,
        )
        
        if graph_data is None:
            return {
                'success': False,
                'error': 'Failed to build graph from unit data',
                'unit_dir': unit_dir,
            }
        
        # Move graph to device
        graph_data = graph_data.to(self.device)
        
        # Check if target component exists
        if not hasattr(graph_data, 'target_component_index'):
            return {
                'success': False,
                'error': f'Target component "{target_component}" not found in unit',
                'unit_dir': unit_dir,
            }
        
        target_component_idx = graph_data.target_component_index
        
        # Make prediction
        with torch.no_grad():
            surface_logits, position_predictions = self.model(graph_data)
            
            # Get all predictions for this component
            edge_index = graph_data[('component', 'near', 'surface')].edge_index
            component_edges = (edge_index[0] == target_component_idx).nonzero(as_tuple=False).squeeze()
            
            if component_edges.numel() == 0:
                return {
                    'success': False,
                    'error': 'No edges found for target component',
                    'unit_dir': unit_dir,
                }
            
            # Get logits and positions for this component's edges
            component_logits = surface_logits[component_edges]
            component_positions = position_predictions[component_edges]
            component_surface_indices = edge_index[1][component_edges]
            
            # Convert to numpy
            logits_np = component_logits.cpu().numpy()
            positions_np = component_positions.cpu().numpy()
            surface_indices_np = component_surface_indices.cpu().numpy()
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(component_logits, dim=0).cpu().numpy()
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            
            # Get surface features for position conversion
            surface_features = graph_data['surface'].x.cpu().numpy()
            surface_centroids = surface_features[:, :3]  # First 3 dimensions are centroid
            surface_normals = surface_features[:, 3:6]   # Next 3 dimensions are normal
            
            # Prepare results
            predictions = []
            
            for i, idx in enumerate(sorted_indices):
                surface_idx = surface_indices_np[idx]
                position_uv = positions_np[idx]
                probability = probabilities[idx]
                
                # Convert UV coordinates to world coordinates
                world_position = self._uv_to_world(
                    position_uv[:2],
                    surface_centroids[surface_idx],
                    surface_normals[surface_idx],
                )
                
                prediction = {
                    'surface_index': int(surface_idx),
                    'surface_centroid': surface_centroids[surface_idx].tolist(),
                    'surface_normal': surface_normals[surface_idx].tolist(),
                    'position_uv': position_uv[:2].tolist(),
                    'position_world': world_position.tolist(),
                    'yaw': float(position_uv[2]),
                    'probability': float(probability),
                    'logit': float(logits_np[idx]),
                }
                
                predictions.append(prediction)
                
                # Only return the best prediction if requested
                if not return_all_predictions and i == 0:
                    break
            
            # Get ground truth if available
            ground_truth = None
            if hasattr(graph_data, 'target_surface_index') and hasattr(graph_data, 'y_pose'):
                gt_surface_idx = graph_data.target_surface_index
                gt_position_uv = graph_data.y_pose.cpu().numpy()
                
                gt_world_position = self._uv_to_world(
                    gt_position_uv[:2],
                    surface_centroids[gt_surface_idx],
                    surface_normals[gt_surface_idx],
                )
                
                ground_truth = {
                    'surface_index': int(gt_surface_idx),
                    'surface_centroid': surface_centroids[gt_surface_idx].tolist(),
                    'surface_normal': surface_normals[gt_surface_idx].tolist(),
                    'position_uv': gt_position_uv[:2].tolist(),
                    'position_world': gt_world_position.tolist(),
                    'yaw': float(gt_position_uv[2]),
                }
            
            return {
                'success': True,
                'unit_dir': unit_dir,
                'target_component': target_component,
                'target_component_index': int(target_component_idx),
                'predictions': predictions,
                'ground_truth': ground_truth,
                'graph_info': {
                    'num_surfaces': int(graph_data['surface'].x.shape[0]),
                    'num_components': int(graph_data['component'].x.shape[0]),
                    'num_edges': int(edge_index.shape[1]),
                },
            }
    
    def _uv_to_world(
        self,
        uv: np.ndarray,
        surface_centroid: np.ndarray,
        surface_normal: np.ndarray,
    ) -> np.ndarray:
        """
        Convert UV coordinates in surface tangent frame to world coordinates
        
        Args:
            uv: UV coordinates
            surface_centroid: Surface centroid in world coordinates
            surface_normal: Surface normal in world coordinates
            
        Returns:
            World coordinates
        """
        # Build local coordinate frame
        n = surface_normal / (np.linalg.norm(surface_normal) + 1e-6)
        
        # Choose an arbitrary up vector
        u = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(u, n)) > 0.9:
            u = np.array([0.0, 1.0, 0.0])
        
        # Gram-Schmidt to get orthonormal basis
        u = u - np.dot(u, n) * n
        u = u / (np.linalg.norm(u) + 1e-6)
        v = np.cross(n, u)
        
        # Convert UV to world coordinates
        world_pos = surface_centroid + uv[0] * u + uv[1] * v
        
        return world_pos
    
    def predict_batch(
        self,
        unit_dirs: List[str],
        target_component: str,
        k_surface: int = 8,
        k_cross: int = 6,
        return_all_predictions: bool = False,
    ) -> List[Dict]:
        """
        Predict socket placement for multiple units
        
        Args:
            unit_dirs: List of unit directory paths
            target_component: Name of the target component to predict
            k_surface: Number of nearest surface neighbors
            k_cross: Number of nearest component-surface neighbors
            return_all_predictions: Whether to return all predictions or just the best one
            
        Returns:
            List of prediction results
        """
        results = []
        
        for unit_dir in unit_dirs:
            result = self.predict_socket_placement(
                unit_dir=unit_dir,
                target_component=target_component,
                k_surface=k_surface,
                k_cross=k_cross,
                return_all_predictions=return_all_predictions,
            )
            results.append(result)
        
        return results
    
    def evaluate_predictions(
        self,
        predictions: List[Dict],
        position_threshold: float = 0.1,
    ) -> Dict[str, float]:
        """
        Evaluate prediction results
        
        Args:
            predictions: List of prediction results
            position_threshold: Threshold for position accuracy
            
        Returns:
            Dictionary with evaluation metrics
        """
        successful_preds = [p for p in predictions if p['success'] and p['ground_truth'] is not None]
        
        if not successful_preds:
            return {
                'total_units': len(predictions),
                'successful_predictions': 0,
                'surface_accuracy': 0.0,
                'position_accuracy': 0.0,
                'position_mae': 0.0,
            }
        
        surface_correct = 0
        position_errors = []
        
        for pred in successful_preds:
            # Check surface prediction
            pred_surface = pred['predictions'][0]['surface_index']
            gt_surface = pred['ground_truth']['surface_index']
            
            if pred_surface == gt_surface:
                surface_correct += 1
                
                # Compute position error
                pred_pos = np.array(pred['predictions'][0]['position_world'])
                gt_pos = np.array(pred['ground_truth']['position_world'])
                
                position_error = np.linalg.norm(pred_pos - gt_pos)
                position_errors.append(position_error)
        
        # Compute metrics
        surface_accuracy = surface_correct / len(successful_preds)
        
        if position_errors:
            position_accuracy = sum(1 for e in position_errors if e < position_threshold) / len(position_errors)
            position_mae = np.mean(position_errors)
            position_rmse = np.sqrt(np.mean(np.array(position_errors) ** 2))
        else:
            position_accuracy = 0.0
            position_mae = 0.0
            position_rmse = 0.0
        
        return {
            'total_units': len(predictions),
            'successful_predictions': len(successful_preds),
            'surface_accuracy': surface_accuracy,
            'position_accuracy': position_accuracy,
            'position_mae': position_mae,
            'position_rmse': position_rmse,
        }


def load_predictor(model_path: str, data_analysis_dir: Optional[str] = None) -> SocketPredictor:
    """
    Convenience function to load a predictor
    
    Args:
        model_path: Path to trained model checkpoint
        data_analysis_dir: Directory containing component CSV files
        
    Returns:
        SocketPredictor instance
    """
    return SocketPredictor(
        model_path=model_path,
        data_analysis_dir=data_analysis_dir,
    )