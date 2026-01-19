import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple


class SocketGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for socket placement prediction.
    
    This model predicts:
    1. Which surface a socket should be placed on (classification)
    2. The precise position on that surface (regression)
    
    The graph has two node types:
    - surface: [centroid(3), normal(3), bbox(3), surface_type_one_hot(4)]
    - component: [centroid(3), bbox(3), component_type_one_hot(|V|)]
    
    And two edge types:
    - (surface)-[near]->(surface)
    - (component)-[near]->(surface)
    """
    
    def __init__(
        self,
        component_vocab_size: int,
        surface_dim: int = 13,  # 3 + 3 + 3 + 4
        component_dim: Optional[int] = None,  # 3 + 3 + vocab_size
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super(SocketGNN, self).__init__()
        
        self.component_vocab_size = component_vocab_size
        self.surface_dim = surface_dim
        self.component_dim = component_dim if component_dim is not None else (6 + component_vocab_size)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projections
        self.surface_proj = Linear(surface_dim, hidden_dim)
        self.component_proj = Linear(self.component_dim, hidden_dim)
        
        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('surface', 'near', 'surface'): GATConv((-1, -1), hidden_dim, heads=num_heads, dropout=dropout, concat=False, add_self_loops=False),
                ('component', 'near', 'surface'): GATConv((-1, -1), hidden_dim, heads=num_heads, dropout=dropout, concat=False, add_self_loops=False),
                ('surface', 'near', 'component'): GATConv((-1, -1), hidden_dim, heads=num_heads, dropout=dropout, concat=False, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
        
        # Output heads
        # 1. Surface selection head (binary classification for component->surface edges)
        self.surface_classifier = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # component + surface features
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1)  # binary classification
        )
        
        # 2. Position regression head (u, v coordinates in surface tangent frame)
        self.position_regressor = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, 3)  # (u, v, yaw)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            data: Heterogeneous graph data
            
        Returns:
            Tuple of (surface_logits, position_predictions)
            - surface_logits: Binary classification logits for component->surface edges
            - position_predictions: (u, v, yaw) coordinates for each component->surface edge
        """
        # Initial node embeddings
        x_dict = {
            'surface': self.surface_proj(data['surface'].x),
            'component': self.component_proj(data['component'].x),
        }
        
        # Apply GNN layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
        
        # Get component->surface edges
        edge_index = data[('component', 'near', 'surface')].edge_index
        component_idx = edge_index[0]
        surface_idx = edge_index[1]
        
        # Combine component and surface features for each edge
        component_features = x_dict['component'][component_idx]
        surface_features = x_dict['surface'][surface_idx]
        edge_features = torch.cat([component_features, surface_features], dim=1)
        
        # Predict surface selection and position
        surface_logits = self.surface_classifier(edge_features).squeeze(-1)
        position_predictions = self.position_regressor(edge_features)
        
        return surface_logits, position_predictions
    
    def predict_surface_and_position(
        self, 
        data: HeteroData,
        component_idx: int
    ) -> Tuple[int, torch.Tensor, float]:
        """
        Predict the best surface and position for a given component
        
        Args:
            data: Heterogeneous graph data
            component_idx: Index of the target component
            
        Returns:
            Tuple of (best_surface_idx, position, confidence)
            - best_surface_idx: Index of the predicted surface
            - position: (u, v, yaw) coordinates on the surface
            - confidence: Confidence score for the surface selection
        """
        self.eval()
        with torch.no_grad():
            surface_logits, position_predictions = self.forward(data)
            
            # Find edges connected to the target component
            edge_index = data[('component', 'near', 'surface')].edge_index
            component_edges = (edge_index[0] == component_idx).nonzero(as_tuple=False).squeeze()
            
            if component_edges.numel() == 0:
                return None, None, 0.0
            
            # Get logits and positions for this component's edges
            component_logits = surface_logits[component_edges]
            component_positions = position_predictions[component_edges]
            component_surface_indices = edge_index[1][component_edges]
            
            # Find the best surface
            best_edge_idx = torch.argmax(component_logits)
            best_surface_idx = component_surface_indices[best_edge_idx].item()
            best_position = component_positions[best_edge_idx]
            confidence = torch.sigmoid(component_logits[best_edge_idx]).item()
            
            return best_surface_idx, best_position, confidence


class SocketGNNLoss(nn.Module):
    """
    Combined loss function for socket placement prediction
    """
    
    def __init__(
        self,
        surface_weight: float = 1.0,
        position_weight: float = 1.0,
        pos_weight: Optional[float] = None,  # For binary classification imbalance
    ):
        super(SocketGNNLoss, self).__init__()
        self.surface_weight = surface_weight
        self.position_weight = position_weight
        
        # Binary cross entropy with logits for surface selection
        self.surface_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # MSE loss for position regression
        self.position_loss_fn = nn.MSELoss()
        
    def forward(
        self,
        surface_logits: torch.Tensor,
        position_predictions: torch.Tensor,
        surface_labels: torch.Tensor,
        position_labels: torch.Tensor,
        surface_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            surface_logits: Predicted surface selection logits
            position_predictions: Predicted positions
            surface_labels: Ground truth surface selection (0/1)
            position_labels: Ground truth positions
            surface_mask: Mask for valid edges (optional)
            
        Returns:
            Dictionary with individual and total losses
        """
        # Surface selection loss
        if surface_mask is not None:
            surface_loss = self.surface_loss_fn(
                surface_logits[surface_mask], 
                surface_labels[surface_mask]
            )
        else:
            surface_loss = self.surface_loss_fn(surface_logits, surface_labels)
        
        # Position regression loss (only for positive edges)
        positive_mask = surface_labels > 0.5
        if surface_mask is not None:
            positive_mask = positive_mask & surface_mask
            
        if positive_mask.sum() > 0:
            position_loss = self.position_loss_fn(
                position_predictions[positive_mask],
                position_labels[positive_mask]
            )
        else:
            position_loss = torch.tensor(0.0, device=surface_logits.device)
        
        # Combined loss
        total_loss = (
            self.surface_weight * surface_loss + 
            self.position_weight * position_loss
        )
        
        return {
            'total_loss': total_loss,
            'surface_loss': surface_loss,
            'position_loss': position_loss
        }