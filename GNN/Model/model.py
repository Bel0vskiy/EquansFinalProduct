import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear # Changed SAGEConv to GATv2Conv

import json
import os # Import os module

class ComponentPlacementGNN(nn.Module):
    """
    A Heterogeneous Graph Neural Network for predicting component placement.

    This model performs two tasks:
    1.  Classification: Predicts which surface is the correct host for a component
        from a list of all possible surfaces.
    2.  Regression: Predicts the precise (u, v) coordinates of the component
        on its predicted host surface.
    """
    def __init__(self, component_in_channels: int, surface_in_channels: int, config_path: str = 'config.json'):
        """
        Initializes the model layers.

        Args:
            component_in_channels (int): Dimensionality of the input component features.
            surface_in_channels (int): Dimensionality of the input surface features.
            config_path (str): Path to the configuration JSON file.
        """
        super().__init__()

        # Load configuration
        # Construct the absolute path to config.json
        script_dir = os.path.dirname(__file__)
        abs_config_path = os.path.join(script_dir, config_path)

        with open(abs_config_path, 'r') as f:
            config = json.load(f)
        model_config = config['model_params']
        hidden_channels = model_config['hidden_channels']
        num_gnn_layers = model_config.get('num_gnn_layers', 2) # Default to 2 if not in config

        self.hidden_channels = hidden_channels

        # 1. Encoders for each node type to project them into the hidden space
        self.component_encoder = Linear(component_in_channels, hidden_channels)
        self.surface_encoder = Linear(surface_in_channels, hidden_channels)

        num_heads = model_config.get('num_heads', 4)

        # VALIDATION: Ensure hidden_channels is divisible by heads
        assert hidden_channels % num_heads == 0, "Hidden dimension must be divisible by number of heads!"

        # 2. Heterogeneous GNN message passing layers
        self.convs = nn.ModuleList()
        head_dim = hidden_channels // num_heads

        for _ in range(num_gnn_layers):
            conv = HeteroConv({
                ('surface', 'adjacent', 'surface'): GATv2Conv(
                    (-1, -1), head_dim, heads=num_heads, concat=True, add_self_loops=False
                ),
                ('component', 'near', 'component'): GATv2Conv(
                    (-1, -1), head_dim, heads=num_heads, concat=True, add_self_loops=False
                ),
                ('component', 'candidate_placement', 'surface'): GATv2Conv(
                    (-1, -1), head_dim, heads=num_heads, concat=True, add_self_loops=False
                ),
                ('surface', 'rev_candidate_placement', 'component'): GATv2Conv(
                    (-1, -1), head_dim, heads=num_heads, concat=True, add_self_loops=False
                ),
            }, aggr='sum')
            self.convs.append(conv) 
        
        # 3. Two-headed output layer
        # Head 1: Classification Head
        # Predicts a single score for a (component, surface) pair.
        self.classification_head = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 1)
        )

        # Head 2: Regression Head
        # Predicts the (u, v) coordinates for a (component, surface) pair.
        self.regression_head = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 2)  # Outputs u and v
        )

    def forward(self, data: HeteroData):
        """
        Defines the forward pass of the model.

        Args:
            data (HeteroData): The input heterogeneous graph data object.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - classification_logits (torch.Tensor): A tensor of logits for each
              candidate edge, shape [num_candidate_edges].
            - regression_output (torch.Tensor): A tensor of predicted (u,v)
              coordinates for each candidate edge, shape [num_candidate_edges, 2].
        """
        x_dict = data.x_dict

        # 1. Encode initial node features
        x_dict['component'] = self.component_encoder(x_dict['component'])
        x_dict['surface'] = self.surface_encoder(x_dict['surface'])

        # 2. Perform message passing
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # 3. Apply prediction heads to candidate edges

        # Get the source (component) and destination (surface) node indices for all candidate edges
        comp_nodes, surf_nodes = data['component', 'candidate_placement', 'surface'].edge_index

        # Get the final embeddings for these specific nodes
        comp_embs = x_dict['component'][comp_nodes]
        surf_embs = x_dict['surface'][surf_nodes]

        # Concatenate the embeddings to form the input for the prediction heads
        head_input = torch.cat([comp_embs, surf_embs], dim=-1)

        # Get the raw outputs from both heads
        classification_logits = self.classification_head(head_input).squeeze(-1)
        regression_output = self.regression_head(head_input)

        return classification_logits, regression_output