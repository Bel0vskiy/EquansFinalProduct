import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

import json
import os

class ComponentPlacementGNN(nn.Module):
    """
    GNN for component placement predicting
    1)classification model guesses surface to put MEP component on
    2)regression model guesses position of component on the surface
    """
    def __init__(self, component_in_channels: int, surface_in_channels: int, config_path: str = 'config.json'):
        super().__init__()

        script_dir = os.path.dirname(__file__)
        abs_config_path = os.path.join(script_dir, config_path)

        with open(abs_config_path, 'r') as f:
            config = json.load(f)
        model_config = config['model_params']
        hidden_channels = model_config['hidden_channels']
        num_gnn_layers = model_config.get('num_gnn_layers', 2)

        self.hidden_channels = hidden_channels

        self.component_encoder = Linear(component_in_channels, hidden_channels)
        self.surface_encoder = Linear(surface_in_channels, hidden_channels)

        num_heads = model_config.get('num_heads', 4)
        assert hidden_channels % num_heads == 0, "Hidden dimension must be divisible by number of heads!"
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
        
        self.classification_head = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 1)
        )

        self.regression_head = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 2)
        )

    def forward(self, data: HeteroData):
        x_dict = data.x_dict

        x_dict['component'] = self.component_encoder(x_dict['component'])
        x_dict['surface'] = self.surface_encoder(x_dict['surface'])

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        comp_nodes, surf_nodes = data['component', 'candidate_placement', 'surface'].edge_index
        
        comp_embs = x_dict['component'][comp_nodes]
        surf_embs = x_dict['surface'][surf_nodes]

        head_input = torch.cat([comp_embs, surf_embs], dim=-1)
        
        classification_logits = self.classification_head(head_input).squeeze(-1)
        regression_output = self.regression_head(head_input)

        return classification_logits, regression_output