# Socket Placement Prediction with Graph Neural Networks

This project implements a graph neural network (GNN) for predicting optimal socket placement in 3D building units. The system uses heterogeneous graph representations of building spaces to predict both which surface a socket should be placed on and the precise position on that surface.

## Overview

The pipeline consists of several key components:

1. **Graph Construction**: Converts 3D building units into heterogeneous graphs with surface and component nodes
2. **GNN Model**: Heterogeneous graph neural network for socket placement prediction
3. **Training Pipeline**: Complete training script with proper loss functions and evaluation metrics
4. **Inference System**: Predict socket placement for new building units
5. **Visualization**: 3D visualization of predictions and ground truth
6. **Evaluation**: Comprehensive metrics for model performance

## Architecture

### Graph Representation

The system represents each building unit as a heterogeneous graph with:

- **Surface Nodes**: Represent walls, floors, and ceilings with features:
  - 3D centroid position
  - Surface normal vector
  - Bounding box dimensions
  - Surface type (wall/floor/ceiling/other) one-hot encoding

- **Component Nodes**: Represent existing components (sockets, switches, etc.) with features:
  - 3D centroid position
  - Bounding box dimensions
  - Component type one-hot encoding

- **Edges**:
  - Surface-to-surface: k-nearest neighbors between surface centroids
  - Component-to-surface: k-nearest neighbors with distance as edge attribute

### Model Architecture

The SocketGNN model uses:
- Heterogeneous graph convolution layers with GAT (Graph Attention Network)
- Separate output heads for:
  - Surface selection (binary classification)
  - Position regression (u, v coordinates in surface tangent frame)

## Installation

1. Install dependencies using conda:
```bash
conda env create -f environment.yml
conda activate roomRender
```

2. Install additional PyTorch Geometric packages:
```bash
pip install torch_geometric
```

## Data Structure

The expected data structure is:

```
Data/
├── gebouwE/
│   ├── unit_0001/
│   │   ├── mesh.obj
│   │   └── data.json
│   └── ...
├── rotterdam/
│   └── ...
└── ...

DataAnalysis/
├── gebouwE_components.csv
├── rotterdam_components.csv
└── ...

DataTraining/
├── wcd_enkelvoudig_index.json
├── wcd_enkelvoudig_split.json
└── ...
```

### Data Format

- **mesh.obj**: 3D mesh of the building unit
- **data.json**: Component information with bounding boxes
- **components.csv**: Component type vocabulary

## Usage

### Quick Start with Demo Script

The easiest way to get started is to run the demo script, which tests the pipeline and optionally trains a model:

```bash
# Test pipeline only (fast)
python GNN_Model/demo.py

# Test pipeline and train a new model
python GNN_Model/demo.py --train

# Custom demo with specific data
python GNN_Model/demo.py --data_root Data --split_file DataTraining/wcd_enkelvoudig_split.json --unit_dir Data/gebouwE/unit_0105 --train
```

### Manual Step-by-Step Usage

#### 1. Test the Pipeline

First, test that all components work correctly:

```bash
python GNN_Model/simple_test.py
```

#### 2. Training a Model

```bash
python GNN_Model/example_usage.py \
    --mode train \
    --data_root Data \
    --split_file DataTraining/wcd_enkelvoudig_split.json \
    --config GNN_Model/config.json \
    --save_dir checkpoints
```

#### 3. Making Predictions

```bash
python GNN_Model/example_usage.py \
    --mode predict \
    --model_path results/best.pth \
    --unit_dir Data/gebouwE/unit_0105 \
    --save_dir results
```

#### 4. Evaluating Model Performance

```bash
python GNN_Model/example_usage.py \
    --mode evaluate \
    --model_path results/best.pth \
    --data_root Data \
    --split_file DataTraining/wcd_enkelvoudig_split.json
```

#### 5. Batch Prediction

```bash
python GNN_Model/example_usage.py \
    --mode batch_predict \
    --model_path results/best.pth \
    --data_root Data \
    --split_file DataTraining/wcd_enkelvoudig_split.json \
    --save_dir batch_results
```

## Testing the Pipeline

### Simple Test (Recommended)

To quickly test all components:

```bash
python GNN_Model/simple_test.py
```

### Comprehensive Test

To test the complete pipeline with sample data:

```bash
python GNN_Model/test_pipeline.py \
    --data_root Data \
    --split_file DataTraining/wcd_enkelvoudig_split.json \
    --test_unit Data/gebouwE/unit_0105 \
    --test_training \
    --test_prediction \
    --test_visualization \
    --test_evaluation \
    --epochs 2
```

## Configuration

The model and training parameters can be configured in `GNN_Model/config.json`:

```json
{
  "model": {
    "hidden_dim": 128,
    "num_heads": 4,
    "num_layers": 3,
    "dropout": 0.2
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.001,
    "surface_weight": 1.0,
    "position_weight": 1.0
  }
}
```

## Key Components

### 1. Graph Construction (`room_graph.py`)

- Converts 3D meshes and component data to heterogeneous graphs
- Handles surface classification (wall/floor/ceiling)
- Creates k-nearest neighbor edges

### 2. GNN Model (`socket_gnn.py`)

- `SocketGNN`: Main model class with heterogeneous GNN layers
- `SocketGNNLoss`: Combined loss for surface selection and position regression

### 3. Data Loading (`data_loader.py`)

- `SocketDataset`: Dataset class for loading and preprocessing
- `SocketDataModule`: Data module with train/val splits

### 4. Training (`train_socket_gnn.py`)

- `SocketGNNTrainer`: Complete training pipeline
- Supports learning rate scheduling, early stopping, gradient clipping

### 5. Prediction (`predict.py`)

- `SocketPredictor`: Inference system for making predictions
- Converts UV coordinates to world coordinates

### 6. Visualization (`visualize.py`)

- `SocketVisualizer`: 3D visualization of predictions
- Supports comparison plots and error analysis

### 7. Evaluation (`evaluation.py`)

- `SocketPlacementMetrics`: Comprehensive evaluation metrics
- Surface selection accuracy, position error, etc.

## Performance Metrics

The system evaluates:

1. **Surface Selection**:
   - Accuracy, Precision, Recall, F1-Score
   - AUC-ROC

2. **Position Regression**:
   - Position accuracy (within threshold)
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)

## Example Results

After training, the system can predict socket placement with:
- High surface selection accuracy (>90%)
- Precise position prediction (<10cm error)
- Real-time inference for new units

## File Structure

```
GNN_Model/
├── README.md                 # This file
├── config.json              # Configuration parameters
├── component_types.py       # Component type utilities
├── room_graph.py           # Graph construction
├── socket_gnn.py           # GNN model architecture
├── data_loader.py          # Data loading pipeline
├── train_socket_gnn.py     # Training script
├── predict.py              # Inference system
├── visualize.py            # Visualization tools
├── evaluation.py           # Evaluation metrics
├── test_pipeline.py        # Comprehensive testing script
├── simple_test.py          # Simple component testing
├── example_usage.py        # Example usage script
└── demo.py                 # Complete demo script
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the roomRender experiment for socket placement prediction using graph neural networks.