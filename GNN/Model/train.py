import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from model import ComponentPlacementGNN
import json
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np # Import numpy for nan checks

# --- Configuration Loading ---
def load_config():
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, 'config.json')
    with open(abs_config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config()
model_config = config['model_params']
training_config = config['training_params']

# --- New, Simplified Dataset for Preprocessed Graphs ---
class PreprocessedComponentDataset(Dataset):
    def __init__(self, root_dir, split_type='train_val', split_file_path=None, kfold_file_path=None, fold_key=None, transform=None, pre_transform=None, split_name=None):
        
        super().__init__(root_dir, transform, pre_transform)
        self.split_type = split_type # 'train_val' or 'kfold'
        self.split_file_path = split_file_path
        self.kfold_file_path = kfold_file_path
        self.fold_key = fold_key # e.g., 'fold_1'
        self.split_name = split_name # <-- Moved here
        self.graph_files = self._get_graph_files()

    def _get_graph_files(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Graphs directory not found at: {self.root}")

        all_graphs_in_dir = glob.glob(os.path.join(self.root, '*.pt'))
        if not all_graphs_in_dir:
            raise FileNotFoundError(f"No preprocessed graph files (.pt) found in {self.root}. Please run preprocess.py first.")

        units_for_split = set()
        if self.split_type == 'train_val':
            if not self.split_file_path:
                raise ValueError("split_file_path must be provided for 'train_val' split_type.")
            with open(self.split_file_path, 'r') as f:
                split_data = json.load(f)
            
            # Assuming split_name is 'train' or 'validation'
            split_name = self.fold_key # Using fold_key to pass 'train' or 'validation'
            if split_name not in split_data:
                 raise ValueError(f"Split name '{split_name}' not found in {self.split_file_path}")

            for building, units in split_data[split_name].items():
                for unit in units:
                    units_for_split.add(f"{building}_{unit}")
            
        elif self.split_type == 'kfold':
            if not self.kfold_file_path or not self.fold_key:
                raise ValueError("kfold_file_path and fold_key must be provided for 'kfold' split_type.")
            
            with open(self.kfold_file_path, 'r') as f:
                kfold_data = json.load(f)
            
            # This part needs to determine if we're building a TRAIN or VALIDATION set for the fold.
            # I'll use `split_name` (passed as fold_key) to distinguish.
            
            # First, collect all unique units from all folds in the kfold_data
            all_units_in_kfold_index = set()
            for fold_name, fold_content in kfold_data['folds'].items():
                for building, units in fold_content.items():
                    for unit in units:
                        all_units_in_kfold_index.add(f"{building}_{unit}")

            current_fold_units = set()
            if self.fold_key in kfold_data['folds']:
                for building, units in kfold_data['folds'][self.fold_key].items():
                    for unit in units:
                        current_fold_units.add(f"{building}_{unit}")
            else:
                raise ValueError(f"Fold '{self.fold_key}' not found in {self.kfold_file_path}")
            
            # Determine if this dataset instance is for training or validation of the current fold
            if self.split_name == 'train_fold': 
                # For training, use all units *not* in the current fold (K-1 folds)
                units_for_split = all_units_in_kfold_index - current_fold_units
            elif self.split_name == 'val_fold':
                # For validation, use the units *in* the current fold (1 fold)
                units_for_split = current_fold_units
            else:
                raise ValueError(f"Invalid split name for kfold: {self.split_name}. Must be 'train_fold' or 'val_fold'.")
            
            print(f"Loading K-fold {self.split_name} for '{self.fold_key}' from: {self.kfold_file_path}")
        
        else:
            raise ValueError("Invalid split_type. Must be 'train_val' or 'kfold'.")

        # Filter all_graphs_in_dir based on the unit set
        filtered_files = [
            f for f in all_graphs_in_dir
            if any(os.path.basename(f).startswith(unit_prefix) for unit_prefix in units_for_split)
        ]
        
        if not filtered_files:
            raise FileNotFoundError(f"No graph files found for units in the '{split_name}' split and '{self.fold_key}' fold from '{self.kfold_file_path}' in '{self.root}'.")
        
        return filtered_files

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        data = torch.load(self.graph_files[idx], weights_only=False)
        return data

# --- Helper Function to Find Target Component ---
def get_target_info(data):
    """
    Finds the target component by looking for the one with a non-NaN pose label.
    Returns the index of the target component and a mask for its associated edges.
    """
    y_pose = data['component', 'candidate_placement', 'surface'].y_pose
    
    # Find the indices of edges with valid (non-NaN) pose labels
    valid_pose_edge_indices = torch.where(~torch.isnan(y_pose).any(dim=1))[0]

    if valid_pose_edge_indices.numel() == 0:
        return None, None # No target component found in this graph

    # Get the source node (component) index from the first valid edge
    edge_index_src = data['component', 'candidate_placement', 'surface'].edge_index[0]
    target_comp_idx = edge_index_src[valid_pose_edge_indices[0]].item()

    # Create a mask for all edges originating from this target component
    target_edge_mask = (edge_index_src == target_comp_idx)
    
    return target_comp_idx, target_edge_mask

# --- Training & Validation Loops ---
def train_epoch(loader, model, optimizer, class_loss_fn, reg_loss_fn, alpha):
    model.train()
    total_loss = 0
    graphs_processed = 0
    
    for data in tqdm(loader, desc="Training"):
        target_comp_idx, edge_mask = get_target_info(data)
        
        if target_comp_idx is None:
            continue

        graphs_processed += 1
        optimizer.zero_grad()
        
        classification_logits, regression_output = model(data)
        
        filtered_logits = classification_logits[edge_mask]
        filtered_y_surface = data['component', 'candidate_placement', 'surface'].y_surface[edge_mask]
        
        class_loss = class_loss_fn(filtered_logits, filtered_y_surface)

        true_edge_mask = (filtered_y_surface == 1)
        reg_loss = torch.tensor(0.0, device=class_loss.device)
        
        if true_edge_mask.any():
            pred_y_pose = regression_output[edge_mask][true_edge_mask]
            target_y_pose = data['component', 'candidate_placement', 'surface'].y_pose[edge_mask][true_edge_mask]
            
            if pred_y_pose.numel() > 0 and target_y_pose.numel() > 0:
                 reg_loss = reg_loss_fn(pred_y_pose, target_y_pose)
        
        loss = class_loss + alpha * reg_loss
        if torch.isnan(loss):
             print("Warning: NaN loss detected. Skipping backpropagation for this batch.")
             continue
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / graphs_processed if graphs_processed > 0 else 0

def validate(loader, model, class_loss_fn, reg_loss_fn, alpha):
    model.eval()
    total_loss, total_correct_surf, total_mae, num_target_comps = 0, 0, 0, 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Validating"):
            target_comp_idx, edge_mask = get_target_info(data)
            
            if target_comp_idx is None:
                continue

            num_target_comps += 1
            classification_logits, regression_output = model(data)
            
            comp_specific_logits = classification_logits[edge_mask]
            comp_specific_y_surface = data['component', 'candidate_placement', 'surface'].y_surface[edge_mask]
            
            class_loss = class_loss_fn(comp_specific_logits, comp_specific_y_surface)

            pred_local_idx = torch.argmax(comp_specific_logits)
            true_local_idx = torch.argmax(comp_specific_y_surface)

            if pred_local_idx == true_local_idx:
                total_correct_surf += 1

            true_edge_mask = (comp_specific_y_surface == 1)
            reg_loss = torch.tensor(0.0, device=class_loss.device)
            if true_edge_mask.any():
                pred_y_pose = regression_output[edge_mask][true_edge_mask]
                target_y_pose = data['component', 'candidate_placement', 'surface'].y_pose[edge_mask][true_edge_mask]
                
                if pred_y_pose.numel() > 0 and target_y_pose.numel() > 0:
                    reg_loss = reg_loss_fn(pred_y_pose, target_y_pose)
                    mae = torch.abs(pred_y_pose - target_y_pose).mean()
                    total_mae += mae.item()

            loss = class_loss + alpha * reg_loss
            total_loss += loss.item()
            
    avg_loss = total_loss / num_target_comps if num_target_comps > 0 else 0
    accuracy = total_correct_surf / num_target_comps if num_target_comps > 0 else 0
    avg_mae = total_mae / total_correct_surf if total_correct_surf > 0 else 0

    return avg_loss, accuracy, avg_mae

# --- K-fold Training Function ---
def run_kfold_training(args):
    print(f"\n--- Starting K-fold Cross-Validation ---")
    
    with open(args.kfold_json, 'r') as f:
        kfold_data = json.load(f)
    
    folds = kfold_data.get('folds', {})
    if not folds:
        raise ValueError(f"No 'folds' found in the K-fold JSON file: {args.kfold_json}")

    num_folds = len(folds)
    print(f"Detected {num_folds} folds.")

    all_fold_val_losses = []
    all_fold_val_accs = []
    all_fold_val_maes = []

    for i, fold_key in enumerate(folds.keys()):
        print(f"\n--- Running Fold {i+1}/{num_folds}: {fold_key} ---")

        # Create datasets for the current fold
        train_dataset = PreprocessedComponentDataset(
            root_dir=args.graphs_dir, 
            split_type='kfold', 
            kfold_file_path=args.kfold_json, 
            fold_key=fold_key, 
            split_name='train_fold'
        )
        val_dataset = PreprocessedComponentDataset(
            root_dir=args.graphs_dir, 
            split_type='kfold', 
            kfold_file_path=args.kfold_json, 
            fold_key=fold_key, 
            split_name='val_fold'
        )

        train_loader = DataLoader(train_dataset, batch_size=training_config.get('batch_size', 1), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=training_config.get('batch_size', 1))

        print(f"Fold {fold_key}: Training dataset contains {len(train_dataset)} graphs.")
        print(f"Fold {fold_key}: Validation dataset contains {len(val_dataset)} graphs.")

        # Initialize a new model and optimizer for each fold
        model = ComponentPlacementGNN(
            component_in_channels=training_config['component_in_channels'],
            surface_in_channels=training_config['surface_in_channels'],
            config_path=os.path.join(os.path.dirname(__file__), 'config.json')
        )
        classification_loss_fn = nn.BCEWithLogitsLoss()
        regression_loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        
        best_fold_val_loss = float('inf')

        for epoch in range(1, training_config['epochs'] + 1):
            train_loss = train_epoch(train_loader, model, optimizer, classification_loss_fn, regression_loss_fn, training_config['alpha'])
            val_loss, val_acc, val_mae = validate(val_loader, model, classification_loss_fn, regression_loss_fn, training_config['alpha'])
            
            tqdm.write(f"Fold {fold_key} - Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MAE: {val_mae:.4f}")

            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                # Optionally save best model for this fold
                # torch.save(model.state_dict(), os.path.join(args.graphs_dir, f'best_model_{fold_key}.pth'))
        
        # Store results for this fold
        all_fold_val_losses.append(val_loss)
        all_fold_val_accs.append(val_acc)
        all_fold_val_maes.append(val_mae)

    print(f"\n--- K-fold Cross-Validation Complete ---")
    print(f"Average Validation Loss: {np.mean(all_fold_val_losses):.4f} +/- {np.std(all_fold_val_losses):.4f}")
    print(f"Average Validation Accuracy: {np.mean(all_fold_val_accs):.4f} +/- {np.std(all_fold_val_accs):.4f}")
    print(f"Average Validation MAE: {np.mean(all_fold_val_maes):.4f} +/- {np.std(all_fold_val_maes):.4f}")



# --- Standard Train/Validation Function ---
def run_standard_train_val(args):
    print(f"\n--- Starting Standard Train/Validation ---")
    # Instantiate Datasets and DataLoaders
    train_dataset = PreprocessedComponentDataset(root_dir=args.graphs_dir, split_type='train_val', split_file_path=args.split_json, fold_key='train')
    val_dataset = PreprocessedComponentDataset(root_dir=args.graphs_dir, split_type='train_val', split_file_path=args.split_json, fold_key='validation')

    train_loader = DataLoader(train_dataset, batch_size=training_config.get('batch_size', 1), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.get('batch_size', 1))

    print(f"Training dataset contains {len(train_dataset)} graphs.")
    print(f"Validation dataset contains {len(val_dataset)} graphs.")

    # --- Checkpointing Setup ---
    script_dir = os.path.dirname(__file__)
    CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')

    # Instantiate Model, Loss, and Optimizer
    model = ComponentPlacementGNN(
        component_in_channels=training_config['component_in_channels'],
        surface_in_channels=training_config['surface_in_channels'],
        config_path=os.path.join(os.path.dirname(__file__), 'config.json')
    )
    classification_loss_fn = nn.BCEWithLogitsLoss()
    regression_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])

    if args.resume and os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"Loaded model weights from {BEST_MODEL_PATH} to resume training.")
    else:
        print("Starting training from scratch.")

    # Training loop
    print(f"\nStarting training for {training_config['epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, training_config['epochs'] + 1):
        train_loss = train_epoch(train_loader, model, optimizer, classification_loss_fn, regression_loss_fn, training_config['alpha'])
        val_loss, val_acc, val_mae = validate(val_loader, model, classification_loss_fn, regression_loss_fn, training_config['alpha'])
        
        print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MAE: {val_mae:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model saved with Val Loss: {best_val_loss:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), LAST_MODEL_PATH)
            print(f"  -> Saved checkpoint at epoch {epoch}")

    print("Standard train/validation finished.")


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Component Placement GNN with preprocessed graphs.")
    parser.add_argument('--graphs_dir', type=str, default='Model/graphs', help='Directory containing the preprocessed graph files.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint if available.')
    
    # --- Split Type Arguments ---
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument('--split_json', type=str, help='Path to the train/validation split JSON file.')
    split_group.add_argument('--kfold_json', type=str, help='Path to the k-fold splits JSON file.')
    
    args = parser.parse_args()

    # Determine which training mode to run
    if args.kfold_json:
        run_kfold_training(args)
    elif args.split_json:
        run_standard_train_val(args)