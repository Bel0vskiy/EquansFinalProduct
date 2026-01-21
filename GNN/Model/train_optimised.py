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
import numpy as np 
from types import SimpleNamespace


def load_config():
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, 'config.json')
    with open(abs_config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config()
model_config = config['model_params']
training_config = config['training_params']

class PreprocessedComponentDataset(Dataset):
    _file_cache = None 

    def __init__(self, root_dir, split_type='train_val', split_file_path=None, kfold_file_path=None, fold_key=None, transform=None, pre_transform=None, split_name=None):
        super().__init__(root_dir, transform, pre_transform)
        self.split_type = split_type
        self.split_file_path = split_file_path
        self.kfold_file_path = kfold_file_path
        self.fold_key = fold_key
        self.split_name = split_name
        
        if PreprocessedComponentDataset._file_cache is None:
            PreprocessedComponentDataset._file_cache = self._index_directory(root_dir)

        self.graph_files = self._get_files_from_index()
        print(f"Loading {len(self.graph_files)} graphs into RAM...")
        self.data_cache = []
        for path in tqdm(self.graph_files, desc="Caching Data"):
            self.data_cache.append(torch.load(path, weights_only=False))
        print("Caching complete.")

    def _index_directory(self, root_dir):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Graphs directory not found: {root_dir}")
            
        print(f"Indexing graph files in {root_dir}...")
        cache = {}
        with os.scandir(root_dir) as entries:
            for entry in entries:
                if entry.name.endswith('.pt'):
                    if '_comp_' in entry.name:
                        unit_id = entry.name.split('_comp_')[0]
                        if unit_id not in cache:
                            cache[unit_id] = []
                        cache[unit_id].append(entry.path)
        return cache

    def _get_files_from_index(self):
        units_for_split = set()
        if self.split_type == 'train_val':
            with open(self.split_file_path, 'r') as f:
                split_data = json.load(f)
            target_split = self.fold_key 
            if target_split not in split_data: target_split = self.split_name
            for building, units in split_data[target_split].items():
                for unit in units:
                    units_for_split.add(f"{building}_{unit}")
        elif self.split_type == 'kfold':
            with open(self.kfold_file_path, 'r') as f:
                kfold_data = json.load(f)
            all_units = set()
            for content in kfold_data['folds'].values():
                for building, units in content.items():
                    for unit in units:
                        all_units.add(f"{building}_{unit}")
            current_fold_units = set()
            for building, units in kfold_data['folds'][self.fold_key].items():
                for unit in units:
                    current_fold_units.add(f"{building}_{unit}")
            if self.split_name == 'train_fold':
                units_for_split = all_units - current_fold_units
            elif self.split_name == 'val_fold':
                units_for_split = current_fold_units

        final_files = []
        for unit_id in units_for_split:
            if unit_id in self._file_cache:
                final_files.extend(self._file_cache[unit_id])
        return final_files

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        return self.data_cache[idx]

def get_target_info(data):
    y_pose = data['component', 'candidate_placement', 'surface'].y_pose
    valid_pose_edge_indices = torch.where(~torch.isnan(y_pose).any(dim=1))[0]

    if valid_pose_edge_indices.numel() == 0:
        return None, None 

    edge_index_src = data['component', 'candidate_placement', 'surface'].edge_index[0]
    target_comp_idx = edge_index_src[valid_pose_edge_indices[0]].item()
    target_edge_mask = (edge_index_src == target_comp_idx)
    
    return target_comp_idx, target_edge_mask

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

def run_kfold_training(args):
    from experiments import run_experiment, load_model
    print(f"\n--- Starting K-fold Cross-Validation ---")
    
    with open(args.kfold_json, 'r') as f:
        kfold_data = json.load(f)
    
    folds = kfold_data.get('folds', {})
    if not folds:
        raise ValueError(f"No 'folds' found in the K-fold JSON file: {args.kfold_json}")

    num_folds = len(folds)
    print(f"Detected {num_folds} folds.")

    script_dir = os.path.dirname(__file__)
    CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints', 'kfold')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    all_fold_val_losses = []
    all_fold_val_accuracies = []
    all_fold_val_maes = []       
    all_fold_experiment_accuracies = [] 
    all_fold_experiment_correct_surf_errors = [] 
    all_fold_experiment_blind_errors = [] 


    for i, fold_key in enumerate(folds.keys()):
        print(f"\n--- Running Fold {i+1}/{num_folds}: {fold_key} ---")

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

        model = ComponentPlacementGNN(
            component_in_channels=training_config['component_in_channels'],
            surface_in_channels=training_config['surface_in_channels'],
            config_path=os.path.join(os.path.dirname(__file__), 'config.json')
        )
        classification_loss_fn = nn.BCEWithLogitsLoss()
        regression_loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=training_config.get('scheduler_factor', 0.5), 
            patience=training_config.get('scheduler_patience_kfold', 10),
        )
        
        best_fold_val_loss = float('inf')
        best_fold_val_acc = 0.0 
        best_fold_val_mae = float('inf') 
        
        early_stopping_patience = training_config.get('early_stopping_patience', 30)
        epochs_no_improve = 0

        current_fold_best_model_path = os.path.join(CHECKPOINT_DIR, f'best_model_{fold_key}.pth')

        for epoch in range(1, training_config['epochs'] + 1):
            train_loss = train_epoch(train_loader, model, optimizer, classification_loss_fn, regression_loss_fn, training_config['alpha'])
            val_loss, val_acc, val_mae = validate(val_loader, model, classification_loss_fn, regression_loss_fn, training_config['alpha'])
            
            scheduler.step(val_loss)
            
            tqdm.write(f"Fold {fold_key} - Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MAE: {val_mae:.4f}")

            is_best_for_fold = False
            if val_acc > best_fold_val_acc or (val_acc == best_fold_val_acc and val_mae < best_fold_val_mae):
                is_best_for_fold = True
            
            if is_best_for_fold:
                best_fold_val_acc = val_acc
                best_fold_val_mae = val_mae
                best_fold_val_loss = val_loss 
                torch.save(model.state_dict(), current_fold_best_model_path)
                tqdm.write(f"  -> New Best Model for Fold {fold_key}! (Acc: {best_fold_val_acc:.4f}, MAE: {best_fold_val_mae:.4f})")
                epochs_no_improve = 0 
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    tqdm.write(f"\n✋ STOPPING EARLY for Fold {fold_key}! Acc/MAE hasn't improved in {early_stopping_patience} epochs.")
                    break 
        
        all_fold_val_losses.append(best_fold_val_loss)
        all_fold_val_accuracies.append(best_fold_val_acc)
        all_fold_val_maes.append(best_fold_val_mae)
        
        tqdm.write(f"Running experiment for best model of Fold {fold_key}...")
        
        model = ComponentPlacementGNN(
            component_in_channels=training_config['component_in_channels'],
            surface_in_channels=training_config['surface_in_channels'],
            config_path=os.path.join(os.path.dirname(__file__), 'config.json')
        )
        model.load_state_dict(torch.load(current_fold_best_model_path))
        model.eval() 

        experiment_args = SimpleNamespace(
            graphs_dir=args.graphs_dir,
            kfold_json=args.kfold_json,
            fold_key=fold_key,
            original_data_dir=args.original_data_dir,
            model_path=current_fold_best_model_path, 
            config_path=os.path.join(os.path.dirname(__file__), 'config.json'),
            split_json=None, 
            graph=args.graph 
        )

        surf_acc, correct_surf_err, blind_err = run_experiment(model, experiment_args)
        
        all_fold_experiment_accuracies.append(surf_acc)
        all_fold_experiment_correct_surf_errors.append(correct_surf_err)
        all_fold_experiment_blind_errors.append(blind_err)

    print(f"\n--- K-fold Cross-Validation Complete ---")
    
    print(f"Average Validation Loss: {np.mean(all_fold_val_losses):.4f} +/- {np.std(all_fold_val_losses):.4f}")
    print(f"Average Validation Accuracy: {np.mean(all_fold_val_accuracies):.4f} +/- {np.std(all_fold_val_accuracies):.4f}")
    print(f"Average Validation MAE: {np.mean(all_fold_val_maes):.4f} +/- {np.std(all_fold_val_maes):.4f} mm")
    print(f"\n--- K-fold Experiment Results (Aggregated) ---")
    
    valid_exp_acc = [x for x in all_fold_experiment_accuracies if not np.isnan(x)]
    valid_exp_correct_err = [x for x in all_fold_experiment_correct_surf_errors if not np.isnan(x)]
    valid_exp_blind_err = [x for x in all_fold_experiment_blind_errors if not np.isnan(x)]

    if valid_exp_acc:
        print(f"Average Experiment Surface Accuracy: {np.mean(valid_exp_acc):.4f} +/- {np.std(valid_exp_acc):.4f}")
    if valid_exp_correct_err:
        print(f"Average Experiment Correct Surface Error: {np.mean(valid_exp_correct_err):.4f} +/- {np.std(valid_exp_correct_err):.4f} mm")
    if valid_exp_blind_err:
        print(f"Average Experiment Blind Error: {np.mean(valid_exp_blind_err):.4f} +/- {np.std(valid_exp_blind_err):.4f} mm")


def run_standard_train_val(args):
    import matplotlib.pyplot as plt

    print(f"\n--- Starting Standard Train/Validation ---")
    train_dataset = PreprocessedComponentDataset(root_dir=args.graphs_dir, split_type='train_val', split_file_path=args.split_json, fold_key='train')
    val_dataset = PreprocessedComponentDataset(root_dir=args.graphs_dir, split_type='train_val', split_file_path=args.split_json, fold_key='validation')

    train_loader = DataLoader(train_dataset, batch_size=training_config.get('batch_size', 1), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.get('batch_size', 1))

    print(f"Training dataset contains {len(train_dataset)} graphs.")
    print(f"Validation dataset contains {len(val_dataset)} graphs.")

    script_dir = os.path.dirname(__file__)
    CHECKPOINT_DIR = os.path.join(script_dir, 'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')

    model = ComponentPlacementGNN(
        component_in_channels=training_config['component_in_channels'],
        surface_in_channels=training_config['surface_in_channels'],
        config_path=os.path.join(os.path.dirname(__file__), 'config.json')
    )
    classification_loss_fn = nn.BCEWithLogitsLoss()
    regression_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_config['learning_rate'])

    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=training_config.get('scheduler_factor', 0.5), 
        patience=training_config.get('scheduler_patience_standard', 15), 
    )

    if args.resume and os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"Loaded model weights from {BEST_MODEL_PATH} to resume training.")
    else:
        print("Starting training from scratch.")

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_dist': []
    }

    print(f"\nStarting training for {training_config['epochs']} epochs...")
    best_val_loss = float('inf')

    early_stopping_patience = training_config.get('early_stopping_patience', 30)  
    epochs_no_improve = 0

    best_val_acc = 0.0
    best_val_mae = float('inf')
    
    for epoch in range(1, training_config['epochs'] + 1):
        train_loss = train_epoch(train_loader, model, optimizer, classification_loss_fn, regression_loss_fn, training_config['alpha'])
        val_loss, val_acc, val_mae = validate(val_loader, model, classification_loss_fn, regression_loss_fn, training_config['alpha'])

  
        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val MAE: {val_mae:.4f}")

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_dist'].append(val_mae)

        is_best = False
        if val_acc >= best_val_acc and val_mae <= best_val_mae:
            is_best = True

        
        if val_acc < best_val_acc and val_mae > best_val_mae:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n✋ STOPPING EARLY! Accuracy/MAE hasn't improved in {early_stopping_patience} epochs.")
                print(f"   Best Acc: {best_val_acc:.4f} | Best MAE: {best_val_mae:.4f}")
                break
        else:
            epochs_no_improve = 0
        if is_best:
            best_val_acc = val_acc
            best_val_mae = val_mae
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ->  New Best Model! (Acc: {best_val_acc:.4f}, MAE: {best_val_mae:.4f})")

    print("Standard train/validation finished.")

    if history['epoch']:
        print("Generating and saving training history plots...")
        script_dir = os.path.dirname(__file__)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(history['epoch'], history['train_loss'], label='Training Loss', marker='o')
        ax1.plot(history['epoch'], history['val_loss'], label='Validation Loss', marker='o')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        loss_plot_path = os.path.join(script_dir, 'standard_training_loss.png')
        try:
            fig1.savefig(loss_plot_path)
            print(f"Successfully saved loss plot to: {loss_plot_path}")
        except Exception as e:
            print(f"Could not save loss plot due to an error: {e}")
        plt.close(fig1)

        fig2, ax2_acc = plt.subplots(figsize=(12, 6))
        ax2_dist = ax2_acc.twinx()

        p1, = ax2_acc.plot(history['epoch'], history['val_acc'], 'g-', label='Validation Accuracy', marker='o')
        ax2_acc.set_xlabel('Epoch')
        ax2_acc.set_ylabel('Accuracy', color='g')
        ax2_acc.tick_params(axis='y', labelcolor='g')

        p2, = ax2_dist.plot(history['epoch'], history['val_dist'], 'b-', label='Validation MAE', marker='o')
        ax2_dist.set_ylabel('MAE', color='b')
        ax2_dist.tick_params(axis='y', labelcolor='b')

        ax2_acc.set_title('Validation Accuracy and MAE')
        ax2_acc.legend(handles=[p1, p2])
        ax2_acc.grid(True)
        
        acc_mae_plot_path = os.path.join(script_dir, 'standard_training_accuracy_mae.png')
        try:
            fig2.savefig(acc_mae_plot_path)
            print(f"Successfully saved accuracy/MAE plot to: {acc_mae_plot_path}")
        except Exception as e:
            print(f"Could not save accuracy/MAE plot due to an error: {e}")
        plt.close(fig2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Component Placement GNN with preprocessed graphs.")
    parser.add_argument('--graphs_dir', type=str, default='Model/graphs', help='Directory containing the preprocessed graph files.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint if available.')
    
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument('--split_json', type=str, help='Path to the train/validation split JSON file.')
    split_group.add_argument('--kfold_json', type=str, help='Path to the k-fold splits JSON file.')
    parser.add_argument('--original_data_dir', type=str, default='Data/DataOriginal',
                        help='Path to the root directory of the original, unnormalized data (for room bounds).')
    parser.add_argument('--graph', action='store_true',
                        help='If present, a CDF graph of placement errors will be generated for each fold.')
    
    args = parser.parse_args()

    if args.kfold_json:
        run_kfold_training(args)
    elif args.split_json:
        run_standard_train_val(args)