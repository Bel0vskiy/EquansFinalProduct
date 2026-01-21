import torch
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple
import re

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from model import ComponentPlacementGNN
from train_optimised import get_target_info, PreprocessedComponentDataset
from evaluate import (
    get_xyz_from_uv,
    global_denormalize,
    get_model_prediction,
    get_ground_truth
)


class ExperimentDataset(PreprocessedComponentDataset):
    def get(self, idx: int) -> Tuple[any, str]:
        filepath = self.graph_files[idx]
        data = torch.load(filepath, weights_only=False)
        return data, filepath

def load_real_bounds(graph_filepath: str, original_data_root: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    filename = os.path.basename(graph_filepath)
    match = re.match(r"^(?P<building>\w+)_unit_(?P<unit>\d+)_comp_\d+\.pt$", filename)
    if not match:
        print(f"Warning: Could not parse filename '{filename}'")
        return None, None
        
    building = match.group('building')
    unit_id = match.group('unit')
    unit_name = f"unit_{unit_id}"
    json_path = os.path.join(original_data_root, building, unit_name, "data.json")
    
    if not os.path.exists(json_path):
        print(f"Warning: Original data.json not found at '{json_path}'")
        return None, None
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        room_min = np.array(data.get("min", [0,0,0]), dtype=np.float32)
        room_max = np.array(data.get("max", [1,1,1]), dtype=np.float32)
        return room_min, room_max
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing '{json_path}': {e}")
        return None, None


def load_model(model_path: str, config_path: str) -> ComponentPlacementGNN:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = ComponentPlacementGNN(
        config['training_params']['component_in_channels'],
        config['training_params']['surface_in_channels'],
        config_path
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    print(f"Successfully loaded model from {model_path}")
    return model

def run_experiment(model: ComponentPlacementGNN, args: argparse.Namespace):
    print("\n--- Starting Validation Set Experiment ---")
    
    if args.split_json:
        print("--- Mode: Standard Validation Set ---")
        val_dataset = ExperimentDataset(
            root_dir=args.graphs_dir, 
            split_type='train_val', 
            split_file_path=args.split_json, 
            fold_key='validation'
        )
    elif args.kfold_json:
        if not args.fold_key:
            raise ValueError("--fold_key is required when using --kfold_json")
        print(f"--- Mode: K-Fold Validation (using {args.fold_key}) ---")
        val_dataset = ExperimentDataset(
            root_dir=args.graphs_dir,
            split_type='kfold',
            kfold_file_path=args.kfold_json,
            fold_key=args.fold_key,
            split_name='val_fold'
        )
    else:
        raise ValueError("Either --split_json or --kfold_json must be provided.")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Batch size must be 1

    print(f"Found {len(val_dataset)} graphs in the validation set.")

    all_errors_mm = []
    correct_surface_errors_mm = []

    for data, filepath in tqdm(val_loader, desc="Evaluating Validation Set"):
        target_comp_idx, edge_mask = get_target_info(data)
        if target_comp_idx is None:
            continue
            
        prediction = get_model_prediction(model, data, target_comp_idx)
        if prediction is None:
            continue
        
        comp_specific_edge_mask = (data['component', 'candidate_placement', 'surface'].edge_index[0] == target_comp_idx)
        ground_truth = get_ground_truth(data, comp_specific_edge_mask)

        current_filepath = filepath[0]

        room_min, room_max = load_real_bounds(current_filepath, args.original_data_dir)
        if room_min is None or room_max is None:
            tqdm.write(f"Skipping graph {os.path.basename(current_filepath)} due to missing room bounds.")
            continue

        pred_surf_features = data['surface'].x[prediction['global_surf_idx']]
        pred_xyz_norm = get_xyz_from_uv(
            prediction['coords'],
            pred_surf_features[0:3].cpu().numpy(),
            pred_surf_features[3:6].cpu().numpy()
        )
        pred_xyz_mm = global_denormalize(pred_xyz_norm, room_min, room_max)

        true_surf_features = data['surface'].x[ground_truth['global_surf_idx']]
        true_xyz_norm = get_xyz_from_uv(
            ground_truth['coords'],
            true_surf_features[0:3].cpu().numpy(),
            true_surf_features[3:6].cpu().numpy()
        )
        true_xyz_mm = global_denormalize(true_xyz_norm, room_min, room_max)

        error_mm = np.linalg.norm(pred_xyz_mm - true_xyz_mm)
        all_errors_mm.append(error_mm)

        if prediction['global_surf_idx'] == ground_truth['global_surf_idx']:
            correct_surface_errors_mm.append(error_mm)

    print("\n--- Experiment Results ---")
    
    surface_accuracy = np.nan
    avg_correct_surf_error = np.nan
    avg_blind_error = np.nan

    if all_errors_mm:
        avg_blind_error = np.mean(all_errors_mm)
        std_blind_error = np.std(all_errors_mm)
        print(f"Blind Error (all predictions):")
        print(f"  - Average: {avg_blind_error:.2f} mm")
        print(f"  - Std Dev: {std_blind_error:.2f} mm")
        print(f"  - Samples: {len(all_errors_mm)}")
    else:
        print("No valid predictions were made.")

    if correct_surface_errors_mm:
        avg_correct_surf_error = np.mean(correct_surface_errors_mm)
        std_correct_surf_error = np.std(correct_surface_errors_mm)
        surface_accuracy = len(correct_surface_errors_mm) / len(all_errors_mm) if all_errors_mm else 0
        print(f"\nCorrect Surface Error (when surface was correct):")
        print(f"  - Surface Accuracy: {surface_accuracy:.1%}")
        print(f"  - Average Error: {avg_correct_surf_error:.2f} mm")
        print(f"  - Std Dev: {std_correct_surf_error:.2f} mm")
        print(f"  - Samples: {len(correct_surface_errors_mm)}")
    else:
        print("\nNo predictions with the correct surface were made.")
    
    
    if args.graph:
        import matplotlib.pyplot as plt
        
        print("\nGenerating CDF plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))

        def plot_cdf(errors, label, color):
            if not errors:
                return
            sorted_errors = np.sort(errors)
            yvals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
            ax.plot(sorted_errors, yvals, label=label, marker='.', linestyle='none', markersize=4)

        if all_errors_mm:
            plot_cdf(all_errors_mm, 'All Predictions (Blind Error)', 'blue')
        if correct_surface_errors_mm:
            plot_cdf(correct_surface_errors_mm, 'Correct Surface Predictions', 'green')

        ax.set_xlabel("Euclidean Distance Error (mm)")
        ax.set_ylabel("CDF (% of Placements at or below Error)")
        ax.set_title("CDF of Placement Prediction Error")
        ax.grid(True, which='both', linestyle='--')
        
        if all_errors_mm or correct_surface_errors_mm:
            ax.legend()
        
        script_dir = os.path.dirname(__file__)
        plot_path = os.path.join(script_dir, 'placement_error_cdf.png')
        try:
            plt.savefig(plot_path)
            print(f"Successfully saved CDF plot to: {plot_path}")
        except Exception as e:
            print(f"Could not save CDF plot due to an error: {e}")

    return surface_accuracy, avg_correct_surf_error, avg_blind_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a detailed, real-world evaluation on the GNN model.")
    
    split_group = parser.add_mutually_exclusive_group(required=True)
    split_group.add_argument('--split_json', type=str,
                        help='Path to the train/validation split JSON file to identify the validation set.')
    split_group.add_argument('--kfold_json', type=str,
                        help='Path to the k-fold splits JSON file.')
    
    parser.add_argument('--fold_key', type=str,
                        help="The key for the specific fold to use for validation (e.g., 'fold_1'), required if using --kfold_json.")
    parser.add_argument('--model_path', type=str, default='Model/checkpoints/best_model.pth')
    parser.add_argument('--config_path', type=str, default='Model/config.json')
    parser.add_argument('--graphs_dir', type=str, default='Model/graphs',
                        help='Directory containing the preprocessed graph files.')
    parser.add_argument('--original_data_dir', type=str, default='Data/DataOriginal',
                        help='Path to the root directory of the original, unnormalized data (for room bounds).')
    parser.add_argument('--graph', action='store_true', help='If present, a CDF graph of the placement errors will be generated and saved.')


    args = parser.parse_args()

    try:
        model = load_model(args.model_path, args.config_path)
        run_experiment(model, args)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
