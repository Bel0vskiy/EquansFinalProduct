"""
generate single-target graph datasets
each target component(socket/lamp) gets its own graph,
others are hidden to focus on the target component
"""
import os
import sys
import json
import argparse
from tqdm import tqdm
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Model.graph_builder import build_room_graph

def preprocess_data(normalized_data_dir: str, data_analysis_dir: str, index_json_path: str, output_dir: str, target_component_type_name: str):

    print(f"Starting preprocessing...")
    print(f"Input data directory: {normalized_data_dir}")
    print(f"Data analysis directory: {data_analysis_dir}")
    print(f"Index JSON file: {index_json_path}")
    print(f"Output graph directory: {output_dir}")
    print(f"Target component type: '{target_component_type_name}'")

    os.makedirs(output_dir, exist_ok=True)

    with open(index_json_path, 'r') as f:
        index_data = json.load(f)
    
    locations = index_data.get("locations", {})

    unit_paths = []
    for building, unit_list in locations.items():
        for unit in unit_list:
            path = os.path.join(normalized_data_dir, building, unit)
            if os.path.isdir(path):
                unit_paths.append((building, unit, path))
            else:
                print(f"Warning: Path not found for {building}/{unit}, skipping.")
    
    if not unit_paths:
        raise FileNotFoundError("No valid unit paths were found based on the provided JSON index. "
                              "Please check your --data_dir and JSON file.")

    print(f"Found {len(unit_paths)} units to process from JSON file.")

    graphs_created = 0
    for building, unit, path in tqdm(unit_paths, desc="Processing units"):
        full_graph = build_room_graph(path, data_analysis_dir=data_analysis_dir) 
        if full_graph is None or 'component' not in full_graph.node_types or 'component_names' not in full_graph.meta:
            tqdm.write(f"Skipping {building}/{unit}, could not build graph or metadata is missing.")
            continue

        num_components = full_graph['component'].num_nodes
        component_names = full_graph.meta['component_names']

        for i in range(num_components):
            if component_names[i] == target_component_type_name:
                target_graph = full_graph.clone()

                target_graph['component'].x[i, :3] = 0.0
                edge_index_src = target_graph['component', 'candidate_placement', 'surface'].edge_index[0]
                label_mask = (edge_index_src != i)
                
                target_graph['component', 'candidate_placement', 'surface'].y_surface[label_mask] = 0.0
                target_graph['component', 'candidate_placement', 'surface'].y_pose[label_mask] = torch.full((label_mask.sum(), 2), np.nan, dtype=torch.float32)

                graph_filename = f"{building}_{unit}_comp_{i}.pt"
                graph_filepath = os.path.join(output_dir, graph_filename)
                torch.save(target_graph, graph_filepath)
                graphs_created += 1

    print(f"\nPreprocessing complete.")
    print(f"Successfully created {graphs_created} graph files in '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess room data into graph datasets using a JSON index.")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='Data/DataNormalized',
        help='Path to the root directory containing normalized building data (e.g., DataNormalized).'
    )
    parser.add_argument(
        '--data_analysis_dir',
        type=str,
        default='Data/DataAnalysis',
        help='Path to the directory containing component type CSVs for vocabulary building (used by graph_builder).'
    )
    parser.add_argument(
        '--index_json',
        type=str,
        default='Data/DataTraining/wcd_enkelvoudig_index.json',
        help='Path to the JSON file containing the list of units to process.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Model/graphs',
        help='Path to the directory where processed graphs will be saved.'
    )
    parser.add_argument(
        '--target_component',
        type=str,
        default='wcd enkelvoudig',
        help='The specific component type to generate single-target graphs for.'
    )
    args = parser.parse_args()

    preprocess_data(args.data_dir, args.data_analysis_dir, args.index_json, args.output_dir, args.target_component)