import os
import csv
from typing import List, Dict

"""
Utilities to build the component type vocabulary from CSV files.
"""


def _read_component_csv(csv_path: str) -> List[str]:
    names: List[str] = []
    if not os.path.exists(csv_path):
        return names
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("component_name") or "").strip()
            if name:
                names.append(name)
    return names


def build_component_types(data_analysis_dir: str) -> List[str]:
    print(f"Building component types from: {data_analysis_dir}")
    files = [
        os.path.join(data_analysis_dir, "gebouwE_components.csv"),
        os.path.join(data_analysis_dir, "paviljoen_components.csv"),
        os.path.join(data_analysis_dir, "opvang_components.csv"),
        os.path.join(data_analysis_dir, "rotterdam_components.csv"),
    ]
    print(f"Component CSV files: {files}")
    all_names: List[str] = []
    for p in files:
        names = _read_component_csv(p)
        print(f"Found {len(names)} components in {os.path.basename(p)}")
        all_names.extend(names)
    # unique, stable order
    seen: Dict[str, bool] = {}
    unique: List[str] = []
    for n in all_names:
        if n not in seen:
            seen[n] = True
            unique.append(n)
    print(f"Total unique components: {len(unique)}")
    return unique


def build_name_to_index(vocab: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(vocab)}


