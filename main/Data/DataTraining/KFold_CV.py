import os
import json
import random
import sys
from math import ceil



def build_kfold_splits(input_json_path, k=5, seed=None):

    if seed is not None:
        random.seed(seed)

    with open(input_json_path, "r") as f:
        data = json.load(f)

    component = data.get("component", "unknown_component")
    locations = data.get("locations", {})

    folds = {f"fold_{i+1}": {} for i in range(k)}

    for building, units in locations.items():
        if not units:
            continue

        shuffled = units.copy()
        random.shuffle(shuffled)

        fold_size = ceil(len(shuffled) / k)

        for i in range(k):
            start = i * fold_size
            end = start + fold_size
            subset = shuffled[start:end]
            folds[f"fold_{i+1}"].setdefault(building, []).extend(subset)

    output_data = {
        "component": component,
        "folds": folds
    }
    base_dir = os.path.dirname(input_json_path)
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    clean_name = base_name.replace("_index", "")
    output_path = os.path.join(base_dir, f"{clean_name}_kfolds.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    total_units = sum(len(u) for b in locations.values() for u in b)
    print(f"Created {output_path}")
    print(f"{k} folds created for '{component}' ({total_units} total units)")

    return output_data


# change to your own path
if __name__ == "__main__":
    build_kfold_splits("/Users/lucabighignoli/Desktop/uniProject/roomRender/DataTraining/wcd_enkelvoudig_index.json", 8, 20)