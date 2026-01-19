import os
import json
import random
import sys

"""
Splits each building's units into training and validation sets.
Produces a single JSON file with "train" and "validation" sections.
"""

def split_dataset(input_json_path, train_ratio=0.8):

    # Load the input JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    component = data.get("component", "unknown_component")
    locations = data.get("locations", {})

    train_split = {}
    val_split = {}

    for building, units in locations.items():
        if not units:
            continue

        # Shuffle for randomness
        random.shuffle(units)

        split_index = int(len(units) * train_ratio)
        train_split[building] = units[:split_index]
        val_split[building] = units[split_index:]

    # Build output JSON
    output_data = {
        "component": component,
        "train": train_split,
        "validation": val_split
    }

    # Save output file
    base_dir = os.path.dirname(input_json_path)
    base_name = os.path.splitext(os.path.basename(input_json_path))[0]
    clean_name = base_name.replace("_index", "")
    output_path = os.path.join(base_dir, f"{clean_name}_split.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    total_train = sum(len(v) for v in train_split.values())
    total_val = sum(len(v) for v in val_split.values())

    print(f"✅ Created {output_path}")
    print(f"📊 Training: {total_train}  |  Validation: {total_val}")

    return output_data

# ---- Run as standalone script ----
if __name__ == "__main__":
    split_dataset("/Users/lucabighignoli/Documents/development/uniDev/roomRender/DataTraining/wcd_enkelvoudig_index.json", 0.9)
