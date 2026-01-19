import os
import json
import re
from collections import defaultdict



"""
This file is meant to create a json containing the info with all the units that contain a certain components
especially useful to create the training dataset
"""


def build_single_component_index(root_folder,root_trainingFolder ,target_component):
    # Clean the component name (normalize like before)
    name_cleaner = re.compile(r"^(.*?)(?::\d+)?$")
    clean_target = name_cleaner.match(target_component.strip()).group(1)

    result = defaultdict(list)

    for building in sorted(os.listdir(root_folder)):
        building_path = os.path.join(root_folder, building)
        if not os.path.isdir(building_path):
            continue

        for unit_folder in sorted(os.listdir(building_path)):
            unit_path = os.path.join(building_path, unit_folder)
            json_path = os.path.join(unit_path, "data.json")

            if not os.path.isfile(json_path):
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                if "objects" in data and "ET" in data["objects"]:
                    for obj in data["objects"]["ET"]:
                        if isinstance(obj, dict) and "name" in obj:
                            raw_name = obj["name"].strip()
                            clean_name = name_cleaner.match(raw_name).group(1)

                            # Match target
                            if clean_name == clean_target:
                                result[building].append(unit_folder)
                                break  # found it in this unit, skip rest

            except Exception as e:
                print(f"⚠️ Error reading {json_path}: {e}")

    # Prepare JSON structure
    output_data = {
        "component": clean_target,
        "locations": dict(result)
    }

    # Write file named after component
    safe_name = clean_target.replace("/", "_").replace(" ", "_")
    output_path = os.path.join(root_trainingFolder, f"{safe_name}_index.json")

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Index for '{clean_target}' created at: {output_path}")
    total_units = sum(len(u) for u in result.values())
    print(f"Found in {len(result)} buildings, {total_units} total units.")

    return output_data


# Example usage:
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.join(script_dir, "..", "Data")
    root_folder = os.path.abspath(root_folder)
    ## relative paths that should work for more or less everybody ##

    build_single_component_index(
        root_folder,
        script_dir,
        "wcd enkelvoudig"  ## to choose the component u want to get the list of units of
    )
