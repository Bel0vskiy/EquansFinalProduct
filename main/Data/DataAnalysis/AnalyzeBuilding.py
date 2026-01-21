import os
import json
import csv
import re
from collections import Counter


"""
File to count amount of different ET compoenents in diff buildings
Done to get 
 """


def count_ET_components(base_folder, pathCSV):
    folder_name = os.path.basename(os.path.normpath(base_folder))
    output_csv = os.path.join(pathCSV, f"{folder_name}_components.csv")

    component_counter = Counter()

    name_cleaner = re.compile(r"^(.*?)(?::\d+)?$")

    for unit_folder in sorted(os.listdir(base_folder)):
        unit_path = os.path.join(base_folder, unit_folder)
        json_path = os.path.join(unit_path, "data.json")

        if os.path.isdir(unit_path) and os.path.isfile(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                if "objects" in data and "ET" in data["objects"]:
                    et_objects = data["objects"]["ET"]
                    if isinstance(et_objects, list):
                        for obj in et_objects:
                            if isinstance(obj, dict) and "name" in obj:
                                raw_name = obj["name"].strip()
                                clean_name = name_cleaner.match(raw_name).group(1)
                                component_counter[clean_name] += 1

            except Exception as e:
                print(f"Error reading {json_path}: {e}")

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["component_name", "count"])
        for name, count in sorted(component_counter.items()):
            writer.writerow([name, count])

    print(f"CSV written to: {output_csv}")
    print(f"Total unique components: {len(component_counter)}")
    print(f"Total ET components counted: {sum(component_counter.values())}")


# example usage
count_ET_components(
    "uniProject/roomRender/Data/paviljoen/",
    "roomRender/Data/DataAnalysis/"
)
