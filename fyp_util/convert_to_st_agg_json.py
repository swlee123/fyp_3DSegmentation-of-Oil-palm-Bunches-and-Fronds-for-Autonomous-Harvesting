# script to convert .txt format to ST required format 

import json
import numpy as np
import os 

# Input .txt file
txt_file = "scene0015_02_labelled.txt"  # Change this to the actual filename

# Generate output filenames dynamically based on the input filename
base_name = os.path.splitext(txt_file)[0]  # Extract filename without extension
aggregation_json_file = f"{base_name}.aggregation.json"
segments_file = f"{base_name}_vh_clean_2.0.010000.segs.json"

# Read the .txt file
with open(txt_file, "r") as f:
    lines = f.readlines()

# Extract metadata (skip header and number of points)
lines = lines[2:]

# Initialize data structures
segGroups = {}
segment_counter = 0

# Iterate through points
for line in lines:
    values = line.strip().split()
    
    # Extract relevant values
    # edit accordingly too ~!! 
    x, y, z = map(float, values[:3])
    r, g, b = map(float, values[3:6])
    scalar_field = float(values[7])
    
    # edit accordingly !!!!
    trunk,fruit,branch,ground= values[8:]

    # Identify the object label
    if trunk != "nan":
        label = "Trunk"
    elif fruit != "nan":
        label = "Fruit"
    elif branch != "nan":
        label = "Branch"
    elif ground != "nan":
        label = "Ground"
    else:
        continue  # Ignore unlabeled points

    # Assign a segment ID based on label type
    if label not in segGroups:
        segGroups[label] = {
            "id": len(segGroups),
            "objectId": len(segGroups),
            "segments": [],
            "label": label
        }

    # Assign a new segment ID
    segGroups[label]["segments"].append(segment_counter)
    segment_counter += 1

# Convert to required JSON format
aggregation_data = {
    "sceneId": "scene_001",
    "appId": "PalmOil_Cloud_v1",
    "segGroups": list(segGroups.values()),
    "segmentsFile": segments_file
}

# Save JSON file
with open(aggregation_json_file, "w") as json_file:
    json.dump(aggregation_data, json_file, indent=4)

print(f"Aggregation JSON saved to {aggregation_json_file}")