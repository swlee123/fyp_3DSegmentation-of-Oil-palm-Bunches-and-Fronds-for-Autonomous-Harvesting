import json
import os

# Input .txt file
name = "scene0015_02_labelled"  # Change this to the actual filename without extension
txt_file = name + ".txt"  # Change this to the actual filename

# Generate output filenames dynamically based on the input filename
base_name = os.path.splitext(txt_file)[0]  # Extract filename without extension
segs_json_file = f"{base_name}_vh_clean_2.0.010000.segs.json"

# Read the .txt file
with open(txt_file, "r") as f:
    lines = f.readlines()

# Extract metadata (skip header and number of points)
lines = lines[2:]

# Initialize segment mapping
segments = []

# Iterate through points and assign segment IDs
segment_counter = 1  # Start from 1 for segmentation indexing
for line in lines:
    values = line.strip().split()
    
    # Extract labels
    # edit accordingly too !! , some label sequence might not be the same as the one in the example
    
    trunk,fruit,branch,ground = values[8:]

    # Identify the object label and assign a segment ID
    if trunk != "nan":
        segment_id = 1  # Example: ID for trunk
    elif fruit != "nan":
        segment_id = 3  # Example: ID for fruit
    elif branch != "nan":
        segment_id = 15  # Example: ID for branch
    elif ground != "nan":
        segment_id = 7  # Example: ID for ground (arbitrary choice)
    else:
        segment_id = -1  # Unlabeled points (optional handling)

    segments.append(segment_id)
    segment_counter += 1

# Define segmentation parameters
segmentation_params = {
    "kThresh": "0.0001",
    "segMinVerts": "20",
    "minPoints": "750",
    "maxPoints": "30000",
    "thinThresh": "0.05",
    "flatThresh": "0.001",
    "minLength": "0.02",
    "maxLength": "1"
}

# Convert to required JSON format
segs_data = {
    "params": segmentation_params,
    "sceneId": name,  # Modify as needed
    "segIndices": segments
}

# Save JSON file
with open(segs_json_file, "w") as json_file:
    json.dump(segs_data, json_file, indent=4)

print(f"Segments JSON saved to {segs_json_file}")
