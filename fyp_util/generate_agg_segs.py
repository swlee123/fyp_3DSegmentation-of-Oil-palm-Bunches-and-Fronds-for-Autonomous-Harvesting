import json
import numpy as np
import os
import argparse

def parse_txt_file(txt_file):
    # Read the file
    with open(txt_file, "r") as f:
        lines = f.readlines()
    
    # Read the header to determine column indices
    header = lines[0].strip().split()
    label_indices = {label: idx for idx, label in enumerate(header)}
    
    print("Label indices:", label_indices)
    
    # Skip header and metadata lines
    lines = lines[2:]
    
    return lines, label_indices

def generate_segmentation_json(txt_file, lines, label_indices):
    base_name = os.path.splitext(txt_file)[0]
    segs_json_file = f"{base_name}_vh_clean_2.0.010000.segs.json"
    
    segments = []
    for line in lines:
        values = line.strip().split()
        
        # Extract labels dynamically
        trunk = values[label_indices.get("Trunk", -1)] if "Trunk" in label_indices else "nan"
        branch = values[label_indices.get("Branch", -1)] if "Branch" in label_indices else "nan"
        fruit = values[label_indices.get("Fruit", -1)] if "Fruit" in label_indices else "nan"
        ground = values[label_indices.get("Ground", -1)] if "Ground" in label_indices else "nan"

        # Assign segment ID
        if trunk != "nan":
            segment_id = 1
        elif fruit != "nan":
            segment_id = 3
        elif branch != "nan":
            segment_id = 15
        elif ground != "nan":
            segment_id = 7
        else:
            segment_id = -1
        
        segments.append(segment_id)
    
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
    
    segs_data = {
        "params": segmentation_params,
        "sceneId": base_name,
        "segIndices": segments
    }
    
    with open(segs_json_file, "w") as json_file:
        json.dump(segs_data, json_file, indent=4)
    
    print(f"Segments JSON saved to {segs_json_file}")

def generate_aggregation_json(txt_file, lines, label_indices):
    base_name = os.path.splitext(txt_file)[0]
    aggregation_json_file = f"{base_name}.aggregation.json"
    segments_file = f"{base_name}_vh_clean_2.0.010000.segs.json"
    
    segGroups = {}
    segment_counter = 0
    
    for line in lines:
        values = line.strip().split()
        
        # Extract labels dynamically
        trunk = values[label_indices.get("Trunk", -1)] if "Trunk" in label_indices else "nan"
        branch = values[label_indices.get("Branch", -1)] if "Branch" in label_indices else "nan"
        fruit = values[label_indices.get("Fruit", -1)] if "Fruit" in label_indices else "nan"
        ground = values[label_indices.get("Ground", -1)] if "Ground" in label_indices else "nan"

        
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
        
        # Assign segment ID
        if label not in segGroups:
            segGroups[label] = {
                "id": len(segGroups),
                "objectId": len(segGroups),
                "segments": [],
                "label": label
            }
        
        segGroups[label]["segments"].append(segment_counter)
        segment_counter += 1
    
    aggregation_data = {
        "sceneId": base_name,
        "appId": "PalmOil_Cloud_v1",
        "segGroups": list(segGroups.values()),
        "segmentsFile": segments_file
    }
    
    with open(aggregation_json_file, "w") as json_file:
        json.dump(aggregation_data, json_file, indent=4)
    
    print(f"Aggregation JSON saved to {aggregation_json_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert .txt point cloud data into ST-required JSON format.")
    parser.add_argument("txt_file", type=str, help="Path to the input .txt file")
    args = parser.parse_args()
    
    lines, label_indices = parse_txt_file(args.txt_file)
    generate_segmentation_json(args.txt_file, lines, label_indices)
    generate_aggregation_json(args.txt_file, lines, label_indices)

if __name__ == "__main__":
    main()
