import numpy as np
import open3d as o3d


# label.npy is npy file generated from dataset
# pred.npy is the predicted class using trainedweight 


# Load original .txt file (X, Y, Z, R, G, B)
txt_file = "scene0015_02_labelled.txt"
data = np.loadtxt(txt_file, skiprows=2)  # Skip header row

xyz = data[:, :3]   # X, Y, Z
rgb = data[:, 3:6]  # R, G, B

# Load predicted labels
pred_file = "/home/swlee/Stratified-Transformer/npyfile/scene0015_02_inst_nostuff_10_pred.npy"
pred_labels = np.load(pred_file)
# print("Unique : ",np.unique(pred_labels))
# Ensure sizes match
if len(pred_labels) != len(xyz):
    raise ValueError(f"Mismatch: {len(pred_labels)} predictions vs {len(xyz)} points.")

class_dict = {
    0 : "Trunk",
    1 : "Ground",
    2 : "Branch",
    3 : "Fruit"
}

num_classes = len(class_dict)

# Initialize one-hot encoding matrix with NaN
one_hot_labels = np.full((len(pred_labels), num_classes), np.nan)

# Assign 1.0 to the correct class index
for i, label in enumerate(pred_labels):
    if label in class_dict:  # Ensure valid label
        one_hot_labels[i, label] = 1.0

# Combine all data (XYZ, RGB, One-Hot Labels)
combined_data = np.column_stack((xyz, rgb, one_hot_labels))


# Save to new .txt file
output_file = "segmented_pred_scene0015_02_scratch.txt"
header = "//X Y Z Rf Gf Bf Trunk Ground Branch Fruit\n"+str(len(xyz))

np.savetxt(output_file, combined_data, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.1f %.1f %.1f %.1f", header=header, comments="")

print(f"Segmented predictions saved to {output_file}")
