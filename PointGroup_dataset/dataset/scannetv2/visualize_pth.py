import torch
import numpy as np
import matplotlib.pyplot as plt


# this visualization shows that , in all converted .pth file , trunk and ground is not saved in .pth ,
# and for fruit , the percentage is too low , so model basically cant learn it
# !!!!!!¬!!!!


# Load the .pth file
file_path = "/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/train/scene0155_24_inst_nostuff.pth"
data = torch.load(file_path, map_location="cpu")  # Load on CPU

# Extract class labels (assuming they are the 3rd element of the tuple)
class_data = data[2]  # The third array contains class labels

print(class_data)

# Convert to NumPy if it's a PyTorch tensor
if isinstance(class_data, torch.Tensor):
    class_data = class_data.numpy()

# Remove ignored labels (-100 or NaN)
class_data = class_data[class_data != -100]

# Get unique classes and their counts
unique_classes, counts = np.unique(class_data, return_counts=True)

# Calculate percentage
total_points = np.sum(counts)
class_percentages = (counts / total_points) * 100 if total_points > 0 else np.zeros_like(counts)

# Print class distribution
print("Class Percentage:", dict(zip(unique_classes, class_percentages)))

# Plot class distribution
plt.figure(figsize=(6, 6))
bars = plt.bar(unique_classes.astype(str), class_percentages, color=["blue", "brown", "green", "red"])

# Add percentage labels on top of each bar
for bar, percent in zip(bars, class_percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{percent:.1f}%", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title("Percentage Distribution of Classes")
plt.ylabel("Percentage (%)")
plt.ylim(0, max(class_percentages) + 5)  # Adjust ylim for better visibility
plt.show()