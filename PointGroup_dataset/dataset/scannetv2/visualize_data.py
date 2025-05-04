import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os



file_path = "txt/L15-P15_labelled_ss_4.txt"
# file_format = ".txt"


# files = [f for f in os.listdir(folder_path) if f.endswith(file_format)]

# print(files)
lines = []


# Read file and process data
with open(file_path, "r") as f:
    lines = f.readlines()
    
# Convert lines to numerical data, skipping the first line if it's a point count
points = []
for line in lines[2:]:  # Skipping the first line
    parts = line.strip().split()
    if len(parts) == 12:  # Ensuring correct format
        points.append([float(x) if x != "nan" else np.nan for x in parts])

# Convert to NumPy array
points = np.array(points)

# Extract the class columns (Branch, Trunk, Ground, Fruit)
class_data = points[:, -4:]  # Last 4 columns

# Count occurrences of non-NaN values (i.e., points belonging to each class)
class_counts = np.count_nonzero(~np.isnan(class_data), axis=0)

total_points = np.nansum(class_counts)

# Calculate percentage
class_labels = ["Branch", "Trunk", "Ground", "Fruit"]
class_percentages = (class_counts / total_points) * 100

print("Class percentage",class_percentages)
# Plot class distribution
plt.figure(figsize=(6, 6))
plt.bar(class_labels,class_percentages, color=["blue", "brown", "green", "red"])
plt.title("Percentage Distribution of Classes")
plt.show()


