import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import os

file_dir = r"C:\Users\dota7\Desktop\Year 3\FYP\palm_oil_dataset\labelled\labelled .ply"
files = [f for f in os.listdir(file_dir) if f.endswith(".txt")]

class_labels = ["Branch", "Trunk", "Ground", "Fruit"]
class_counts_total = np.zeros(len(class_labels))

for file in files:
    file_path = os.path.join(file_dir, file)
    
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Parse header from first 
    header_line = lines[0].strip()
    headers = header_line.split()

    # Get index for each class
    class_indices = []
    for label in class_labels:
        try:
            idx = headers.index(label)
            class_indices.append(idx)
        except ValueError:
            print(f"Warning: {label} not found in {file}. Skipping this file.")
            class_indices = []
            break  # Skip this file if any class is missing

    if not class_indices:
        continue

    # Process point data (skip first two lines)
    points = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= len(headers):
            try:
                point = [float(x) if x.lower() != "nan" else np.nan for x in parts]
                points.append(point)
            except ValueError:
                continue  # Skip lines with bad formatting

    points = np.array(points)
    if points.size == 0:
        continue

    # Get class columns dynamically
    class_data = points[:, class_indices]
    class_counts = np.count_nonzero(~np.isnan(class_data), axis=0)
    class_counts_total += class_counts

# Final percentage
total_points = np.sum(class_counts_total)
class_percentages = (class_counts_total / total_points) * 100

print("Class percentages:", dict(zip(class_labels, class_percentages)))

# Plot
plt.figure(figsize=(6, 6))
bars = plt.bar(class_labels, class_percentages, color=["blue", "brown", "green", "red"])
plt.title("Percentage Distribution of Classes in Dataset")
plt.ylabel("Percentage (%)")
plt.xlabel("Classes")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate each bar with the percentage
for bar, percentage in zip(bars, class_percentages):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"{percentage:.2f}%", ha='center', va='bottom')


plt.tight_layout()
plt.show()



