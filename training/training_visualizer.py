import pandas as pd
import matplotlib.pyplot as plt


# add visualizer for iou and acc for classes


# Load the CSV file
file_path = "train_6/training_logs_6.csv"  # Update this with the correct path
df = pd.read_csv(file_path)

# Group data by epochs and compute average values
epochs = df["Epochs"]
training_loss = df["Training Loss"]
training_accuracy = df["Training Accuracy"]
training_avg_loss = df["Training Avg Loss (so far)"]

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, label="Training Loss", color='blue', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_accuracy, label="Training Accuracy", color='green', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Average Loss (so far)
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_avg_loss, label="Training Avg Loss (so far)", color='red', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Training Average Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()
