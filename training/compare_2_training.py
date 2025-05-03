import pandas as pd
import matplotlib.pyplot as plt


# Script to comapre 2 training instance's Loss Acc and Average Loss

# Load the CSV file
file_path_1 = "train_4/training_logs_4_scratch_epoch_10_dc.csv"  # Update this with the correct path
file_path_2 = "train_5/training_logs_5_pre_trained_epoch_10_dc.csv"


df_1 = pd.read_csv(file_path_1)
df_2 = pd.read_csv(file_path_2)


# Group data by epochs and compute average values
epochs_1 = df_1["Epochs"]
training_loss_1 = df_1["Training Loss"]
training_accuracy_1 = df_1["Training Accuracy"]
training_avg_loss_1 = df_1["Training Avg Loss (so far)"]

epochs_2 = df_2["Epochs"]
training_loss_2 = df_2["Training Loss"]
training_accuracy_2 = df_2["Training Accuracy"]
training_avg_loss_2 = df_2["Training Avg Loss (so far)"]

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_1, training_loss_1, label="From Scratch Loss", color='blue', alpha=0.7)
plt.plot(epochs_2, training_loss_2, label="Pretrained Weight Loss", color='red', alpha=0.7)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss,From Scratch VS Pretrained Weight")
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs_1, training_accuracy_1, label="From Scratch Acc", color='blue', alpha=0.7)
plt.plot(epochs_2, training_accuracy_2, label="Pretrained Weight Acc", color='red', alpha=0.7)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy,From Scratch VS Pretrained Weight")
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Average Loss (so far)
plt.figure(figsize=(10, 5))
plt.plot(epochs_1, training_avg_loss_1, label="From Scratch Avg Loss", color='blue', alpha=0.7)
plt.plot(epochs_2, training_avg_loss_2, label="Pretrained Weight Avg Loss", color='red', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.title("Training Average,From Scratch VS Pretrained Weight")
plt.legend()
plt.grid(True)
plt.show()
