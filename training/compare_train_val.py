import pandas as pd
import matplotlib.pyplot as plt

# Plot Training vs Validation metrics

# Load the CSV file
file_path_1 = "train_6/training_logs_6_mean.csv"  # Update this with the correct path
file_path_2 = "train_6/validation_logs_6_mean.csv"


df_1 = pd.read_csv(file_path_1)
df_2 = pd.read_csv(file_path_2)


# Group data by epochs and compute average values
epochs_1 = df_1["Epochs"]
training_mIou = df_1["mIoU"]
training_mAcc = df_1["mAcc"]


epochs_2 = df_2["Epochs"]
val_mIou = df_2["mIoU"]
val_mAcc = df_2["mAcc"]


# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs_1, training_mIou, label="Training", color='blue', alpha=0.7)
plt.plot(epochs_2, val_mIou, label="Validation", color='red', alpha=0.7)

plt.xlabel("Epochs")
plt.ylabel("mIou")
plt.title("Training vs Validation mIou")
plt.legend()
plt.grid(True)
plt.show()

# Plot Training Accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs_1, training_mAcc, label="Training", color='blue', alpha=0.7)
plt.plot(epochs_2, val_mAcc, label="Validation", color='red', alpha=0.7)

plt.xlabel("Epochs")
plt.ylabel("mAccuracy")
plt.title("Training vs Validation mAcc")
plt.legend()
plt.grid(True)
plt.show()
