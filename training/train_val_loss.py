import pandas as pd
import matplotlib.pyplot as plt
import os

curr_dir = os.listdir()
train_dir = [i for i in curr_dir if i.startswith("train_")]

total_training_loss = []
total_training_accuracy = []
total_validation_loss = []
total_validation_accuracy = []

for dir in train_dir[0:1]:
    print(dir)
    dir_num = dir.split("_")[1]

    training_logs_path = os.path.join(dir, f"training_logs_{dir_num}.csv")
    validation_logs_path = os.path.join(dir, f"validation_logs_{dir_num}.csv")

    training_logs_df = pd.read_csv(training_logs_path)
    validation_logs_df = pd.read_csv(validation_logs_path)

    # Last row for each epoch
    last_rows_per_epoch = training_logs_df.groupby('Epochs').last()
    training_loss = last_rows_per_epoch["Training Loss"].values
    training_accuracy = last_rows_per_epoch["Training Accuracy"].values

    # Every 5th validation row (starting from 5th)
    validation_every_fifth = validation_logs_df.iloc[4::5, :].reset_index(drop=True)
    validation_loss = validation_every_fifth["Validation Loss"].values
    validation_accuracy = validation_every_fifth["Validation Accuracy"].values

    total_training_loss.append(training_loss)
    total_training_accuracy.append(training_accuracy)
    total_validation_loss.append(validation_loss)
    total_validation_accuracy.append(validation_accuracy)

# Averages
average_training_loss = [sum(x) / len(x) for x in zip(*total_training_loss)]
average_training_accuracy = [sum(x) / len(x) for x in zip(*total_training_accuracy)]
average_validation_loss = [sum(x) / len(x) for x in zip(*total_validation_loss)]
average_validation_accuracy = [sum(x) / len(x) for x in zip(*total_validation_accuracy)]

# Epochs
epochs = list(range(1, len(average_training_loss) + 1))

# --- Plot Loss ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, average_training_loss, label="Training Loss", marker='o', color='blue', alpha=0.7)
plt.plot(epochs, average_validation_loss, label="Validation Loss", marker='o', color='red', alpha=0.7)

# Annotate numbers
for i, (tr, val) in enumerate(zip(average_training_loss, average_validation_loss)):
    plt.text(epochs[i], tr, f"{tr:.2f}", fontsize=8, ha='right', va='bottom', color='blue')
    plt.text(epochs[i], val, f"{val:.2f}", fontsize=8, ha='left', va='top', color='red')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Average Training and Validation Loss across All Folds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Accuracy ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, average_training_accuracy, label="Training Accuracy", marker='o', color='blue', alpha=0.7)
plt.plot(epochs, average_validation_accuracy, label="Validation Accuracy", marker='o', color='red', alpha=0.7)

# Annotate numbers
for i, (tr, val) in enumerate(zip(average_training_accuracy, average_validation_accuracy)):
    plt.text(epochs[i], tr, f"{tr:.2f}", fontsize=8, ha='right', va='bottom', color='blue')
    plt.text(epochs[i], val, f"{val:.2f}", fontsize=8, ha='left', va='top', color='red')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Average Training and Validation Accuracy across All Folds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
