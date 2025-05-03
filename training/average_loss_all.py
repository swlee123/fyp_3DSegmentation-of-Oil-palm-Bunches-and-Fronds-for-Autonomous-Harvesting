import pandas as pd
import matplotlib.pyplot as plt
import os

curr_dir = os.listdir()
train_dir = []

for i in curr_dir:
    if i.startswith("train_"):
        train_dir.append(i)


total_training_loss = []
total_training_accuracy = []
total_validation_loss = []
total_validation_accuracy = []

for dir in train_dir:
    # Load the CSV file

    dir_num = dir.split("_")[1]
    
    training_logs_path = os.path.join(dir, f"training_logs_{dir_num}.csv")
    validation_logs_path = os.path.join(dir, f"validation_logs_{dir_num}.csv")
    
    training_logs_df = pd.read_csv(training_logs_path)
    validation_logs_df = pd.read_csv(validation_logs_path)
    
    training_loss = training_logs_df["Training Loss"]
    training_accuracy = training_logs_df["Training Accuracy"]
    validation_loss = validation_logs_df["Validation Loss"]
    validation_accuracy = validation_logs_df["Validation Accuracy"]
    
    epochs = [i for i in range(1, len(training_loss) + 1)]

    # Append the data to the total lists
    total_training_loss.append(training_loss)
    total_training_accuracy.append(training_accuracy)
    total_validation_loss.append(validation_loss)
    total_validation_accuracy.append(validation_accuracy)


# Calculate the average values
average_training_loss = [sum(x) / len(x) for x in zip(*total_training_loss)]
average_training_accuracy = [sum(x) / len(x) for x in zip(*total_training_accuracy)]
average_validation_loss = [sum(x) / len(x) for x in zip(*total_validation_loss)]
average_validation_accuracy = [sum(x) / len(x) for x in zip(*total_validation_accuracy)]

# Plot Training Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, average_training_loss, label="Training Loss", color='blue', alpha=0.7)
plt.plot(epochs, average_training_accuracy, label="Training Acc", color='red', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.title("Average Training Loss and Accuracy across All Folds")
plt.legend()
plt.show()

# plot validation
plt.figure(figsize=(10, 5))
plt.plot(epochs[:25], validation_loss, label="Val Loss", color='blue', alpha=0.7)
plt.plot(epochs[:25], validation_accuracy, label="Val Acc", color='red', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.title("Average Validation Loss and Accuracy across All Folds")
plt.legend()
plt.show()


    