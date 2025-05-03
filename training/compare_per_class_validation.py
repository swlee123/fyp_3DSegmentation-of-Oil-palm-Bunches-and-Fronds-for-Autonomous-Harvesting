import pandas as pd
import matplotlib.pyplot as plt


# Plot validation mIou and mAcc
# Load the CSV file

file_path = "train_6/validation_per_class_6.csv"  # Update this with the correct path
df = pd.read_csv(file_path)


class_dict = {
    0 : 'Trunk',
    1 : 'Ground',
    2 : 'Branch',
    3 : 'Fruit'
}

# Plot IoU for each class
plt.figure(figsize=(10, 5))
for i in range(4):  # Assuming 4 classes
    plt.plot(df["Epochs"], df[f"Class {i} Iou"], marker='o', label=f'{class_dict[i]} IoU')
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.title("IoU per Class over Epochs")
plt.legend()
plt.grid()
plt.show()

# Plot Accuracy for each class
plt.figure(figsize=(10, 5))
for i in range(4):  # Assuming 4 classes
    plt.plot(df["Epochs"], df[f"Class {i} Acc"], marker='s', label=f'{class_dict[i]} Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy per Class over Epochs")
plt.legend()
plt.grid()
plt.show()
