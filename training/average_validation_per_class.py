import pandas as pd
import matplotlib.pyplot as plt
import os

curr_dir = os.listdir()
train_dir = []

for i in curr_dir:
    if i.startswith("train_"):
        train_dir.append(i)


total_class_0_iou = []
total_class_0_acc = []
total_class_1_iou = []
total_class_1_acc = []
total_class_2_iou = []
total_class_2_acc = []
total_class_3_iou = []
total_class_3_acc = []

for dir in train_dir:
    # Load the CSV file

    dir_num = dir.split("_")[1]
    
    validation_class_path = os.path.join(dir, f"validation_per_class_{dir_num}.csv")

    
    validation_df = pd.read_csv(validation_class_path)
    
    # read row by row of Class 0 Iou,Class 0 Acc,Class 1 Iou,Class 1 Acc,Class 2 Iou,Class 2 Acc,Class 3 Iou,Class 3 Acc
    for i in range(0, 8, 2):
        class_iou = validation_df.iloc[:, i]
        class_acc = validation_df.iloc[:, i + 1]
        
        if i == 0:
            total_class_0_iou.append(class_iou)
            total_class_0_acc.append(class_acc)
        elif i == 2:
            total_class_1_iou.append(class_iou)
            total_class_1_acc.append(class_acc)
        elif i == 4:
            total_class_2_iou.append(class_iou)
            total_class_2_acc.append(class_acc)
        elif i == 6:
            total_class_3_iou.append(class_iou)
            total_class_3_acc.append(class_acc)
    
# Calculate the average values
average_class_0_iou = [sum(x) / len(x) for x in zip(*total_class_0_iou)]
average_class_0_acc = [sum(x) / len(x) for x in zip(*total_class_0_acc)]
average_class_1_iou = [sum(x) / len(x) for x in zip(*total_class_1_iou)]
average_class_1_acc = [sum(x) / len(x) for x in zip(*total_class_1_acc)]
average_class_2_iou = [sum(x) / len(x) for x in zip(*total_class_2_iou)]
average_class_2_acc = [sum(x) / len(x) for x in zip(*total_class_2_acc)]
average_class_3_iou = [sum(x) / len(x) for x in zip(*total_class_3_iou)]
average_class_3_acc = [sum(x) / len(x) for x in zip(*total_class_3_acc)]

# epoch 
epoch = [i for i in range(1, len(average_class_0_iou) + 1)]

class_dict = {
    0 : 'Trunk',
    1 : 'Ground',
    2 : 'Branch',
    3 : 'Fruit'
}


# Plot all class iou 
plt.figure(figsize=(10, 5))
plt.plot(epoch, average_class_0_iou, label="Trunk Iou",  marker='o',color='blue', alpha=0.7)
plt.plot(epoch, average_class_1_iou, label="Ground Iou", marker='o',color='red', alpha=0.7)   
plt.plot(epoch, average_class_2_iou, label="Branch Iou", marker='o',color='green', alpha=0.7)
plt.plot(epoch, average_class_3_iou, label="Fruit Iou", marker='o',color='orange', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Iou")
plt.title("Average Validation Iou per Class Across All Folds")
plt.legend()
plt.grid()
plt.show()

# plot all class acc
plt.figure(figsize=(10, 5))
plt.plot(epoch, average_class_0_acc, label="Trunk Acc",marker='o', color='blue', alpha=0.7)
plt.plot(epoch, average_class_1_acc, label="Ground Acc",marker='o', color='red', alpha=0.7)
plt.plot(epoch, average_class_2_acc, label="Branch Acc",marker='o', color='green', alpha=0.7)
plt.plot(epoch, average_class_3_acc, label="Fruit Acc", marker='o',color='orange', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.title("Average Validation Acc per Class Across All Folds")
plt.legend()
plt.grid()
plt.show()


        