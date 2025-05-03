import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
curr_dir = os.listdir()
train_dir = []

for i in curr_dir:
    if i.startswith("train_") and not i.endswith(".py"):
        train_dir.append(i)


total_Iou =[]
total_Acc = []
total_val_iou = []
total_val_acc = []

for dir in train_dir:
    # Load the CSV file

    dir_num = dir.split("_")[1]
    print(dir_num)
    training_logs_path = os.path.join(dir, f"training_logs_{dir_num}_mean.csv")
    validation_logs_path = os.path.join(dir, f"validation_logs_{dir_num}_mean.csv")
    
    training_logs_df = pd.read_csv(training_logs_path)
    validation_logs_df = pd.read_csv(validation_logs_path)
    
    Iou = training_logs_df["mIou"]
    Acc = training_logs_df["mAcc"]
    
    val_iou = validation_logs_df["mIou"]
    val_acc = validation_logs_df["mAcc"]
    
    total_Iou.append(Iou)
    total_Acc.append(Acc)
    total_val_iou.append(val_iou)
    total_val_acc.append(val_acc)
    
    
# Calculate the average values

average_Iou = [sum(x) / len(x) for x in zip(*total_Iou)]
average_Acc = [sum(x) / len(x) for x in zip(*total_Acc)]

average_val_iou = [sum(x) / len(x) for x in zip(*total_val_iou)]
average_val_acc = [sum(x) / len(x) for x in zip(*total_val_acc)]


# epoch
epochs = [i for i in range(1, len(average_Iou) + 1)]

# Plot Training 
plt.figure(figsize=(10, 5))
plt.plot(epochs, average_Iou, label="Mean Iou", marker = 'o',color='blue', alpha=0.7)
plt.plot(epochs, average_Acc, label="Mean Acc", marker = 'o',color='red', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Mean Iou and Mean Acc")
plt.title("Training Mean Iou and Mean Acc across All Folds")
plt.legend()
plt.show()

# plot validation
plt.figure(figsize=(10, 5))
plt.plot(epochs, average_val_iou, label="Mean Iou", marker = 'o',color='blue', alpha=0.7)
plt.plot(epochs, average_val_acc, label="Mean Acc", marker = 'o',color='red', alpha=0.7)
plt.xlabel("Epochs")
plt.ylabel("Mean Iou and Mean Acc")
plt.title("Validation Mean Iou and Mean Acc across All Folds")
plt.legend()
plt.show()

# put all validation miou and acc into a excel 
df_1 = pd.DataFrame(total_val_acc)
df_2 = pd.DataFrame(total_val_iou)


df_1.columns = ['Epoch1','Epoch2','Epoch3','Epoch4','Epoch5']
# 2. add Fold column on the left
df_1.insert(0, 'Fold', np.arange(len(df_1),dtype=int))

df_2.columns = ['Epoch1','Epoch2','Epoch3','Epoch4','Epoch5']
# 2. add Fold column on the left
df_2.insert(0, 'Fold', np.arange(len(df_2),dtype=int))

# save to csv 

df_1.to_csv('total_val_acc.csv', index=False)
df_2.to_csv('total_val_iou.csv', index=False)


