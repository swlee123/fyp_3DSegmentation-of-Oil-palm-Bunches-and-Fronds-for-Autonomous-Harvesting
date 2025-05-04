import torch

# Load the processed data
data_path = "/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/train/scene0155_24_inst_nostuff.pth"
coords, colors, sem_labels = torch.load(data_path)

# Convert sem_labels to a PyTorch tensor
sem_labels = torch.tensor(sem_labels)  

# Now this should work
print("Unique class labels:", torch.unique(sem_labels))

# expect there will be 0,1,2,3 ,  -100 is unlabelled data 