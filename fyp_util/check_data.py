import os
import shutil
from collections import defaultdict
import re

TRAIN_DIR = '/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/train'
VAL_DIR = '/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/val'
TEST_DIR = '/home/swlee/scannetdata/PointGroup/dataset/scannetv2/PointGroup/dataset/scannetv2/test'

def get_ss_scene_prefixes(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    scene_groups = defaultdict(list)

    for f in files:
        if f.startswith("scene"):
            match = re.match(r"(scene\d+_\d+)",f)
            if match:
                prefix = match.group(1)
                scene_groups[prefix].append(f)
            else:
                print(f"Warning: No match for file {f}")

    return scene_groups

def get_ori_scene_prefixes(ss_list):

    original_list = set()
    for f in ss_list:
        if f.startswith("scene"): # all looks like scene0152_15 , wan to look at last 2 number 
            prefix = f.split("_")[1]  # e.g. scene0151_03
            name = f[:-4]
            ori_name = name+prefix
            original_list.add(ori_name)
    
    return list(original_list)


if __name__ == "__main__":
    os.makedirs(VAL_DIR, exist_ok=True)

    # Get all unique scene prefixes
    ss_train_scenes = list(get_ss_scene_prefixes(TRAIN_DIR))
    ss_train_scenes.sort()
    
    train_scenes = get_ori_scene_prefixes(ss_train_scenes)
    train_scenes.sort()
    
    
    print(f"✅Total original scenes in train dir: {len(train_scenes)}")
    print("train/ original scenes:")
    for scene in train_scenes:
        print(scene)
    
    print(f"✅Total subsampled scenes in train dir: {len(ss_train_scenes)}")
    print("train/ ss scenes:")
    for scene in ss_train_scenes:
        print(scene)
        
    ss_val_scenes = list(get_ss_scene_prefixes(VAL_DIR).keys())
    ss_val_scenes.sort()

    val_scenes = get_ori_scene_prefixes(ss_val_scenes)
    val_scenes.sort()
    
    print(f"✅Total original scenes in val dir: {len(val_scenes)}")
    print("val/ original scenes:")
    for scene in val_scenes:
        print(scene)
        
    print(f"✅Total subsampled scenes in val dir: {len(ss_val_scenes)}")
    print("val/ scenes:")
    for scene in ss_val_scenes:
        print(scene)
        
