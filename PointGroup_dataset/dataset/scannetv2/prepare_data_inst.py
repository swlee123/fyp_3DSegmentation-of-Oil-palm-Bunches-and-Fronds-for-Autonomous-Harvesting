'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

# 24/3 task : fix this file so it could convert those 4 files into .pth with all class correctly

import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse
import scannet_util


parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))

# Match files with labels
files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))  # Labels file
files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))


print(len(files))
print(len(files3))
assert len(files) == len(files2), "Mismatch between .ply and .labels.ply files"
assert len(files) == len(files3)
assert len(files) == len(files4), "{} {}".format(len(files), len(files4))

def f(fn, fn2):
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'
    print(f"Processing: {fn}")

    # Read point cloud
    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    num_points = coords.shape[0]  # Store the number of points

    # Read labels from `*_vh_clean_2.labels.ply`
    f2 = plyfile.PlyData().read(fn2)
    labels_data = f2.elements[0]  # Get labels

    # Ensure label file has the correct number of points
    label_names = ['scalar_Trunk', 'scalar_Ground', 'scalar_Branch', 'scalar_Fruit']
    sem_labels = np.ones(num_points) * -100  # Initialize labels to -100 (unlabeled)

    if len(labels_data) != num_points:
        print(f"⚠️ Warning: {fn2} has {len(labels_data)} points, but {fn} has {num_points} points!")

    # Match labels by index (ensure both have the same length)
    min_points = min(num_points, len(labels_data))  # Take the smaller count
    sem_labels = sem_labels[:min_points]  # Trim if necessary

    for i, label_name in enumerate(label_names):
        if label_name in labels_data.data.dtype.names:
            label_values = np.array(labels_data[label_name])
            label_values = label_values[:min_points]  # Trim to match point cloud size
            sem_labels[label_values >= 0] = i  # Assign class index if value is not NaN
    
    # Save processed data
    torch.save((coords, colors, sem_labels), fn[:-15] + '_inst_nostuff.pth')
    print(f"✅ Saved: {fn[:-15]}_inst_nostuff.pth")

# Run multiprocessing
p = mp.Pool(processes=mp.cpu_count())
p.starmap(f, zip(files, files2))  # Use `zip` to pass both .ply and .labels.ply
p.close()
p.join()
