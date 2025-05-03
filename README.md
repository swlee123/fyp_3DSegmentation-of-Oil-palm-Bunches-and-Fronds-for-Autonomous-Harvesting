
# 3D Segmentation of Oil palm Bunches and Fronds for Autonomous Harvesting

This file documented common instruction to apply dataset processing , labelling , training , validation ,testing and evaluation on Stratitfied Transformer for palm oil 3D point cloud segmentation.

Official Github Implementation of Stratified Transformer : [**Stratified Transformer for 3D Point Cloud Segmentation**](https://github.com/dvlab-research/Stratified-Transformer?tab=readme-ov-file)


## Model Overview 
<div align="center">
  <img src="figs/fig.jpg"/>
</div>


# Get Started

## Environment Setup 

The setup in official github lack of certain information and detail instructions.
ST setup only works for Linux , as certain dependencies requires file that is only available in Linux, so I opt to use WSL on my Windows machine.

```  
Window Subsystem for Linux (WSL)   20.04 
Python                             3.7.4 
torch                              1.10.0+cu113
Cuda Toolkit                       11.3
GCC                                7.5
```


1. Install WSL 

Installed Ubuntu 20.04 using the Microsoft Store

Open `cmd` and input the command
```
wsl
```
Make sure to set default WSL Distribution to the Ubuntu environment
```  
wsl --set-default Ubuntu-20.04
```

List out all the distribution available 
```
wsl --list --verbose
```

Start the Ubuntu sys in WSL
```
wsl -d Ubuntu-20.04
```



2. Install python 3.7.4 in WSL

Get python 3.7.4 
```bash
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

wget https://www.python.org/ftp/python/3.7.4/Python-3.7.4.tgz
sudo tar xzf Python-3.7.4.tgz
```

Then install
```
cd Python-3.7.4

sudo ./configure
sudo make
sudo make install
```

Set python3.7 as default version 

```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.7
```

3. Install CUDA 11.3 

For Cuda toolkit installation : https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

Tips : directly using wget in WSL is x10 slow, can try to curl or download in Window first then move it to WSL directory to gain access 

4. Install gcc 7.5 in WSl

```bash
sudo add-apt-repository ppa:jonathonf/gcc
sudo apt-get update
```

Then run 

```bash
apt-cache search gcc-7
```

Install `gcc-7`
```bash
sudo apt install gcc-7

```
verify with 
```bash

gcc-7 --version
```
Set GCC-7 as default ( if have another gcc version)

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70
sudo update-alternatives --config gcc
```


## Dependencies Setup

1. Virtual Environment 

```bash
python -m venv env  

source env/bin/activate
```

2. Install dependencies

```
pip install -r requirements.txt
```


3. Compile pointops

Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (Note that if you install cuda by conda, it won't provide nvcc and you should install cuda manually.). Then, compile and install pointops2 as follows. 
```
cd lib/pointops2
python3 setup.py install
```


### Common error and fix 

`ImportError: No module named bz2` 
fix : https://stackoverflow.com/questions/12806122/missing-python-bz2-module















## Datasets Preparation

### S3DIS
Please refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch for S3DIS preprocessing. Then modify the `data_root` entry in the .yaml configuration file.

### ScanNetv2
Please refer to https://github.com/dvlab-research/PointGroup for the ScanNetv2 preprocessing. Then change the `data_root` entry in the .yaml configuration file accordingly.

## Training

### S3DIS
- Stratified Transformer
```
python3 train.py --config config/s3dis/s3dis_stratified_transformer.yaml
```

- 3DSwin Transformer (The vanilla version shown in our paper)
```
python3 train.py --config config/s3dis/s3dis_swin3d_transformer.yaml
```

### ScanNetv2
- Stratified Transformer
```
python3 train.py --config config/scannetv2/scannetv2_stratified_transformer.yaml
```

- 3DSwin Transformer (The vanilla version shown in our paper)
```
python3 train.py --config config/scannetv2/scannetv2_swin3d_transformer.yaml
```

Note: It is normal to see the the results on S3DIS fluctuate between -0.5\% and +0.5\% mIoU maybe because the size of S3DIS is relatively small, while the results on ScanNetv2 are relatively stable.

## Testing
For testing, first change the `model_path`, `save_folder` and `data_root_val` (if applicable) accordingly. Then, run the following command. 
```
python3 test.py --config [YOUR_CONFIG_PATH]
```
