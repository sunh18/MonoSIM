# MonoSIM:Simulating Learning Behaviors of Heterogeneous Point Cloud Object Detectors for Monocular 3D Object Detection

This repository is modified from Garrick Brazil and Xiaoming Liu's [Monocular 3D Region Proposal Network](https://github.com/garrickbrazil/M3D-RPN) based on their ICCV 2019 arXiv report and JuliaChae's [M3D-RPN-Waymo](https://github.com/JuliaChae/M3D-RPN-Waymo) repository to realize the MonoSIM's function. In addition, we ran the [PV-RCNN](https://github.com/open-mmlab/OpenPCDet)'s code to obtain the necessary data. Please see their project page for more information. 

# Introduction

This repository was used to train and evaluate the MonoSIM monocular detector on the the [Waymo Open Dataset](https://waymo.com/open/), the purpose is to research the influence of monocular detector simulating the behaviors of heterogeneous point cloud object detector. To achieve it, we propose one scene-level simulation module, one RoI-level simulation module and one response-level simulation module, which are progressively used for the detector's full feature learning and prediction pipeline. Results show that our method consistently improves the performance for a large margin without changing their network architectures.

# Installation
## MonoSIM vitural environment
This repo is tested on our local environment (python=3.6.13, cuda=10.2, pytorch=1.7.1), and we recommend you to use anaconda to create a vitural environment:
```
conda create -n monosim python=3.6.13
conda activate monosim
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
git clone https://github.com/sunh18/MonoSIM.git
```

## PyTorch3D Core Library
* We used the [PyTorch3D](https://github.com/facebookresearch/pytorch3d) to build the feature rendering module, so we need to install the corresponding dependencies first:
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install numpy==1.19.5
```
* Here we suggest installing Python3D locally (provided by MonoSIM) to avoid version errors:
```
conda install pytorch3d -c pytorch3d
cd pytorch3d && pip install -e .
```
* To rebuild after installing from a local clone run, ```rm -rf build/ **/*.so ```then ```pip install -e .```. For more detailed installation introduction, please refer to [PyTorch3D](https://github.com/facebookresearch/pytorch3d) page.

## Other Libraries
```
cd ..
conda install  easydict shapely
conda install -c menpo opencv3=3.1.0 openblas
conda install cython scikit-image h5py nose pandas protobuf atlas libgfortran jsonpatch
conda install -c conda-forge visdom
cd lib/nms && python setup.py build_ext --inplace && rm -rf build
```

# Data setup
## Waymo Dataset
Please download the official [Waymo Open Dataset](https://waymo.com/open/) and organize the downloaded files in any location as follows:
```
├── Waymo
│   ├── original
│   │   │──training
│   │   │   ├──training_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──training_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
│   │   │──validation
│   │   │   ├──validation_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──validation_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
│   │   │──testing
│   │   │   ├──testing_0000
│   │   │   │   ├─.tfrecord files
│   │   │   ├──testing_0001
│   │   │   │   ├─.tfrecord files
│   │   │   ├──...
```
