# MonoSIM:Simulating Learning Behaviors of Heterogeneous Point Cloud Object Detectors for Monocular 3D Object Detection
This repository is the official code for paper "MonoSIM: Simulating Learning Behaviors of Heterogeneous Point Cloud Object Detectors for Monocular 3D Object Detection". The paper can be download in .

<img src="https://github.com/sunh18/MonoSIM/blob/master/MonoSIM.png" width="600" height="400" alt="MonoSIM"/><br/>

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
Please download the official [Waymo Open Dataset](https://waymo.com/open/) and run the Waymo-Kitti adapter to reformat the data appropriately. Clone the repo [Waymo-Kitti-Adapter](https://github.com/JuliaChae/Waymo-Kitti-Adapter) and follow the instructions in its README file. Convert all training, testing and validation files. After runing the adapter, the Waymo data path should look something like with reformatted dataset in the "adapted" folder:
```
├── Waymo
│   ├── original
│   │   │──training & testing & validation
│   ├── adapted
│   │   │──training
│   │   │   ├──calib & velodyne & label_0 & image_0
│   │   │──validation
│   │   │   ├──calib & velodyne & label_0 & image_0
│   │   │──testing
│   │   │   ├──calib & velodyne & label_0
```
Finally, symlink it using the following:
```
mkdir data/waymo
ln -s ${ADAPTED_WAYMO_DIR} data/waymo
```
Then use the following scripts to extract the data splits, which use softlinks to the above directory for efficient storage. Modify the camera view argument at the top of the code to correspond to the views that you are working with:
```
python data/waymo_split/setup_split.py
```

## Simulation Material
We provide material used for MonoSIM simulation, including scene-level and RoI-level point clouds and feature from PV-RCNN network. Limited by the file size, we store it in [pvrcnn_material](https://pan.baidu.com/s/1KPpWg8YN290h1YGEuP2pCg?pwd=nzi7) for downloading. After downloading and unzip, the folder path should look like:
```
├── data
│   ├── kitti_split1
│   ├── kitti_split2
│   ├── pvrcnn_material
│   │   │──render_roi_feature
│   │   │   ├──000000000000000.pth
│   │   │   ├──000000000000005.pth
│   │   │   ├──...
│   │   │──render_scene_feature
│   │   │   ├──000000000000000.pth
│   │   │   ├──000000000000005.pth
│   │   │   ├──...
│   │   │──roi_feature
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_000.pth
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_005.pth
│   │   │   ├──...
│   │   │──roi_pts
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_000.pth
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_005.pth
│   │   │   ├──...
│   │   │──scene_feature
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_000.pth
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_005.pth
│   │   │   ├──...
│   │   │──scene_pts
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_000.pth
│   │   │   ├──segment-10017090168044687777_6380_000_6400_000_with_camera_labels_005.pth
│   │   │   ├──...
│   │   │──waymo_id
│   │   │   ├──train_waymo_to_kitti_id_dict.pkl
│   │   │   ├──train.txt
│   │   │   ├──validation_waymo_to_kitti_id_dict.pkl
│   │   │   ├──validation.txt
│   │   │──soft_label_0.zip
│   ├── waymo
│   ├── waymo_split
```
To conduct Response-Level simulation, ```unzip soft_label_0.zip``` to replace ```waymo/training/label_0```, and after that, run ```python data/waymo_split/setup_split.py``` again.

The render_roi_feature and render_scene_feature can be generated by following command： 
```
python scripts/render_feature.py
```

# Pretrained checkpoints
We provide [pretrained_ckpt](https://pan.baidu.com/s/1kFPkUvc9thE0mgzaJPVLpA?pwd=ljjq) for downloading. Please place the extracted file under /pretrained_ckpt.

# Training
We use visdom for visualization and graphs. Optionally, start the server by command line:
```
python -m visdom.server -port 8100 -readonly
```
The port can be customized in scripts/config files. The training monitor can be viewed at http://localhost:8100
Training is split into a warmup and main configurations. Review the configurations in scripts/config for details. The model output path and gpu id can be self-defined.
```
// First train the warmup (without depth-aware)
python scripts/train_rpn_3d.py --config=waymo_3d_multi_warmup --output=debug --gpu_id=2

// Then train the main experiment (with depth-aware)
python scripts/train_rpn_3d.py --config=waymo_3d_multi_main --output=debug --gpu_id=2
```
If your training is accidentally stopped, you can resume at a checkpoint based on the snapshot with the restore flag. For example to resume training starting at iteration 10k, use the following command.
```
python scripts/train_rpn_3d.py --config=waymo_3d_multi_main --restore=10000 --output=debug --gpu_id=2
```

# Testing
Testing requires paths to the configuration file and model weights, exposed variables near the top scripts/test_rpn_3d.py. To test a configuration and model, simply update the variables and run the test file as below.
```
python scripts/test_rpn_3d.py 
```
After running test_rpn_3d.py, the predicted labels are stored under /eval/data. Utilizing [Waymo-Kitti-Adapter](https://github.com/JuliaChae/Waymo-Kitti-Adapter) to create waymo eval bins by running:
```
python create_waymo_eval_bins.py --preds="pred labels path" --gt="gt labels path" --calib="calib path" --output_dir="bins path"
```
Utilizing [Waymo eval tool](https://github.com/waymo-research/waymo-open-dataset/blob/master/docs/quick_start.md) to get official evaluation results.

# Acknowlegment
This repository is modified from Garrick Brazil and Xiaoming Liu's [Monocular 3D Region Proposal Network](https://github.com/garrickbrazil/M3D-RPN) based on their ICCV 2019 arXiv report and JuliaChae's [M3D-RPN-Waymo](https://github.com/JuliaChae/M3D-RPN-Waymo) repository to realize the MonoSIM's function. In addition, we ran the [PV-RCNN](https://github.com/open-mmlab/OpenPCDet)'s code to obtain the necessary data. Please see their project page for more information. This repo also benefits from the excellent work [PyTorch3D](https://github.com/facebookresearch/pytorch3d). Please also consider citing them.
