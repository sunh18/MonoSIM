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
Please download the official [Waymo Open Dataset](https://waymo.com/open/) and run the Waymo-Kitti adapter to reformat the data appropriately. Clone the repo Waymo-[Kitti-Adapter](https://github.com/JuliaChae/Waymo-Kitti-Adapter) and follow the instructions in its README file. Convert all training, testing and validation files. After runing the adapter, the Waymo data path should look something like with reformatted dataset in the "adapted" folder:
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

# Pretrained checkpoints

[pretrained_ckpt](https://pan.baidu.com/s/1kFPkUvc9thE0mgzaJPVLpA?pwd=ljjq)


<div class="WordSection1" style="layout-grid:15.6pt">

<div align="center">

<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="63" rowspan="2" style="width:47.45pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width="245" colspan="4" style="width:183.95pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">3D
  <span class="SpellE">mAP</span></span></p>
  </td>
  <td width="245" colspan="4" style="width:183.4pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">3DmAPH</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="57" style="width:42.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">overall</span></p>
  </td>
  <td width="59" style="width:44.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0~30m</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">30~50m</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">50m~∞</span></p>
  </td>
  <td width="56" style="width:42.35pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">overall<o:p></o:p></span></p>
  </td>
  <td width="59" style="width:44.0pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0~30m<o:p></o:p></span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">30~50m<o:p></o:p></span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">50m~∞<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td width="63" style="width:47.45pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">LEVEL_1</span></p>
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US">IoU</span></span><span lang="EN-US">=0.7</span></p>
  </td>
  <td width="57" style="width:42.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">1.60</span></p>
  </td>
  <td width="59" style="width:44.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">6.08</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.40</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.01</span></p>
  </td>
  <td width="56" style="width:42.35pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">1.59</span></p>
  </td>
  <td width="59" style="width:44.0pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">6.02</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.39</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.01</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3">
  <td width="63" style="width:47.45pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">LEVEL_2<o:p></o:p></span></p>
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US">IoU</span></span><span lang="EN-US">=0.7<o:p></o:p></span></p>
  </td>
  <td width="57" style="width:42.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">1.49</span></p>
  </td>
  <td width="59" style="width:44.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">6.05</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.38</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.004</span></p>
  </td>
  <td width="56" style="width:42.35pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">1.48</span></p>
  </td>
  <td width="59" style="width:44.0pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">6.00</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.38</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.004</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4">
  <td width="63" style="width:47.45pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">LEVEL_1</span></p>
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US">IoU</span></span><span lang="EN-US">=0.5</span></p>
  </td>
  <td width="57" style="width:42.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">8.16</span></p>
  </td>
  <td width="59" style="width:44.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">25.86</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">3.62</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.08</span></p>
  </td>
  <td width="56" style="width:42.35pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">8.04</span></p>
  </td>
  <td width="59" style="width:44.0pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">25.48</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">3.55</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.08</span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;mso-yfti-lastrow:yes">
  <td width="63" style="width:47.45pt;border:solid windowtext 1.0pt;border-top:
  none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">LEVEL_2<o:p></o:p></span></p>
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US">IoU</span></span><span lang="EN-US">=0.5</span></p>
  </td>
  <td width="57" style="width:42.6pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">7.58</span></p>
  </td>
  <td width="59" style="width:44.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">25.75</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">3.49</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.07</span></p>
  </td>
  <td width="56" style="width:42.35pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">7.47</span></p>
  </td>
  <td width="59" style="width:44.0pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">25.37</span></p>
  </td>
  <td width="65" style="width:48.8pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">3.43</span></p>
  </td>
  <td width="64" style="width:48.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US">0.07</span></p>
  </td>
 </tr>
</tbody></table>

</div>

<p class="MsoNormal"><span lang="EN-US"><o:p>&nbsp;</o:p></span></p>

</div>

# Training


# Testing

