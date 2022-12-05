# MonoSIM:Simulating Learning Behaviors of Heterogeneous Point Cloud Object Detectors for Monocular 3D Object Detection

This repository is modified from Garrick Brazil and Xiaoming Liu's [Monocular 3D Region Proposal Network](https://github.com/garrickbrazil/M3D-RPN) based on their ICCV 2019 arXiv report and JuliaChae's [M3D-RPN-Waymo](https://github.com/JuliaChae/M3D-RPN-Waymo) repository to realize the MonoSIM's function. In addition, we ran the [PV-RCNN](https://github.com/open-mmlab/OpenPCDet)'s code to obtain the necessary data. Please see their project page for more information. 

# Introduction

This repository was used to train and evaluate the MonoSIM monocular detector on the the [Waymo Open Dataset](https://waymo.com/open/), the purpose is to research the influence of monocular detector simulating the behaviors of heterogeneous point cloud object detector. To achieve it, we propose one scene-level simulation module, one RoI-level simulation module and one response-level simulation module, which are progressively used for the detector's full feature learning and prediction pipeline. Results show that our method consistently improves the performance for a large margin without changing their network architectures.

# Usage

This repo is tested on our local environment (python=3.6.13, cuda=10.2, pytorch=1.7.1), and we recommend you to use anaconda to create a vitural environment:
    conda create -n monosim python=3.6.13
