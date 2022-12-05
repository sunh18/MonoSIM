import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import pickle
import sys
sys.path.append(os.getcwd() + '/pytorch3d')
import pytorch3d
print(pytorch3d.__file__)
from pts_render import render as rd

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class PVRCNN_render(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_path = os.getcwd()
        
        self.scene_pts_path = self.base_path + '/data/pvrcnn_material/scene_pts'
        self.scene_feature_path = self.base_path + '/data/pvrcnn_material/scene_feature'
        self.roi_pts_path = self.base_path + '/data/pvrcnn_material/roi_pts'
        self.roi_feature_path = self.base_path + '/data/pvrcnn_material/roi_feature'
        
        self.waymo_train_list_path = self.base_path + '/data/pvrcnn_material/waymo_id/train.txt'
        self.waymo_to_kitti_dict_path = self.base_path + '/data/pvrcnn_material/waymo_id/train_waymo_to_kitti_id_dict.pkl'
        self.calib_path = self.base_path + '/data/waymo/training/calib'
        self.image_path = self.base_path + '/data/waymo/training/image_0'
        
        self.render_scene_feature_path = self.base_path + '/data/pvrcnn_material/render_scene_feature'
        self.render_roi_feature_path = self.base_path + '/data/pvrcnn_material/render_roi_feature'


    def forward(self):
        if not os.path.isdir(self.render_scene_feature_path):
            os.mkdir(self.render_scene_feature_path)
        if not os.path.isdir(self.render_roi_feature_path):
            os.mkdir(self.render_roi_feature_path) 

        train_idx = []
        for line in open(self.waymo_train_list_path,"r"):
            line = line[:-1]
            train_idx.append(line)

        with open(self.waymo_to_kitti_dict_path, 'rb') as f2:
            waymo_to_kitti_id_dict = pickle.load(f2)

        count = 0
        total = len(train_idx)
        for idx_waymo in train_idx:
            try:
                idx_kitti = waymo_to_kitti_id_dict[idx_waymo]
                self.im = cv2.imread(self.image_path + '/' + idx_kitti + '.png')
                self.K, self.RT = rd.get_calib(self.calib_path + '/' + idx_kitti + '.txt')
                self.roi_pts = torch.load(self.roi_pts_path + '/' + idx_waymo + '.pth')
                self.roi_feature = torch.load(self.roi_feature_path + '/' + idx_waymo + '.pth')
                self.scene_pts = torch.load(self.scene_pts_path + '/' + idx_waymo + '.pth')[:,1:4]
                self.scene_feature = torch.load(self.scene_feature_path + '/' + idx_waymo + '.pth')
                # self.rawpoints_coords = torch.load(self.rawpoints_coords_path + '/' + idx_waymo + '.pth')[:,0:3]

                render_roi_feature = self.pvrcnn_roi_render()
                torch.save(render_roi_feature, self.render_roi_feature_path + '/'+ idx_kitti + '.pth', _use_new_zipfile_serialization=False)
                render_scene_feature = self.pvrcnn_scene_render()
                torch.save(render_scene_feature, self.render_scene_feature_path + '/'+ idx_kitti + '.pth', _use_new_zipfile_serialization=False)
            except:
                print(idx_waymo, 'not existed or package version problems')

            count = count + 1
            print('render finished: ',count,'/',total)

        # return res_roi_render, res_keypoints_render


    def pvrcnn_roi_render(self):
        K = self.K
        RT = self.RT
        roi_pts = self.roi_pts
        rois_feature = self.roi_feature

        # roi区域6*6*6划分，点坐标投影到相机系
        n_roi, n_roi_p, d_p = roi_pts.size()
        roi_grid_points = roi_pts.view(n_roi*n_roi_p,d_p).permute(1,0).cuda()
        roi_grid_points = torch.cat((roi_grid_points,torch.ones(1,roi_grid_points.shape[1]).cuda()),0)
        roi_grid_points_to_cam = np.dot(RT, roi_grid_points.cpu().numpy())  # 刚体变换
        point_behind_idx = roi_grid_points_to_cam[2, :]>1
        roi_grid_points_to_cam = roi_grid_points_to_cam[:, roi_grid_points_to_cam[2, :] > 1]
        roi_grid_points_to_cam = torch.from_numpy(roi_grid_points_to_cam.astype(np.float32)).unsqueeze(dim=0).cuda()

        # roi_points的特征
        roi_grid_points_features = torch.empty(n_roi, n_roi_p, 256, 1)
        for i in range(n_roi):
            for j in range(n_roi_p):
                roi_grid_points_features[i][j] = rois_feature[i]
        roi_grid_points_features = roi_grid_points_features.view(n_roi*n_roi_p, 256).permute(1,0)
        roi_grid_points_features = roi_grid_points_features[:,point_behind_idx].unsqueeze(dim=0).cuda()

        # render
        opts = rd.Options()
        c = (512/self.im.shape[0])/16
        K = K*c
        K[2,2] = 1
        K = np.array(K, dtype=np.float32)
        K = torch.from_numpy(K).unsqueeze(0).cuda()
        pts_manipulator = rd.PtsManipulator((32, 48), 256, opts)
        roi_render_res = pts_manipulator(roi_grid_points_features, roi_grid_points_to_cam, K).detach().squeeze().permute(1,2,0).data.cpu().numpy()

        return roi_render_res

    def pvrcnn_scene_render(self):
        K = self.K
        RT = self.RT
        scene_pts = self.scene_pts
        scene_feature = self.scene_feature
       
        # scene keypoints投影到相机系
        cloud_points = scene_pts.permute(1,0).cuda()
        cloud_points = torch.cat((cloud_points,torch.ones(1,cloud_points.shape[1]).cuda()),0)
        cloud_points_to_cam = np.dot(RT, cloud_points.cpu().numpy())  # 刚体变换
        point_behind_idx = cloud_points_to_cam[2, :]>1
        cloud_points_to_cam = cloud_points_to_cam[:, cloud_points_to_cam[2, :] > 1]
        cloud_points_to_cam = torch.tensor(cloud_points_to_cam.astype(np.float32)).unsqueeze(dim=0).cuda()

        # scene keypoints的特征
        cloud_points_features = scene_feature.permute(1,0)
        cloud_points_features = cloud_points_features[:,point_behind_idx].unsqueeze(dim=0).cuda()

        # render
        opts = rd.Options()
        c = (512/self.im.shape[0])/16
        K = K*c
        K[2,2] = 1
        K = np.array(K, dtype=np.float32)
        K = torch.from_numpy(K).unsqueeze(0).cuda()
        pts_manipulator = rd.PtsManipulator((32, 48), 544, opts)
        scene_render_res = pts_manipulator(cloud_points_features, cloud_points_to_cam, K).detach().squeeze().permute(1,2,0).data.cpu().numpy()

        return scene_render_res

if __name__ == "__main__":
    render = PVRCNN_render().cuda()
    res = render()