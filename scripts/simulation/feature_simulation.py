import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from global_var import globalvar as glo
import torch.nn.functional as F
import pickle

class feature_simulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.beta = nn.Parameter(torch.tensor(10e-5).type(torch.cuda.FloatTensor))
        self.sigmoid = nn.Sigmoid()


    # using KL to simulate PC detector on scene level
    def scene_simulation_by_KL(self):
        alpha = self.sigmoid(self.alpha)
        
        pc_scene_feature_path = glo.get_value('pc_scene_feature_path')
        batch_im_id = glo.get_value('batch_im_id')
        batch_m3drpn_scene_glo_feature = glo.get_value('m3drpn_scene_glo_feature') # [2,544,32,110]
        batch_m3drpn_scene_loc_feature = glo.get_value('m3drpn_scene_loc_feature')

        Loss_scene = 0
        b,c,w,h = batch_m3drpn_scene_glo_feature.size()
        for i in range(0, b):
            pc_scene_feature = torch.load(pc_scene_feature_path + '/' + batch_im_id[i] + '.pth')
            pc_scene_feature =torch.from_numpy(pc_scene_feature).permute(2,0,1).cuda()

            zero = torch.zeros(w,h)
            one = torch.ones(w,h)
            mask = torch.where(torch.sum(pc_scene_feature,dim=0)!=0, one, zero) # [32,110]
            mask = mask.unsqueeze(0).repeat([c,1,1]) # [544,32,110]

            p_y = F.softmax(pc_scene_feature.view(c,-1), dim=-1)

            m3drpn_scene_glo_feature = batch_m3drpn_scene_glo_feature[i] # [544,32,110]
            m3drpn_scene_glo_feature_with_mask = m3drpn_scene_glo_feature * mask
            logp_x1 = F.log_softmax(m3drpn_scene_glo_feature_with_mask.view(c,-1), dim=-1)
            KL_global = F.kl_div(logp_x1, p_y, reduction='batchmean')

            m3drpn_scene_loc_feature = batch_m3drpn_scene_loc_feature[i]
            m3drpn_scene_loc_feature_with_mask = m3drpn_scene_loc_feature * mask
            logp_x2 = F.log_softmax(m3drpn_scene_loc_feature_with_mask.view(c,-1), dim=-1)
            KL_local = F.kl_div(logp_x2, p_y, reduction='batchmean')

            Loss_scene = alpha*KL_global + (1-alpha)*KL_local + Loss_scene

        return Loss_scene


    # using KL to simulate PC detector on RoI level
    def RoI_simulation_by_KL(self):
        beta = self.sigmoid(self.beta)

        pc_roi_feature_path = glo.get_value('pc_roi_feature_path')
        batch_im_id = glo.get_value('batch_im_id')
        batch_m3drpn_roi_glo_feature = glo.get_value('m3drpn_roi_glo_feature') # [2,256,32,110]
        batch_m3drpn_roi_loc_feature = glo.get_value('m3drpn_roi_loc_feature')

        Loss_roi = 0
        b,c,w,h = batch_m3drpn_roi_glo_feature.size()
        for i in range(0, b):             
            pc_roi_feature = torch.load(pc_roi_feature_path + '/' + batch_im_id[i] + '.pth')
            pc_roi_feature =torch.from_numpy(pc_roi_feature).permute(2,0,1).cuda()

            zero = torch.zeros(w,h)
            one = torch.ones(w,h)
            mask = torch.where(torch.sum(pc_roi_feature,dim=0)!=0, one, zero) # [32,110]
            mask = mask.unsqueeze(0).repeat([c,1,1]) # [256,32,110]
            
            p_y = F.softmax(pc_roi_feature.view(c,-1), dim=-1)

            m3drpn_roi_glo_feature = batch_m3drpn_roi_glo_feature[i] # [256,32,110] 
            m3drpn_roi_glo_feature_with_mask = m3drpn_roi_glo_feature * mask
            logp_x1 = F.log_softmax(m3drpn_roi_glo_feature_with_mask.view(c,-1), dim=-1)
            KL_global = F.kl_div(logp_x1, p_y, reduction='batchmean')

            m3drpn_roi_loc_feature = batch_m3drpn_roi_loc_feature[i]
            m3drpn_roi_loc_feature_with_mask = m3drpn_roi_loc_feature * mask
            logp_x2 = F.log_softmax(m3drpn_roi_loc_feature_with_mask.view(c,-1), dim=-1)
            KL_local = F.kl_div(logp_x2, p_y, reduction='batchmean')

            Loss_roi = beta*KL_global + (1-beta)*KL_local + Loss_roi

        return Loss_roi
    
    # using L1 to simulate PC detector on scene level
    def scene_simulation_by_L1(self):
        alpha = self.sigmoid(self.alpha)
        
        pc_scene_feature_path = glo.get_value('pc_scene_feature_path')
        batch_im_id = glo.get_value('batch_im_id')
        batch_m3drpn_scene_glo_feature = glo.get_value('m3drpn_scene_glo_feature') # [2,544,32,110]
        batch_m3drpn_scene_loc_feature = glo.get_value('m3drpn_scene_loc_feature')

        Loss_scene = 0
        b,c,w,h = batch_m3drpn_scene_glo_feature.size()
        for i in range(0, b):
            pc_scene_feature = torch.load(pc_scene_feature_path + '/' + batch_im_id[i] + '.pth')
            pc_scene_feature =torch.from_numpy(pc_scene_feature).permute(2,0,1).cuda()

            zero = torch.zeros(w,h)
            one = torch.ones(w,h)
            mask = torch.where(torch.sum(pc_scene_feature,dim=0)!=0, one, zero) # [32,110]
            mask = mask.unsqueeze(0).repeat([c,1,1]) # [544,32,110]

            m3drpn_scene_glo_feature = batch_m3drpn_scene_glo_feature[i] # [544,32,110]
            m3drpn_scene_glo_feature_with_mask = m3drpn_scene_glo_feature * mask
            L1_global = F.l1_loss(m3drpn_scene_glo_feature_with_mask.view(c,-1), pc_scene_feature.view(c,-1))

            m3drpn_scene_loc_feature = batch_m3drpn_scene_loc_feature[i]
            m3drpn_scene_loc_feature_with_mask = m3drpn_scene_loc_feature * mask
            L1_local = F.l1_loss(m3drpn_scene_loc_feature_with_mask.view(c,-1), pc_scene_feature.view(c,-1))

            Loss_scene = alpha*L1_global + (1-alpha)*L1_local + Loss_scene

        return Loss_scene
    
    
    # using L1 to simulate PC detector on RoI level
    def RoI_simulation_by_L1(self):
        beta = self.sigmoid(self.beta)

        pc_roi_feature_path = glo.get_value('pc_roi_feature_path')
        batch_im_id = glo.get_value('batch_im_id')
        batch_m3drpn_roi_glo_feature = glo.get_value('m3drpn_roi_glo_feature') # [2,256,32,110]
        batch_m3drpn_roi_loc_feature = glo.get_value('m3drpn_roi_loc_feature')

        Loss_roi = 0
        b,c,w,h = batch_m3drpn_roi_glo_feature.size()
        for i in range(0, b):             
            pc_roi_feature = torch.load(pc_roi_feature_path + '/' + batch_im_id[i] + '.pth')
            pc_roi_feature =torch.from_numpy(pc_roi_feature).permute(2,0,1).cuda()

            zero = torch.zeros(w,h)
            one = torch.ones(w,h)
            mask = torch.where(torch.sum(pc_roi_feature,dim=0)!=0, one, zero) # [32,110]
            mask = mask.unsqueeze(0).repeat([c,1,1]) # [256,32,110]

            m3drpn_roi_glo_feature = batch_m3drpn_roi_glo_feature[i] # [256,32,110] 
            m3drpn_roi_glo_feature_with_mask = m3drpn_roi_glo_feature * mask
            L1_global = F.l1_loss(m3drpn_roi_glo_feature_with_mask.view(c,-1), pc_roi_feature.view(c,-1))

            m3drpn_roi_loc_feature = batch_m3drpn_roi_loc_feature[i]
            m3drpn_roi_loc_feature_with_mask = m3drpn_roi_loc_feature * mask
            L1_local = F.l1_loss(m3drpn_roi_loc_feature_with_mask.view(c,-1), pc_roi_feature.view(c,-1))

            Loss_roi = beta*L1_global + (1-beta)*L1_local + Loss_roi

        return Loss_roi
