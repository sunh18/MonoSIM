from easydict import EasyDict as edict
import numpy as np
import torch
from global_var import globalvar as glo

glo._init()

def Config():

    conf = edict()
        
    # ----------------------------------------
    #  general
    # ----------------------------------------

    conf.model = 'densenet121_3d_dilate_for_MonoSIM'
    
    # solver settings
    conf.solver_type = 'sgd'
    conf.lr = 0.004
    conf.momentum = 0.9
    conf.weight_decay = 0.0005
    conf.max_iter = 50000
    conf.snapshot_iter = 10000
    conf.display = 250
    conf.do_test = False
    
    # sgd parameters
    conf.lr_policy = 'poly'
    conf.lr_steps = None
    conf.lr_target = conf.lr * 0.00001
    
    # random
    conf.rng_seed = 2
    conf.cuda_seed = 2
    
    # misc network
    conf.image_means = [0.353, 0.389, 0.440]
    conf.image_stds = [0.219, 0.241, 0.284]
    conf.feat_stride = 16
    
    conf.has_3d = True

    # ----------------------------------------
    #  image sampling and datasets
    # ----------------------------------------

    # scale sampling  
    conf.test_scale = 512
    # conf.crop_size = [512, 1760]
    conf.crop_size = [512, 768]
    conf.mirror_prob = 0.50
    conf.distort_prob = -1
    
    # datasets
    conf.dataset_test = 'waymo_split'
    conf.datasets_train = [{'name': 'waymo_split', 'anno_fmt': 'waymo_det', 'im_ext': '.png', 'scale': 1}]
    conf.use_3d_for_2d = True
    conf.dataset_type = 'WaymoDataset'
    conf.camera = 0
    conf.camera_str = '_front'
    
    # percent expected height ranges based on test_scale
    # used for anchor selection 
    conf.percent_anc_h = [0.0625, 0.75]
    
    # labels settings
    conf.min_gt_h = conf.test_scale*conf.percent_anc_h[0]
    conf.max_gt_h = conf.test_scale*conf.percent_anc_h[1]
    conf.min_gt_vis = 0.65
    conf.ilbls = ['Van', 'ignore']
    conf.lbls = ['Car', 'Pedestrian', 'Cyclist']
    
    # ----------------------------------------
    #  detection sampling
    # ----------------------------------------
    
    # detection sampling
    conf.batch_size = 2
    conf.fg_image_ratio = 1.0
    conf.box_samples = 0.20
    conf.fg_fraction = 0.20
    conf.bg_thresh_lo = 0
    conf.bg_thresh_hi = 0.5
    conf.fg_thresh = 0.5
    conf.ign_thresh = 0.5
    conf.best_thresh = 0.35

    # ----------------------------------------
    #  inference and testing
    # ----------------------------------------

    # nms
    conf.nms_topN_pre = 3000
    conf.nms_topN_post = 40
    conf.nms_thres = 0.4
    conf.clip_boxes = False

    conf.test_protocol = 'waymo'
    conf.test_db = 'waymo'
    conf.test_min_h = 0
    conf.min_det_scales = [0, 0]

    # ----------------------------------------
    #  anchor settings
    # ----------------------------------------
    
    # clustering settings
    conf.cluster_anchors = 0
    conf.even_anchors = 0
    conf.expand_anchors = 0
                             
    conf.anchors = None

    conf.bbox_means = None
    conf.bbox_stds = None
    
    # initialize anchors
    base = (conf.max_gt_h / conf.min_gt_h) ** (1 / (12 - 1))
    conf.anchor_scales = np.array([conf.min_gt_h * (base ** i) for i in range(0, 12)])
    conf.anchor_ratios = np.array([0.5, 1.0, 1.5])
    
    # loss logic
    conf.hard_negatives = True
    conf.focal_loss = 0 #0
    conf.cls_2d_lambda = 1
    conf.iou_2d_lambda = 1
    conf.bbox_2d_lambda = 0
    conf.bbox_3d_lambda = 1
    conf.bbox_3d_proj_lambda = 0
    
    conf.hill_climbing = True
    
    # visdom
    conf.visdom_port = 8100

    # warmup or main
    conf.is_main = False
    
    # use simulation
    conf.scene_level_simulation = False
    conf.RoI_level_simulation = False
    conf.Loss_roi_lambda = 1
    conf.Loss_scene_lambda = 1

    return conf

