# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os
from global_var import globalvar as glo

glo._init()

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.waymo_imdb_util import *
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

base_path = os.getcwd()

model_base_path = '/nas_data/sunhan/MonoSIM_m3drpn_waymo/pretrained_ckpt/main_pkl_folder'
model_name = 'final_model_pkl'
weights_path = model_base_path + '/' + model_name
conf_path = model_base_path + '/conf.pkl'
# label_path = base_path + '/data/waymo_split/validation_front/label_0'

cam_num = 0
cam_train_view = "front"
cam_test_num = 0
cam_test_view = "front"

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

data_path = os.path.join(os.getcwd(), 'data')
results_path = os.path.join('output', 'eval/', model_name, 'data')
save_path = os.path.join('output', 'eval/', model_name, 'eval.txt')

# make directory
mkdir_if_missing(results_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + conf.model).build(conf)

# load weights
load_weights(net, weights_path, remove_module=True)

# switch modes for evaluation
net.eval()

print(pretty_print('conf', conf))

# -----------------------------------------
# generate waymo test data
# -----------------------------------------

test_waymo_3d(conf.dataset_test, net, conf, results_path, data_path, use_log=False)

# -----------------------------------------
# evaluate generated test data
# -----------------------------------------
# from eval.evaluate import *
# evaluate(label_path, results_path, save_path)
