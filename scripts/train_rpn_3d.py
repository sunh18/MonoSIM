# -----------------------------------------
# python modules
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os

import math
from torch import det
from global_var import globalvar as glo
from simulation import feature_simulation as fs

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.core import *
from lib.loss.rpn_3d import *
from lib.waymo_imdb_util import WaymoDataset
from lib.kitti_imdb_util import KittiDataset 

__all__ = {
    'WaymoDataset': WaymoDataset,
    'KittiDataset': KittiDataset
}

def main(argv):
    # -----------------------------------------
    # parse arguments
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore=', 'output=', 'gpu_id='])

    # defaults
    conf_name = None
    restore = None
    # output_version = "2_warmup"
    output_version = None
    gpu_id = None

    # read opts
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = int(arg)
        if opt in ('--output'): output_version = arg
        if opt in ('--gpu_id'): gpu_id = arg

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # required opt
    if conf_name is None:
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # basic setup
    # -----------------------------------------
    conf = init_config(conf_name)
    paths = init_training_paths(conf_name, name="_" + output_version)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)
    vis = init_visdom(conf_name, conf.visdom_port)

    # defaults
    start_iter = 0
    tracker = edict()
    iterator = None
    has_visdom = vis is not None
    has_visdom = False

    dataset = __all__[conf.dataset_type](conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)


    # -----------------------------------------
    # store config
    # -----------------------------------------

    # store configuration
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # show configuration
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # network and loss
    # -----------------------------------------

    # training network
    rpn_net, optimizer = init_training_model(conf, paths.output)

    # setup loss
    criterion_det = RPN_3D_loss(conf)

    # custom pretrained network
    if 'pretrained' in conf:

        load_weights(rpn_net, conf.pretrained)

    # resume training
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    optimizer.zero_grad()

    # joint pc detector feature path
    glo.set_value('pc_scene_feature_path',os.getcwd()+'/data/pvrcnn_material/render_scene_feature')
    glo.set_value('pc_roi_feature_path',os.getcwd()+'/data/pvrcnn_material/render_roi_feature')

    start_time = time()

    # -----------------------------------------
    # train
    # -----------------------------------------
    train_idx=[]
    for line in open(os.getcwd()+'/data/waymo_split/train_front.txt',"r"):
        line = line[:-1]
        train_idx.append(line)

    Simulation = fs.feature_simulation()

    for iteration in range(start_iter, conf.max_iter):

        torch.cuda.empty_cache()
        
        # next iteration
        iterator, images, imobjs = next_iteration(dataset.loader, iterator)
        
        # save real ids in a batch
        batch_im_id = []
        for im in imobjs:
            batch_im_id.append(train_idx[int(im['id'])])
        glo.set_value('batch_im_id',batch_im_id)

        #  learning rate
        adjust_lr(conf, optimizer, iteration)

        # forward
        cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(images)

        # loss
        det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

        if math.isnan(det_loss):
            print(batch_im_id)

        Loss_scene = 0
        Loss_roi = 0
        if conf.is_main:
            # scene level simulation
            if conf.scene_level_simulation:
                Loss_scene = conf.Loss_scene_lambda * Simulation.scene_simulation_by_KL() / len(batch_im_id)
            # roi level simulation
            if conf.RoI_level_simulation:
                Loss_roi = conf.Loss_roi_lambda * Simulation.RoI_simulation_by_KL() / len(batch_im_id)
                
        total_loss = det_loss + Loss_scene + Loss_roi
        det_stats.append({'name': 'Loss_scene', 'val': Loss_scene, 'format': '{:0.4f}', 'group': 'loss'})
        det_stats.append({'name': 'Loss_roi', 'val': Loss_roi, 'format': '{:0.4f}', 'group': 'loss'})
        stats = det_stats

        # backprop
        if total_loss > 0:

            total_loss.backward()

            # batch skip, simulates larger batches by skipping gradient step
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                optimizer.step()
                optimizer.zero_grad()

        # keep track of stats
        compute_stats(tracker, stats)

        # -----------------------------------------
        # display
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:

            # log results
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # display results
            if has_visdom:
                display_stats(vis, tracker, iteration, start_time, start_iter, conf.max_iter, conf_name, pretty)

            # reset tracker
            tracker = edict()

        # -----------------------------------------
        # test network
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:

            # store checkpoint
            save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))
            if conf.do_test and ((iteration + 1) % conf.test_iter) == 0:

                # eval mode
                rpn_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'waymo':

                    # delete and re-make
                    results_path = os.path.join(results_path, 'data')
                    #mkdir_if_missing(results_path, delete_if_exist=True)

                    #test_waymo_3d(conf.dataset_test, rpn_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)


# run from command line
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])
