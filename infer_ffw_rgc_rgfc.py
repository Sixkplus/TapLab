#!/usr/bin/env python3
# encoding: utf-8
from __future__ import division
import os.path as osp
import sys, os, time
import argparse
from tqdm import tqdm
from glob import glob
from skimage.io import imread

import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
import random
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from tqdm import trange
#from tensorboardX import SummaryWriter

# from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader import get_train_loader, get_eval_loader
from network import BiSeNet
from mobile_light_fpn import Mobile_Light_FPN, Res18_Light_FPN
from datasets import Cityscapes
from datasets.cityscapes import colors, class_names

from utils.init_func import init_weight, group_weight
from utils.pyt_utils import all_reduce_tensor, link_file
from utils.visualize import decode_color, de_nomalize, decode_labels, decode_ids
from utils.img_utils import normalize


# random seeds
seed = config.seed
torch.manual_seed(seed) # cpu
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(seed) #gpu
random.seed(seed)
np.random.seed(seed)
cudnn.benchmark = True


def warp(x, grid, flow):
    '''
    x: Tensor (n, c, h, w)

    grid: Tensor (n,H,W,2)
    flow: Tensor (n,H,W,2)
    '''
    h, w = x.shape[2], x.shape[3]
    H, W = grid.shape[1], grid.shape[2]
    flow = torch.from_numpy(flow)
    flow = flow.cuda()
    grid = grid.float() - flow.float()

    grid[:,:,:,0], grid[:,:,:,1] = ((grid[:,:,:,1])/W*2 - 1), ((grid[:,:,:,0])/H*2 - 1)
    
    x_next = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='zeros')

    return x_next


def grid_gen(shape):
    '''
    shape: n, H, W

    return: grids: grids[i][j] = [i, j]
    '''
    n, height, width = shape
    h_grid = torch.arange(0, height).cuda()
    w_grid = torch.arange(0, width).cuda()
    h_grid = h_grid.repeat(width, 1).permute(1,0) 
    w_grid = w_grid.repeat(height,1)              
    grid = torch.stack((h_grid,w_grid),0).permute(1,2,0).repeat(n,1,1,1).reshape(n, height, width, 2)
    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()

    h, w = 1024, 2048
    input_size = (1024, 2048)
    feat_size = (h, w)
    block_size = (h//2, w//4)

    grid = grid_gen((1,feat_size[0], feat_size[1]))

    # model
    # val_model = Mobile_Light_FPN(config.num_classes, is_training=False,
    #             criterion=None)
    
    # val_model = Res18_Light_FPN(config.num_classes, is_training=False,
    #              criterion=None)
    
    val_model = BiSeNet(config.num_classes, False, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_model.to(device)
    val_model.eval()
    
    # load from checkpoint
    try:
        ckpt_dir = osp.join(config.snapshot_dir,'epoch-%s.pth' % args.epochs)
        ckpt_dict = torch.load(ckpt_dir)()
    except:
        ckpt_dir = args.epochs
        
        if ckpt_dir == "./cityscapes-bisenet-R18.pth":
            ckpt_dict = torch.load(ckpt_dir)
            ckpt_dict = ckpt_dict['model']
        else:
            ckpt_dict = torch.load(ckpt_dir)()

    val_model.load_state_dict( { key.replace("module.", ""):ckpt_dict[key] for key in ckpt_dict.keys() } )

    # upsample
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    
    np.random.seed(5)

    val_folders = ['frankfurt', 'lindau', 'munster']

    block_index_w = [ [i*block_size[1]//2, i*block_size[1]//2 + block_size[1]] for i in range(input_size[1]//block_size[1] * 2 -1)]
    block_index_h = [ [i*block_size[0]//2, i*block_size[0]//2 + block_size[0]] for i in range(input_size[0]//block_size[0] * 2 -1)]
    for VAL_FOLDER in val_folders:
    #VAL_FOLDER = 'frankfurt'
        PATH_MV = 'val_12/' + VAL_FOLDER[0]
        PATH_IMG = 'val_12/' + VAL_FOLDER

        img_names = glob(PATH_IMG+'/*')
        mv_dir_names = glob(PATH_MV+ '/*')

        img_names.sort()
        mv_dir_names.sort()

        flow = np.zeros([1, feat_size[0], feat_size[1], 2])
        for ind in range(len(mv_dir_names)):
            cur_dir_name = mv_dir_names[ind]

            rand_num = np.random.randint(8, 20)
            cur_img_name_start = ind * 30 + rand_num
            cur_img_name_end = ind * 30 + 19

            st = time.time()
            img = cv2.imread(img_names[cur_img_name_start]).astype(np.float32)
            img = img[:,:,::-1]
            img = normalize(img, config.image_mean, config.image_std).reshape(1,h,w,3).transpose(0,3,1,2)
            img = torch.Tensor(img).cuda(non_blocking=True)
            torch.cuda.synchronize()
            time_f = time.time()
            with torch.no_grad():
                feature = val_model(img)
            
            torch.cuda.synchronize()
            print("        Feature time: ", time.time() - time_f)
            # key frame
            if(cur_img_name_start != cur_img_name_end):
                block_scores = np.zeros(len(block_index_w)*len(block_index_h))
                cum_res = 0
                for temp_ind in range(rand_num+1, 19+1):
                    img = cv2.imread(img_names[cur_img_name_start - rand_num + temp_ind]).astype(np.float32)
                    img = img[:,:,::-1]
                    img = normalize(img, config.image_mean, config.image_std).reshape(1,h,w,3).transpose(0,3,1,2)
                    img = torch.Tensor(img).cuda(non_blocking=True)
                    save_mvPng = imread(cur_dir_name + "/mv_cont" + '/frame'+ str(temp_ind - 8) + '.png' ).astype(np.int16)
                    flow_origin = np.array([ (save_mvPng[:,:,0] << 8) + (save_mvPng[:,:,1]), (save_mvPng[:,:,2] << 8) + (save_mvPng[:,:,3]) ])
                    flow_origin = np.transpose(flow_origin, [1,2,0]).reshape(1, input_size[0], input_size[1], 2)
                    flow_origin -= 2048

                    flow[0,:,:,0] = cv2.resize(np.float32(flow_origin[0,:,:,1]), (0,0), fx=feat_size[1]/input_size[1], fy=feat_size[0]/input_size[0], interpolation = cv2.INTER_LINEAR)*feat_size[0]/input_size[0]
                    flow[0,:,:,1] = cv2.resize(np.float32(flow_origin[0,:,:,0]), (0,0), fx=feat_size[1]/input_size[1], fy=feat_size[0]/input_size[0], interpolation = cv2.INTER_LINEAR)*feat_size[1]/input_size[1]
                    
                    flow_origin = abs(flow_origin.reshape(input_size[0], input_size[1], 2))
                    flow_origin = np.sum(flow_origin, axis=2, keepdims=True)
                    res = imread(cur_dir_name + "/res_cont"+ '/frame'+ str(temp_ind - 8) + '.png').astype(np.float32)
                    #res = abs(res * 2 - 256)
                    res = np.sum(res, axis=2, keepdims=True)
                    cum_res += res.sum()

                    
                    if(res.sum() > 36000000 or cum_res > 100000000):
                        torch.cuda.synchronize()
                        st1 = time.time()
                        with torch.no_grad():
                            feature = val_model(img)
                        cum_res = 0
                        torch.cuda.synchronize()
                        proc_time = time.time()-st1
                        print("RGFS time: ", proc_time)
                    else:
                        res = res + flow_origin*0.1
                        #res = np.sum(res > 200, axis=2)
                        #res = res > 25

                        block_idx = 0
                        score_max = block_scores.max()
                        score_max_idx = 0

                        for w_ in block_index_w:
                            for h_ in block_index_h:
                                w_s, w_e = w_
                                h_s, h_e = h_
                                cur_score = np.sum(res[h_s:h_e, w_s:w_e])
                                block_scores[block_idx] += cur_score
                                if block_scores[block_idx] > score_max:
                                    max_w = w_
                                    max_h = h_
                                    score_max = block_scores[block_idx]
                                    score_max_idx = block_idx
                                block_idx += 1
                        # Reset the score of current max-block to zero
                        # print("Block scores: ", block_scores)
                        block_scores[score_max_idx] = 0
                        
                        print("Current block:  ", max_h, max_w)
                        torch.cuda.synchronize()
                        st1 = time.time()
                        input_block = img[:,:,max_h[0]:max_h[1], max_w[0]:max_w[1]]
                        with torch.no_grad():
                            block_feature = val_model(input_block)
                        torch.cuda.synchronize()
                        proc_time = time.time()-st1
                        print("        Block time: ", proc_time)
                        torch.cuda.synchronize()
                        st1 = time.time()
                        with torch.no_grad():
                            feature = warp(feature, grid, flow)
                        torch.cuda.synchronize()
                        feature[:, :, max_h[0]:max_h[1], max_w[0]:max_w[1]] = block_feature * 0.6 + feature[:, :, max_h[0]:max_h[1], max_w[0]:max_w[1]]*0.4
                        proc_time = time.time()-st1
                        print("        Flow time: ", proc_time)

            proc_time = time.time() - time_f
            print("  Handling the " + cur_dir_name + "slice, time consumed: " + str(proc_time) )
            output = interp(feature).cpu().numpy().transpose(0,2,3,1)
            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            output_im = seg_pred[0]
            result_id = decode_ids(output_im, [1024, 2048], 19)

            result_color = decode_labels(output_im, [1024, 2048], 19)
            # result_color = cv2.cvtColor(output_color, cv2.COLOR_RGB2BGR)

            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # result_alpha = 0.5 * img + 0.5 * result_color
            
            if not os.path.exists('Test'):
                os.makedirs('Test')
            if not os.path.exists('Test_id'):
                os.makedirs('Test_id')
            
            cv2.imwrite('Test/' + cur_dir_name.split("/")[-1]+'_color.png', cv2.cvtColor(np.uint8(result_color), cv2.COLOR_RGB2BGR))
            #cv2.imwrite(os.path.join('./Test', imgName.split(test_dir)[1].replace('.png', '_ab.png')), np.uint8(result_alpha))
            cv2.imwrite('Test_id/' + cur_dir_name.split("/")[-1]+'_id.png', result_id)


        



