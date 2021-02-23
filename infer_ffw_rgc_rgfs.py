#!/usr/bin/env python3
# encoding: utf-8
from __future__ import division
import os.path as osp
import sys, os, time, argparse
from glob import glob
from skimage.io import imread

import cv2, random, torch
import torch.nn as nn
import numpy as np
from torch.backends import cudnn

from config import config
from network import BiSeNet
from mobile_light_fpn import Mobile_Light_FPN, Res18_Light_FPN

from utils.visualize import decode_color, de_nomalize, decode_labels, decode_ids
from utils.img_utils import normalize


# -----------------------------------------------------------
# ----------------------Random Seeds-------------------------
# -----------------------------------------------------------
seed = config.seed
torch.manual_seed(seed) # cpu
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(seed) #gpu
random.seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

# -----------------------------------------------------------
# -------------------------Warping---------------------------
# -----------------------------------------------------------
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

# generate a coordinate map 
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
    parser.add_argument('-p', '--path', default='./cityscapes-bisenet-R18.pth', type=str, help='The path of checkpoint.')
    parser.add_argument('--rgc', default=False, action='store_true', help='Use the RGC module.')
    parser.add_argument('--rgfs', default=False, action='store_true', help='Use the RGFS module.')
    parser.add_argument('-g','--gop', type=int, default=12, help='The GOP number.')
    args = parser.parse_args()

    data_dir = './val_sequence'
    val_folders = ['frankfurt', 'lindau', 'munster']

    h, w = 1024, 2048
    input_size = (1024, 2048)
    feat_size = (h, w)
    block_size = (512, 512)

    grid = grid_gen((1,feat_size[0], feat_size[1]))
    block_index_w = [ [i*block_size[1]//2, i*block_size[1]//2 + block_size[1]] for i in range(input_size[1]//block_size[1] * 2 -1)]
    block_index_h = [ [i*block_size[0]//2, i*block_size[0]//2 + block_size[0]] for i in range(input_size[0]//block_size[0] * 2 -1)]
    flow = np.zeros([1, feat_size[0], feat_size[1], 2])

    # -----------------------------------------------------------
    # ----------------------Set the model------------------------
    # -----------------------------------------------------------

    # val_model = Mobile_Light_FPN(config.num_classes, is_training=False,
    #             criterion=None)
    
    # val_model = Res18_Light_FPN(config.num_classes, is_training=False,
    #              criterion=None)
    
    val_model = BiSeNet(config.num_classes, False, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_model.to(device)
    val_model.eval()
    
    # load from checkpoint
    ckpt_dir = args.path
    
    if ckpt_dir == "./cityscapes-bisenet-R18.pth":
        ckpt_dict = torch.load(ckpt_dir)
        ckpt_dict = ckpt_dict['model']
    else:
        ckpt_dict = torch.load(ckpt_dir)()

    val_model.load_state_dict( { key.replace("module.", ""):ckpt_dict[key] for key in ckpt_dict.keys() } )
    # upsample after prediction
    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)

    # -----------------------------------------------------------
    # ------------------------Inference--------------------------
    # -----------------------------------------------------------

    for val_folder in val_folders:
        path_mv = osp.join(data_dir, val_folder[0])
        path_img = osp.join(data_dir, val_folder)

        img_names = glob(path_img+'/*')
        mv_dir_names = glob(path_mv+ '/*')
        img_names.sort()
        mv_dir_names.sort()

        for ind in range(len(mv_dir_names)):
            cur_dir_name = mv_dir_names[ind]
            rand_num = np.random.randint(20-args.gop, 20)
            start_idx = ind * 30 + rand_num
            end_idx = ind * 30 + 19

            img = cv2.imread(img_names[start_idx]).astype(np.float32)
            img = img[:,:,::-1]
            img = normalize(img, config.image_mean, config.image_std).reshape(1,h,w,3).transpose(0,3,1,2)
            img = torch.Tensor(img).cuda(non_blocking=True)

            # Key-frame
            torch.cuda.synchronize()
            st = time.time()
            with torch.no_grad():
                feature = val_model(img)
            torch.cuda.synchronize()
            print("Key frame time: ", time.time() - st)

            # Non-key frame
            if(start_idx != end_idx):
                block_scores = np.zeros(len(block_index_w)*len(block_index_h))
                cum_res = 0
                for temp_ind in range(rand_num+1, 19+1):

                    # load rgb image and decode
                    img_origin = cv2.imread(img_names[start_idx - rand_num + temp_ind]).astype(np.float32)
                    img = img_origin[:,:,::-1]
                    img = normalize(img, config.image_mean, config.image_std).reshape(1,h,w,3).transpose(0,3,1,2)
                    img = torch.Tensor(img).cuda(non_blocking=True)

                    # load motion vector and decode
                    save_mvPng = imread(cur_dir_name + "/mv_cont" + '/frame'+ str(temp_ind - 8) + '.png' ).astype(np.int16)
                    flow_origin = np.array([ (save_mvPng[:,:,0] << 8) + (save_mvPng[:,:,1]), (save_mvPng[:,:,2] << 8) + (save_mvPng[:,:,3]) ])
                    flow_origin = np.transpose(flow_origin, [1,2,0]).reshape(1, input_size[0], input_size[1], 2)
                    flow_origin -= 2048
                    flow[0,:,:,0] = cv2.resize(np.float32(flow_origin[0,:,:,1]), (0,0), fx=feat_size[1]/input_size[1], fy=feat_size[0]/input_size[0], interpolation = cv2.INTER_LINEAR)*feat_size[0]/input_size[0]
                    flow[0,:,:,1] = cv2.resize(np.float32(flow_origin[0,:,:,0]), (0,0), fx=feat_size[1]/input_size[1], fy=feat_size[0]/input_size[0], interpolation = cv2.INTER_LINEAR)*feat_size[1]/input_size[1]
                    
                    # load residual map
                    res = imread(cur_dir_name + "/res_cont"+ '/frame'+ str(temp_ind - 8) + '.png').astype(np.float32)
                    res = abs(res * 2 - 255)
                    res = np.sum(res, axis=2, keepdims=True)
                    cum_res += res.sum()
                    
                    if args.rgfs and (res.sum() > 36000000 or cum_res > 100000000):
                        # RGFS
                        torch.cuda.synchronize()
                        st = time.time()
                        with torch.no_grad():
                            feature = val_model(img)
                        cum_res = 0
                        torch.cuda.synchronize()
                        proc_time = time.time()-st
                        print("RGFS time: ", proc_time)
                        block_scores = np.zeros(len(block_index_w)*len(block_index_h))
                    else:
                        # FFW
                        torch.cuda.synchronize()
                        st = time.time()
                        with torch.no_grad():
                            feature = warp(feature, grid, flow)
                        torch.cuda.synchronize()
                        proc_time = time.time()-st
                        print("    Flow time: ", proc_time)

                        # RGC 
                        if args.rgc:
                            # res = res[res > 20]
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
                            block_scores[score_max_idx] = 0
                            # print("Current block:  ", max_h, max_w)
                            torch.cuda.synchronize()
                            st = time.time()
                            input_block = img[:,:,max_h[0]:max_h[1], max_w[0]:max_w[1]]
                            with torch.no_grad():
                                block_feature = val_model(input_block)
                            torch.cuda.synchronize()
                            proc_time = time.time()-st
                            print("    RGC time: ", proc_time)

                            # linear combination
                            feature[:, :, max_h[0]:max_h[1], max_w[0]:max_w[1]] = block_feature * 0.6 + feature[:, :, max_h[0]:max_h[1], max_w[0]:max_w[1]]*0.4

            output = interp(feature).cpu().numpy().transpose(0,2,3,1)
            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            output_img = seg_pred[0]
            
            result_id = decode_ids(output_img, [1024, 2048], 19)
            result_color = decode_labels(output_img, [1024, 2048], 19)
            result_alpha = 0.5 * img_origin + 0.5 * result_color
            
            if not os.path.exists('Test'):
                os.makedirs('Test')
            if not os.path.exists('Test_id'):
                os.makedirs('Test_id')
            if not os.path.exists('Test_alpha'):
                os.makedirs('Test_alpha')
            
            cv2.imwrite('Test/' + cur_dir_name.split("/")[-1]+'_color.png', cv2.cvtColor(np.uint8(result_color), cv2.COLOR_RGB2BGR))
            cv2.imwrite('Test_id/' + cur_dir_name.split("/")[-1]+'_id.png', result_id)
            cv2.imwrite('Test_alpha/' + cur_dir_name.split("/")[-1]+'_ab.png', cv2.cvtColor(np.uint8(result_alpha), cv2.COLOR_RGB2BGR))
