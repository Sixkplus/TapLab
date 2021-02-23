# The script aims to obtain the compressed information
# And perform video segmentation

import os, glob, time, argparse
from skimage.io import imread, imsave
import cv2, random, torch
import torch.nn as nn
import numpy as np
from torch.backends import cudnn

# Model-related
from config import config
from network import BiSeNet
from mobile_light_fpn import Mobile_Light_FPN, Res18_Light_FPN
from utils.visualize import decode_color, de_nomalize, decode_labels, decode_ids
from utils.img_utils import normalize

# For decoding 
from coviar import load
from coviar import get_num_frames

GOP_FRAMES_NUM = 12
video_name = './stuttgart_00.avi'
path_output = './stuttgart_00/output/'
if not os.path.exists(path_output):
    os.makedirs(path_output)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='./cityscapes-bisenet-R18.pth', type=str, help='The path of checkpoint.')
    parser.add_argument('--rgc', default=False, action='store_true', help='Use the RGC module.')
    parser.add_argument('--rgfs', default=False, action='store_true', help='Use the RGFS module.')
    args = parser.parse_args()

    # -----------------------------------------------------------
    # ----------------------Params Config------------------------
    # -----------------------------------------------------------
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
    NUM_FRAMES = get_num_frames(video_name)
    print(NUM_FRAMES)
    # The index of GOP
    curGopIdx = 0
    for curGopIdx in range(max(NUM_FRAMES // GOP_FRAMES_NUM, 1)):
        block_scores = np.zeros(len(block_index_w)*len(block_index_h))
        cum_res = 0
        for innerGopIdx in range(GOP_FRAMES_NUM):
            curFrameIdx = curGopIdx * GOP_FRAMES_NUM + innerGopIdx
            
            rgbFrame = load(video_name, curGopIdx, innerGopIdx, 0, True)
            img = rgbFrame[:,:,::-1].astype(np.uint8)
            img = normalize(img, config.image_mean, config.image_std).reshape(1,h,w,3).transpose(0,3,1,2)
            img = torch.Tensor(img).cuda(non_blocking=True)
            
            # I-frame
            if innerGopIdx % GOP_FRAMES_NUM == 0:
                print('Frame '+str(curFrameIdx)+": I-frame")
                # Key-frame: CNN
                with torch.no_grad():
                    feature = val_model(img)

            # P-frame
            else:
                # Load motion vector
                mvCont_origin = load(video_name, curGopIdx, innerGopIdx, 1, False)
                mvCont = mvCont_origin.reshape(1, input_size[0], input_size[1], 2)
                flow[0,:,:,0] = cv2.resize(np.float32(mvCont[0,:,:,1]), (0,0), fx=feat_size[1]/input_size[1], fy=feat_size[0]/input_size[0], interpolation = cv2.INTER_LINEAR)*feat_size[0]/input_size[0]
                flow[0,:,:,1] = cv2.resize(np.float32(mvCont[0,:,:,0]), (0,0), fx=feat_size[1]/input_size[1], fy=feat_size[0]/input_size[0], interpolation = cv2.INTER_LINEAR)*feat_size[1]/input_size[1]
                
                if args.rgc or args.rgfs:
                    # Load residual
                    resCont = load(video_name, curGopIdx, innerGopIdx, 2, False)
                    res = abs(resCont).astype(np.float32)
                    res = np.sum(res, axis=2, keepdims=True)
                    cum_res += res.sum()
                    
                    # RGFS
                    if args.rgfs and (res.sum() > 36000000 or cum_res > 100000000):
                        print('Frame '+str(curFrameIdx)+": P-frame, RGFS")
                        with torch.no_grad():
                            feature = val_model(img)
                        cum_res = 0
                        block_scores = np.zeros(len(block_index_w)*len(block_index_h))
                    
                    elif args.rgc:
                        print('Frame '+str(curFrameIdx)+": P-frame, FFW+RGC")
                        # FFW
                        with torch.no_grad():
                            feature = warp(feature, grid, flow)
                        
                        # RGC
                        if args.rgc:
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
                            
                            input_block = img[:,:,max_h[0]:max_h[1], max_w[0]:max_w[1]]
                            with torch.no_grad():
                                block_feature = val_model(input_block)

                            # linear combination
                            feature[:, :, max_h[0]:max_h[1], max_w[0]:max_w[1]] = block_feature * 0.6 + feature[:, :, max_h[0]:max_h[1], max_w[0]:max_w[1]]*0.4
                        
                    else:
                        print('Frame '+str(curFrameIdx)+": P-frame, FFW")
                        # FFW
                        with torch.no_grad():
                            feature = warp(feature, grid, flow)
                else:
                    print('Frame '+str(curFrameIdx)+": P-frame, FFW")
                    # FFW
                    with torch.no_grad():
                        feature = warp(feature, grid, flow)

            output = interp(feature).cpu().numpy().transpose(0,2,3,1)
            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            seg_pred = seg_pred[0]
            
            # result_id = decode_ids(seg_pred, [1024, 2048], 19)
            result_color = decode_labels(seg_pred, [1024, 2048], 19)
            result_alpha = 0.5 * rgbFrame + 0.5 * result_color
            imsave(path_output+'frame'+str(curFrameIdx)+'.png', np.uint8(result_alpha))

if __name__ == "__main__":
    main()