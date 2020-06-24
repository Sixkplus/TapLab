#!/usr/bin/env python3
# encoding: utf-8
from __future__ import division
import os.path as osp
import sys, os, glob, time
import argparse
from tqdm import tqdm

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




def fast_hist(preds, gts, num_classes = config.num_classes):
    '''
    Compute the confusion matrix in a hist form ([num_class, num_class] <-> [num_class**2, 1])

    Args:
        preds:  prediction tensor [n,c,h,w]
        gts:    ground truth tensor [n,h,w]
    
    Return:
        valid_labels: number of valid labels
        corrects:     number of TP
        cur_hist:     current histogram (confusion matrix)
    '''

    preds = preds.argmax(1).view(-1).cpu().numpy().astype(np.uint8)
    gts = gts.view(-1).cpu().numpy().astype(np.uint8)

    assert (preds.shape == gts.shape)
    k = (gts >= 0) & (gts < num_classes)
    valid_labels = np.sum(k)
    corrects = np.sum((preds[k] == gts[k]))
    cur_hist = np.bincount(num_classes * gts[k].astype(int) + preds[k].astype(int),
                       minlength=num_classes ** 2).reshape(num_classes,
                                                    num_classes)

    return valid_labels, corrects, cur_hist



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
    feat_size = (h//2, w//2)
    block_size = (h//4, w//4)

    val_folder = "datasets/cityscapes/input/val"
    val_list = glob.glob(os.path.join(val_folder, "*.png"))

    # model
    # val_model = Mobile_Light_FPN(config.num_classes, is_training=False,
    #             criterion=None)
    
    # val_model = Res18_Light_FPN(config.num_classes, is_training=False,
    #              criterion=None)
    
    val_model = BiSeNet(config.num_classes, False, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_model.to(device)
    
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

    with torch.no_grad():
        val_model.eval()

        for idx in trange(len(val_list)):
            img_name = val_list[idx]
            img = cv2.imread(img_name)
            img = img[:,:,::-1]
            imgs = normalize(img, config.image_mean, config.image_std).reshape(1,h,w,3).transpose(0,3,1,2)
            imgs = torch.Tensor(imgs).cuda()

            torch.cuda.synchronize()
            start_time = time.time()
            # network start
            feature = val_model(imgs)
            # network finish
            torch.cuda.synchronize()
            print("Process time: ", time.time()-start_time)
            output = interp(feature).cpu().numpy().transpose(0,2,3,1)
            seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
            

            output_im = seg_pred[0]
            result_id = decode_ids(output_im, [h, w], 19)

            result_color = decode_labels(output_im, [h, w], 19)
            
            if not os.path.exists('Test'):
                os.makedirs('Test')
            if not os.path.exists('Test_id'):
                os.makedirs('Test_id')
            
            cv2.imwrite('Test/' + img_name.split("/")[-1]+'_color.png', cv2.cvtColor(np.uint8(result_color), cv2.COLOR_RGB2BGR))
            #cv2.imwrite(os.path.join('./Test', imgName.split(test_dir)[1].replace('.png', '_ab.png')), np.uint8(result_alpha))
            cv2.imwrite('Test_id/' + img_name.split("/")[-1]+'_id.png', result_id)


        



