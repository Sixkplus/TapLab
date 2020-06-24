#!/usr/bin/env python3
# encoding: utf-8
from __future__ import division
import os.path as osp
import sys
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
from utils.visualize import decode_color, de_nomalize


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

    # model
    # val_model = Mobile_Light_FPN(config.num_classes, is_training=False,
    #             criterion=None)
    
    val_model = Res18_Light_FPN(config.num_classes, is_training=False,
                 criterion=None)
    
    val_model = BiSeNet(config.num_classes, False, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_model.to(device)

    # data loader
    # batch size = 4
    eval_loader, eval_sampler = get_eval_loader(None, Cityscapes, 4)
    
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

    # hist
    hist = np.zeros((config.num_classes, config.num_classes))
    valid_labels = 0
    corrects = 0

    with torch.no_grad():
        val_model.eval()
        eval_dataloader = iter(eval_loader)

        for _ in trange(eval_dataloader.__len__()):
            eval_batch = eval_dataloader.next()
            imgs = eval_batch['data']
            gts = eval_batch['label']

            imgs = imgs.cuda(non_blocking=True)

            preds = val_model(imgs)
            # change prediction to color for visualization
            # preds_color = decode_color(colors, preds, config.num_classes)
            cur_valid_labels, cur_corrects, cur_hist = fast_hist(preds, gts)
            corrects += cur_corrects
            valid_labels += cur_valid_labels
            hist += cur_hist
    
    # mIoU
    #hist = hist.reshape((config.num_classes, config.num_classes))

    iou = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
    print("Accuracy: {:.2%}".format(corrects/valid_labels))
    #print("IoU: ", iou)
    print("{:<20s}{:>8s}".format("Class ", "IoU"))
    for i in range(config.num_classes):
        print("{:<20s}{:>8.2%}".format(class_names[i], iou[i]))
    print("------------------------------------\n")
    print("Total mIoU: {:.2%}".format(iou.mean()))


        



