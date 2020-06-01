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
#from tensorboardX import SummaryWriter

# from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader import get_train_loader, get_eval_loader
from network import BiSeNet
from datasets import Cityscapes
from datasets.cityscapes import colors

from utils.init_func import init_weight, group_weight
from utils.pyt_utils import all_reduce_tensor, link_file
from utils.visualize import decode_color, de_nomalize
from engine.lr_policy import PolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d


parser = argparse.ArgumentParser()
args = parser.parse_args()

# tensorboard
writer = SummaryWriter("logs/cityscapes_train_log")


# random seeds
seed = config.seed
torch.manual_seed(seed) # cpu
if torch.cuda.device_count() > 0:
    torch.cuda.manual_seed_all(seed) #gpu
random.seed(seed)
np.random.seed(seed)
cudnn.benchmark = True
#torch.backends.cudnn.deterministic=True # cudnn

# num gpus
config.ngpus = torch.cuda.device_count()

# total batch size
config.batch_size = config.batch_size * config.ngpus

# iterations per epoch
config.niters_per_epoch = config.niters_per_epoch // config.ngpus

# data loader
train_loader, train_sampler = get_train_loader(None, Cityscapes, config)
eval_loader, eval_sampler = get_eval_loader(None, Cityscapes)

# config network and criterion
min_kept = int(config.batch_size // config.ngpus * config.image_height * config.image_width // 16)
criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                    min_kept=min_kept,
                                    use_weight=False)

# torch sync_bn
# BatchNorm2d = torch.nn.SyncBatchNorm
BatchNorm2d = nn.BatchNorm2d

model = BiSeNet(config.num_classes, is_training=True,
                criterion=criterion,
                pretrained_model=config.pretrained_model,
                norm_layer=BatchNorm2d)

val_model = BiSeNet(config.num_classes, is_training=False,
                criterion=criterion,
                pretrained_model=None,
                norm_layer=BatchNorm2d)

init_weight(model.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, config.bn_eps, config.bn_momentum,
            mode='fan_in', nonlinearity='relu')

# group weight and config optimizer
base_lr = config.lr
# if engine.distributed:
#     base_lr = config.lr * engine.world_size

params_list = []
params_list = group_weight(params_list, model.context_path,
                            BatchNorm2d, base_lr)
params_list = group_weight(params_list, model.spatial_path,
                            BatchNorm2d, base_lr * 10)
params_list = group_weight(params_list, model.global_context,
                            BatchNorm2d, base_lr * 10)
params_list = group_weight(params_list, model.arms,
                            BatchNorm2d, base_lr * 10)
params_list = group_weight(params_list, model.refines,
                            BatchNorm2d, base_lr * 10)
params_list = group_weight(params_list, model.heads,
                            BatchNorm2d, base_lr * 10)
params_list = group_weight(params_list, model.ffm,
                            BatchNorm2d, base_lr * 10)

optimizer = torch.optim.SGD(params_list,
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)

# config lr policy
total_iteration = config.nepochs * config.niters_per_epoch
lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model.to(device)
val_model = nn.DataParallel(val_model)
val_model.to(device)

# Add the compute graph
dataiter = iter(eval_loader)
minibatch = dataiter.next()
images = minibatch['data']
#writer.add_graph(val_model, images.to(device))
#writer.close()



for epoch in range(config.nepochs):
    model.train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                bar_format=bar_format)
    dataloader = iter(train_loader)
    running_loss = 0
    for idx in pbar:
        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch['data']
        gts = minibatch['label']

        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)

        losses = model(imgs, gts)
        loss = losses.mean()
        running_loss += loss

        current_idx = epoch * config.niters_per_epoch + idx
        lr = lr_policy.get_lr(current_idx)

        for i in range(2):
            optimizer.param_groups[i]['lr'] = lr
        for i in range(2, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * 10

        loss.backward()
        optimizer.step()
        print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                    + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.2e' % lr \
                    + ' loss=%.2f' % loss.item()

        pbar.set_description(print_str, refresh=False)

    # validation and tensorboard
    val_model.load_state_dict(model.state_dict())
    writer.add_scalar("training_loss", running_loss/config.niters_per_epoch, current_idx)
    with torch.no_grad():
        val_model.eval()
        eval_dataloader = iter(eval_loader)
        eval_batch = eval_dataloader.next()
        imgs = eval_batch['data']
        gts = eval_batch['label']

        imgs = imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)

        preds = val_model(imgs)
        # change prediction to color for visualization
        preds_color = decode_color(colors, preds, config.num_classes)

        # tensorboard visulization
        vis_imgs = torchvision.utils.make_grid(de_nomalize(imgs, config.image_mean, config.image_std))
        vis_preds = torchvision.utils.make_grid(preds_color)

        writer.add_image('Input_images', vis_imgs, current_idx)
        writer.add_image('Predictions', 255-vis_preds, current_idx)

    # save
    # the last 20 or according to the selected interval
    if (epoch > config.nepochs - 20) or (epoch % config.snapshot_iter == 0):
        #engine.save_and_link_checkpoint(config.snapshot_dir,
        #                                config.log_dir,
        #                                config.log_dir_link)
        current_epoch_checkpoint = osp.join(config.snapshot_dir, 'epoch-{}.pth'.format(epoch))
        #self.save_checkpoint(current_epoch_checkpoint)
        torch.save(model.state_dict, current_epoch_checkpoint)
        last_epoch_checkpoint = osp.join(config.snapshot_dir,
                                         'epoch-last.pth')
        link_file(current_epoch_checkpoint, last_epoch_checkpoint)
