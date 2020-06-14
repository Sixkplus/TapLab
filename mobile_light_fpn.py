# FPN + SPP for semantic segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

from MobileNetV1 import MobileNet
from base_model import resnet18
from config import config

class PoolAndAlign(nn.Module):
    def __init__(self, in_planes, out_planes, scale, norm_layer=nn.BatchNorm2d):
        super(PoolAndAlign, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, 1, 1, bias=False)
        self.bn = norm_layer(out_planes)
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        _, _, h, w = x.shape
        x = F.adaptive_avg_pool2d(x, self.scale)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, (h,w), mode='bilinear', align_corners=True)
        return x
    



class SpatailPyramidPooling(nn.Module):
    '''
    Spatial pyramid pooling
    '''
    def __init__(self, in_planes=1024, spp_planes=512, py_planes=128,
                 out_planes=256, rates=[1,2,3,6], norm_layer=nn.BatchNorm2d):
        super(SpatailPyramidPooling, self).__init__()

        self.num_scales = len(rates)
        self.conv1 = nn.Conv2d(in_planes, spp_planes, 1, 1, bias=False)
        self.spp_layers = [PoolAndAlign(spp_planes, py_planes, rates[i], norm_layer) for i in range(len(rates))]
        self.spp_layers = nn.ModuleList(self.spp_layers)
        self.conv2 = nn.Conv2d(spp_planes+spp_planes, out_planes, 1, 1, bias=False)
        self.bn = norm_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        spatial_pyramids = [x]
        for spp_layer in self.spp_layers:
            spatial_pyramids.append(spp_layer(x))
        x = torch.cat(spatial_pyramids, 1)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class LateralConnect(nn.Module):
    '''
    Lateral Connection
    '''
    def __init__(self, in_planes, lateral_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(LateralConnect, self).__init__()
        self.conv1 = nn.Conv2d(lateral_planes, in_planes, 1, 1, bias=False)
        self.bn1 = norm_layer(in_planes)

        self.conv2 = nn.Conv2d(in_planes, out_planes, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(out_planes)

        self.relu = nn.ReLU(True)

    def forward(self, x, lateral):
        lateral = self.relu( self.bn1(self.conv1(lateral)) )
        _, _, h, w = lateral.shape
        x = F.interpolate(x, (h,w), mode='bilinear', align_corners=True)
        x = x + lateral
        return self.relu( self.bn2(self.conv2(x)) )
        


class Mobile_Light_FPN(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(Mobile_Light_FPN, self).__init__()
        
        # encoder
        self.backbone = MobileNet(norm_layer=norm_layer)

        # spp default pyramids: [1, 2, 4, 8]
        self.spp = SpatailPyramidPooling(1024, 512, 128, 256, [1,2,4,8])

        # decoder - lateral connections
        self.decoder_configs = [
            # n_features, n_laterals, n_output
            [256, 512, 128],
            [128, 256, 64],
            [64, 128, 64]
        ]

        decoder = []
        for i in range(len(self.decoder_configs)):
            in_planes, lateral_planes, out_planes = self.decoder_configs[i]
            decoder.append(LateralConnect(in_planes, lateral_planes, out_planes))
        
        # classifier
        self.conv = nn.Conv2d(self.decoder_configs[-1][-1], out_planes, 3, 1, 1)


        self.business_layer = []
        self.is_training = is_training

        self.decoder = nn.ModuleList(decoder)

        self.business_layer.append(self.backbone)
        self.business_layer.append(self.spp)
        self.business_layer.append(self.decoder)
        self.business_layer.append(self.conv)

        if is_training:
            self.criterion = criterion

    def forward(self, data, label=None):
        _, _, h, w = data.shape

        encoder_layers = self.backbone(data)
        x = self.spp(encoder_layers["layer5"])

        x = self.decoder[0](x, encoder_layers["layer4"])
        x = self.decoder[1](x, encoder_layers["layer3"])
        x = self.decoder[2](x, encoder_layers["layer2"])

        pred_out = F.interpolate(self.conv(x), size=(h,w), align_corners=True, mode="bilinear")

        if self.is_training:
            loss = self.criterion(pred_out, label)
            return loss

        return F.log_softmax(pred_out, dim=1)

class Res18_Light_FPN(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(Res18_Light_FPN, self).__init__()
        
        # encoder
        self.backbone = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=config.bn_eps,
                                     bn_momentum=config.bn_momentum,
                                     deep_stem=False, stem_width=64)
        

        # spp default pyramids: [1, 2, 4, 8]
        self.spp = SpatailPyramidPooling(512, 512, 128, 256, [1,2,4,8])

        # decoder - lateral connections
        self.decoder_configs = [
            # n_features, n_laterals, n_output
            [256, 256, 128],
            [128, 128, 64],
            [64, 64, 64]
        ]

        decoder = []
        for i in range(len(self.decoder_configs)):
            in_planes, lateral_planes, out_planes = self.decoder_configs[i]
            decoder.append(LateralConnect(in_planes, lateral_planes, out_planes))
        
        # classifier
        self.conv = nn.Conv2d(self.decoder_configs[-1][-1], out_planes, 3, 1, 1)


        self.business_layer = []
        self.is_training = is_training

        self.decoder = nn.ModuleList(decoder)

        self.business_layer.append(self.backbone)
        self.business_layer.append(self.spp)
        self.business_layer.append(self.decoder)
        self.business_layer.append(self.conv)

        if is_training:
            self.criterion = criterion

    def forward(self, data, label=None):
        _, _, h, w = data.shape

        encoder_layers = self.backbone(data)
        x = self.spp(encoder_layers[3])

        x = self.decoder[0](x, encoder_layers[2])
        x = self.decoder[1](x, encoder_layers[1])
        x = self.decoder[2](x, encoder_layers[0])

        pred_out = F.interpolate(self.conv(x), size=(h,w), align_corners=True, mode="bilinear")

        if self.is_training:
            loss = self.criterion(pred_out, label)
            return loss

        return F.log_softmax(pred_out, dim=1)