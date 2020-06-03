"""MobileNet"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileNet', 'get_mobilenet', 'mobilenet1_0', 'mobilenet0_75', 'mobilenet0_5', 'mobilenet0_25']


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _DWConvBNReLU(nn.Module):
    def __init__(self, in_channels, dw_channels, out_channels, stride, dilation, norm_layer=nn.BatchNorm2d):
        super(_DWConvBNReLU, self).__init__()
        self.dwconv = nn.Sequential(
            _ConvBNReLU(in_channels, dw_channels, 3, stride, padding=dilation, dilation=dilation, groups=dw_channels, norm_layer=norm_layer),
            _ConvBNReLU(dw_channels, out_channels, 1, padding=0, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.dwconv(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNet, self).__init__()
        layer1_setting = [
            # dwc, c, n, s
            [64, 1, 1]]
        layer2_setting = [
            [128, 2, 2]]
        layer3_setting = [
            [256, 2, 2]]
        layer4_setting = [
            [512, 6, 2]]
        layer5_setting = [
            [1024, 2, 2]]
        self.in_channels = int(32 * width_mult) if width_mult > 1.0 else 32
        self.conv1 = _ConvBNReLU(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)

        # building layers
        self.layer1 = self._make_layer(_DWConvBNReLU, layer1_setting, width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(_DWConvBNReLU, layer2_setting, width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(_DWConvBNReLU, layer3_setting, width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(_DWConvBNReLU, layer4_setting, width_mult,
                                           dilation=2, norm_layer=norm_layer)
            self.layer5 = self._make_layer(_DWConvBNReLU, layer5_setting, width_mult,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(_DWConvBNReLU, layer4_setting, width_mult, norm_layer=norm_layer)
            self.layer5 = self._make_layer(_DWConvBNReLU, layer5_setting, width_mult, norm_layer=norm_layer)

        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(int(1024 * width_mult), num_classes, 1))

        # self._init_weights()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for c, n, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            layers.append(
                block(self.in_channels, out_channels // 2, out_channels, stride, dilation, norm_layer=norm_layer))
            self.in_channels = out_channels
            for _ in range(n - 1):
                layers.append(block(self.in_channels, out_channels, out_channels, 1, 1, norm_layer=norm_layer))
                self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # dict of layers
        layers = {}

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        layers["layer2"] = x
        x = self.layer3(x)
        layers["layer3"] = x
        x = self.layer4(x)
        layers["layer4"] = x
        x = self.layer5(x)
        layers["layer5"] = x
        # x = self.classifier(x)
        # x = x.view(x.size(0), x.size(1))
        return layers

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_mobilenet(width_mult=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = MobileNet(width_mult=width_mult, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def mobilenet1_0(**kwargs):
    return get_mobilenet(1.0, **kwargs)


def mobilenet0_75(**kwargs):
    return get_mobilenet(0.75, **kwargs)


def mobilenet0_5(**kwargs):
    return get_mobilenet(0.5, **kwargs)


def mobilenet0_25(**kwargs):
    return get_mobilenet(0.25, **kwargs)


if __name__ == '__main__':
    model = mobilenet1_0()
    parameters = model.state_dict()