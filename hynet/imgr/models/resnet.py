import warnings

import torch
from torch import nn
import torch.nn.functional as F

from hynet.layers.batchnorm_wobias import BatchNorm2d # BatchNorm2d = nn.BatchNorm2d
from hynet.imgr.models.brew_module import BrewModel, BrewModuleList

from torchvision.models.resnet import conv1x1, conv3x3

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.bn1 = norm_layer(width)
            self.bn2 = norm_layer(width)
            self.bn3 = norm_layer(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.norm_layer is not None:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if self._norm_layer is not None:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if self._norm_layer is not None:
            if zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer is None:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride)
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self._norm_layer is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

class EnDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 batch_norm=False,
                 bias=False,
                 model_type='wrn50_2'):
        super(EnDecoder, self).__init__()

        if batch_norm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None
        if model_type == 'wrn50_2':
            self.endecoder = ResNet(Bottleneck, [3, 4,  6,  3], num_classes, norm_layer=norm_layer)
        elif model_type == 'wrn40_4':
            from hynet.imgr.models.temp import WideResNet
            self.endecoder = WideResNet(40, num_classes, widen_factor=4, cfg=[16, 32, 64])
        elif model_type == 'wrn28_10':
            from hynet.imgr.models.temp import WideResNet
            self.endecoder = WideResNet(28, num_classes, widen_factor=10, cfg=[16, 32, 64])
        else:
            raise AttributeError("This model type is not supported!!")

        self.focused_layer = self.endecoder.conv1

        # check network wrong classification case
        def all_zero_hook(self, input, result):
            if isinstance(result, tuple):
                res = result[0]
            else:
                res = result
            aggregate = res.abs().flatten(start_dim=1).sum(-1)
            flag = (aggregate > 0).float().mean()
            if flag != 1.0:
                warnings.warn("{} layer has all zero value : {}".format(self, flag))
        for m in self.endecoder.named_modules():
            m[1].register_forward_hook(all_zero_hook)

    def forward(self, x, save_grad=False):
        return self.endecoder.forward(x)
