import warnings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, cfg, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, cfg, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(cfg, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, cfg, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, cfg, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, cfg, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, cfg, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, cfg=None):
        super(WideResNet, self).__init__()
        
        if cfg == None:
            cfg = [16, 32, 64]

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, cfg[0]*widen_factor, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, cfg[1]*widen_factor, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, cfg[2]*widen_factor, dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        return self.fc(out)

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1

        return [bn1, bn2, bn3]

    def get_channel_num(self):

        return self.nChannels[1:]

    def extract_feature(self, x, preReLU=False):
        out = self.conv1(x)
        feat1 = self.block1(out)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)
        out = self.relu(self.bn1(feat3))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        out = self.fc(out)

        if preReLU:
            feat1 = self.block2.layer[0].bn1(feat1)
            feat2 = self.block3.layer[0].bn1(feat2)
            feat3 = self.bn1(feat3)

        return [feat1, feat2, feat3]
    
    def ware(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        return self.fc(out).cpu().detach().numpy()

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