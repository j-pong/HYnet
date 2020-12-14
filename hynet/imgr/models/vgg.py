import warnings

import torch
from torch import nn
import torch.nn.functional as F

from hynet.layers.batchnorm_wobias import BatchNorm2d
from hynet.imgr.models.brew_module import BrewModel, BrewModuleList

cfgs = {
    'A':  [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B0': [64, 64], # cehck 0
    'B1': [64, 64, 'M'], # cehck 5
    'B2': [64, 64, 'M', 128, 128, 'M'], # check 7 
    'B3': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M'], # check 9
    'B4': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M'], # check 11
    'B':  [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'], # check 13
    'D':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'], # check 16
    'E':  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # check 19
}

activation = nn.ReLU

def make_layers(in_channels , cfg, batch_norm=False, bias=False):
    layers = []
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), activation()]
            else:
                layers += [conv2d, activation()]
            in_channels = v

    return layers

class EnDecoder(BrewModel):

    def __init__(self,
                 in_channels,
                 num_classes,
                 batch_norm=False,
                 bias=False,
                 model_type='B3'):
        super(EnDecoder, self).__init__()

        self.cfg = cfgs[model_type]
        wh = 32
        for i in self.cfg:
            if i == 'M':
                wh = wh / 2
        self.img_size = [int(wh), int(wh)]
        
        self.out_channels = self.cfg[-2]

        self.encoder = BrewModuleList(
            make_layers(in_channels, self.cfg, batch_norm=batch_norm, bias=bias)
        )
        self.decoder = BrewModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(self.out_channels * self.img_size[0] * self.img_size[1], 4096, bias=bias),
            activation(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=bias),
            activation(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=bias)
        ])

        self.focused_layer = self.encoder[0]

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
        for m in self.encoder.named_modules():
            m[1].register_forward_hook(all_zero_hook)
        for m in self.decoder.named_modules():
            m[1].register_forward_hook(all_zero_hook)

        # intislaization whole network module
        self._initialization(self.encoder)
        self._initialization(self.decoder)
    
    def _initialization(self, mlist):
        for idx, m in enumerate(mlist):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)