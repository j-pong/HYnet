import torch
from torch import nn
import torch.nn.functional as F

from hynet.layers.brew import BrewModel, BrewModuleList

def make_layers(in_channels , cfg, batch_norm=False, bias=False):
    layers = []
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    return layers

class EnDecoder(BrewModel):

    def __init__(self,
                 in_channels,
                 num_classes,
                 bias=False):
        super(EnDecoder, self).__init__()

        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.encoder = BrewModuleList(
            make_layers(in_channels, self.cfg, bias=bias)
        )
        self.img_size = [2, 2]
        self.out_channels = 512
        self.decoder = BrewModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(self.out_channels * self.img_size[0] * self.img_size[1], 4096, bias=bias),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=bias),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=bias)
        ])

        self._initialization(self.encoder)
        self._initialization(self.decoder)
    
    def _initialization(self, mlist):
        for idx, m in enumerate(mlist):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)