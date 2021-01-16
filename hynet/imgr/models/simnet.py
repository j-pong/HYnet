import warnings

import torch
from torch import nn
import torch.nn.functional as F

from hynet.layers.batchnorm_wobias import BatchNorm2d # BatchNorm2d = nn.BatchNorm2d
from hynet.imgr.models.brew_module import BrewModel, BrewModuleList

activation = nn.ReLU

class EnDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes,
                 batch_norm=False,
                 bias=False,
                 model_type='simple',
                 wh=32):
        super(EnDecoder, self).__init__()
        self.img_size = [int(wh), int(wh)]
        
        self.out_channels = 3

        self.endecoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.out_channels * self.img_size[0] * self.img_size[1], 4096, bias=bias),
            activation(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=bias),
            activation(),
            nn.Dropout(),
            nn.Linear(4096, num_classes, bias=bias)
        )

        self.focused_layer = self.endecoder[0]

        # intislaization whole network module
        self._initialization(self.endecoder)
    
    def forward(self, x, save_grad=False):
        return self.endecoder(x)
    
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