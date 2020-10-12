import torch
from torch import nn
import torch.nn.functional as F

from hynet.layers.brew_layer import linear_linear, linear_conv2d, linear_maxpool2d, calculate_ratio, make_layers, BrewModel

class EnDecoder(BrewModel):

    def __init__(self,
                 in_channels,
                 num_classes,
                 bias=False):
        super().__init__()

        # self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.cfg = [64, 64, 'M', 128, 128, 'M']
        self.encoder = nn.ModuleList(
            make_layers(in_channels, self.cfg)
        )
        self.img_size = [8, 8]
        self.out_channels = 128
        self.decoder = nn.ModuleList([
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