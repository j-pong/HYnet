import torch
from torch import nn
import torch.nn.functional as F

from hynet.layers.brew_layer import linear_linear, linear_conv2d, linear_maxpool2d, calculate_ratio


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

class EnDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes):
        super().__init__()

        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.encoder = nn.ModuleList(
            make_layers(in_channels, self.cfg)
        )

        self.img_size = [2, 2]
        self.out_channels = 512

        self.decoder = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(self.out_channels * self.img_size[0] * self.img_size[1], 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
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
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def backward_linear(self, x, mlist, ratio, b_hat=None):
        rats = []
        max_len = mlist.__len__()
        for idx in range(max_len):
            m = mlist.__getitem__(max_len - idx - 1)
            # print(x.size(), m)
            if isinstance(m, nn.Conv2d):
                if len(rats) == 0:
                    raise NotImplementedError
                else:
                    rat = 1.0
                    for r in rats:
                        rat = r * rat
                    rats = []
                x = linear_conv2d(x, m, rat)
            elif isinstance(m, nn.Linear):
                if len(rats) == 0:
                    rat = None
                else:
                    rat = 1.0
                    for r in rats:
                        rat = r * rat
                    rats = []
                x = linear_linear(x, m, rat)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                rats.append(ratio.pop())
            elif isinstance(m, nn.MaxPool2d):
                rat = ratio.pop()
                x = linear_maxpool2d(x, m, rat)
            elif isinstance(m, nn.Flatten):
                pass
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError
            else:
                raise NotImplementedError
        assert len(ratio) == 0

        return x

    def forward_linear_impl(self, x, mlist, ratio):
        for m in mlist:
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Flatten)):
                x = m(x)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                rat = ratio.pop(0)
                x = x * rat
            elif isinstance(m, nn.MaxPool2d):
                rat = ratio.pop(0)
                x, _ = m(x)
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError
            else:
                raise NotImplementedError

        return x

    def forward_impl(self, x, mlist, ratio):
        for m in mlist:
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Flatten)):
                x = m(x)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                x, rat = calculate_ratio(x, m, mode='grad', training=self.training)
                ratio.append(rat)
            elif isinstance(m, nn.MaxPool2d):
                x, rat = m(x)
                ratio.append(rat)
            else:
                raise NotImplementedError
            
        return x, ratio

    def forward_linear(self, x, ratio):
        x = self.forward_linear_impl(x, self.encoder, ratio)
        x = self.forward_linear_impl(x, self.decoder, ratio)
        assert len(ratio) == 0

        return x

    def forward(self, x):
        ratio = []

        x, ratio = self.forward_impl(x, self.encoder, ratio)
        ratio_split_idx = len(ratio)
        x, ratio = self.forward_impl(x, self.decoder, ratio)
        
        return x, ratio, ratio_split_idx