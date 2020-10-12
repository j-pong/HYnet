import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

@torch.enable_grad()
def calculate_ratio(x, module, mode='ratio', training=True):
    if training or mode == 'ratio':
        x_base = x
    else:
        # validation mode has no backward graph for autograd 
        # Thus, we force variable to make graph locally 
        x_base = torch.autograd.Variable(x)
        x_base.requires_grad = True
    x = module(x_base)

    if mode == 'ratio':
        mask = (x_base == 0)
        rat = x / x_base
        rat[mask] = 0.0

    elif mode == 'grad':
        # checkout inplace option for accuratly gradient
        if module.inplace is not None:
            assert module.inplace is False
        # caculate gradient
        rat = torch.autograd.grad(x.sum(), x_base, 
                                    retain_graph=True,
                                    create_graph=True)[0]
        rat = rat.data

    return x, rat.detach()

def linear_linear(x, m, r=None):
    if isinstance(m, nn.Linear):
        w = m.weight
        b = m.bias
                    
        # w = r.unsqueeze(2) * w.unsqueeze(0)
        # x = torch.matmul(x.unsqueeze(1), w).squeeze(1)
        if r is None:
            x = x 
        else:
            x = x * r
        x = F.linear(x.unsqueeze(1), w.t()).squeeze(1)
    else:                
        raise AttributeError("This module is not approprate to this function.")
    
    return x

def linear_conv2d(x, m, r=None):
    if isinstance(m, nn.Conv2d):
        w = m.weight
        b = m.bias 
        assert b is None
        
        if r is not None:
            x = x * r
        x = F.conv_transpose2d(x, w, stride=m.stride)
        pads = m.padding
        pad_h,  pad_w = pads 
        if pad_h > 0:
            x = x[:, :, pad_h:-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, pad_w:-pad_w]
    else:
        raise AttributeError("This module is not approprate to this function.")
    
    return x

def linear_maxpool2d(x, m, r=None):
    if isinstance(m, nn.MaxPool2d):
        x = F.max_unpool2d(x, r, m.kernel_size, m.stride)
    else:
        raise AttributeError("This module is not approprate to this function.")
    
    return x

class BrewModel(nn.Module):
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
    
    def backward_linear_impl(self, x, mlist, ratio, b_hat=None):
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

    def forward_linear_impl(self, x, mlist, ratio, b_hat=None):
        # Todo(j-pong): extract bias that first apeeal and forward and add with layers
        # b_1 = m.bias
        # b_2 = rat * m(b_1) + b_r
        # this lines is solution of bias case 
        for m in mlist:
            if isinstance(m, nn.Conv2d):
                x = m(x)
            elif isinstance(m, nn.Linear):
                x = m(x)
                b = m.bias
                if b is not None:
                    if b_hat is None:
                        b_hat = b.unsqueeze(0)
                    else:
                        b_hat = m(b_hat)
                    x = x - b
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                rat = ratio.pop(0)
                x = x * rat
                if b_hat is not None:
                    b_hat = b_hat * rat
            elif isinstance(m, nn.MaxPool2d):
                rat = ratio.pop(0)
                x, _ = m(x)
            elif isinstance(m, nn.Flatten):
                x = m(x)
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError
            else:
                raise NotImplementedError

        return x, b_hat

    def forward_impl(self, x, mlist, ratio):
        for m in mlist:
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Flatten)):
                x = m(x)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                x, rat = calculate_ratio(x, m, training=self.training)
                ratio.append(rat)
            elif isinstance(m, nn.MaxPool2d):
                x, rat = m(x)
                ratio.append(rat)
            else:
                raise NotImplementedError
            
        return x

    def forward_linear(self, x, ratios):
        x, b_hat = self.forward_linear_impl(x, self.encoder, ratios[0])
        x, b_hat = self.forward_linear_impl(x, self.decoder, ratios[1], b_hat=b_hat)
        assert len(ratios[0]) == 0
        assert len(ratios[1]) == 0

        return x, b_hat

    def forward(self, x):
        ratios = [[], []]

        x = self.forward_impl(x, self.encoder, ratios[0])
        x = self.forward_impl(x, self.decoder, ratios[1])
        
        return x, ratios


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