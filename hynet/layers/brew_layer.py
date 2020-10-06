import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

@torch.enable_grad()
def calculate_ratio(x, module, mode='grad', training=True):
    if training:
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
            x = x - b
        else:
            x = (x - r * b) * r
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