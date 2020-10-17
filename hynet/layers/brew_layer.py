import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

@torch.enable_grad()
def grad_activation(x, module, mode='hard', training=True):
    if training or mode == 'hard':
        x_base = x
    else:
        # validation mode has no backward graph for autograd 
        # Thus, we force variable to make graph locally 
        x_base = torch.autograd.Variable(x)
        x_base.requires_grad = True
    x = module(x_base)

    if mode == 'hard':
        # approximate grad of activation f(x)/x = f'(0)
        dfdx = x / x_base
        # prevent inf or nan case
        mask = (x_base == 0)
        dfdx[mask] = 0.0
        epsil = 0.0
    elif mode == 'soft':
        # checkout inplace option for accuratly gradient
        if module.inplace is not None:
            assert module.inplace is False
        # caculate grad via auto grad respect to x_base
        dfdx = torch.autograd.grad(x.sum(), x_base, retain_graph=True, create_graph=True)
        dfdx = dfdx[0].data
        epsil = x - dfdx * x_base
        epsil = epsil.detach()

    return x, dfdx.detach(), epsil

def linear_linear(x, m, r=None):
    if isinstance(m, nn.Linear):
        w = m.weight
        b = m.bias
        # if r is not None:
        #     w_ = r.unsqueeze(2) * w.unsqueeze(0)
        # else:
        #     w_ = w
        # x_hat = torch.matmul(x.unsqueeze(1), w_).squeeze(1)

        if r is not None:
            x = x * r
        x = F.linear(x.unsqueeze(1), w.t()).squeeze(1)
        # Todo(j-pong): check this line for equal to original
        # print(F.mse_loss(x, x_hat))
    else:                
        raise AttributeError("This module is not approprate to this function.")
    
    return x

def linear_conv2d(x, m, r=None):
    if isinstance(m, nn.Conv2d):
        w = m.weight
        b = m.bias 
        
        if r is not None:
            x = x * r
        x = F.conv_transpose2d(x, w, stride=m.stride)
        pads = m.padding
        pad_h, pad_w = pads 
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