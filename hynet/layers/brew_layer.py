import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

@torch.enable_grad()
def grad_activation(x, module, training=True, shrink=False):
    if training:
        x_base = x
    else:
        # validation mode has no backward graph for autograd 
        # Thus, we force variable to make graph locally 
        x_base = torch.autograd.Variable(x)
        x_base.requires_grad = True

    if shrink:
        x = module(x_base)[0]
    else:
        x = module(x_base)

    if shrink:
        x = module(x_base)[0]
        # checkout inplace option for accuratly gradient
        if isinstance(module, nn.ReLU):
            assert module.inplace is False
        # caculate grad via auto grad respect to x_base
        dfdx = torch.autograd.grad(x.sum(), x_base, retain_graph=True, create_graph=True)
        dfdx = dfdx[0].data

        epsil = 0.0
    else:
        x = module(x_base)
        # approximate grad of activation f(x) / x = f'(0)
        dfdx = x / x_base
        # prevent inf or nan case
        mask = (x_base == 0)
        dfdx[mask] = 0.0
        
        epsil = x - dfdx * x_base
        epsil = epsil.detach()

    return x, dfdx.detach(), epsil

def linear_linear(x, m, a=None, gamma=1.0):
    if isinstance(m, nn.Linear):
        w = m.weight
        if gamma < 1.0:
            w = F.leaky_relu(w, gamma)
        b = m.bias
        # if r is not None:
        #     w_ = r.unsqueeze(2) * w.unsqueeze(0)
        # else:
        #     w_ = w
        # x_hat = torch.matmul(x.unsqueeze(1), w_).squeeze(1)
        if a is not None:
            x = x * a
        x = F.linear(x.unsqueeze(1), w.t()).squeeze(1)
        # Todo(j-pong): check this line for equal to original
        # print(F.mse_loss(x, x_hat))
    else:                
        raise AttributeError("This module is not approprate to this function.")
    
    return x

def linear_conv2d(x, m, a=None, gamma=1.0):
    if isinstance(m, nn.Conv2d):
        w = m.weight
        if gamma < 1.0:
            w = F.leaky_relu(w, gamma)
        b = m.bias 
        
        if a is not None:
            x = x * a
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

def linear_maxpool2d(x, m, ind=None):
    if isinstance(m, nn.MaxPool2d):
        x = F.max_unpool2d(x, ind, m.kernel_size, m.stride)
    else:
        raise AttributeError("This module is not approprate to this function.")
    
    return x