import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
    
def grad_activation(x, module, training=True, shrink=False):
    # checkout inplace option for accuratly gradient
    if isinstance(module, nn.ReLU):
        assert module.inplace is False

    with torch.autograd.set_grad_enabled(True):
        if training:
            x_base = x
        else:
            # validation mode has no backward graph for autograd 
            # Thus, we force variable to make graph locally 
            x_base = torch.autograd.Variable(x.data)
            x_base.requires_grad = True

        if shrink:
            x = module(x_base)[0]
        else:
            x = module(x_base)
        # caculate grad via auto grad respect to x_base
        dfdx = torch.autograd.grad(x.sum(), x_base, retain_graph=True)
        dfdx = dfdx[0].data

        # # approximate grad of activation f(x) / x = f'(0)
        # dfdx = x / x_base
        # # prevent inf or nan case
        # mask = (x_base == 0)
        # dfdx[mask] = 0.0

    if shrink:
        epsil = 0.0
    else:
        x_hat = (dfdx * x_base)
        epsil = x - x_hat
        x_hat = x_hat + epsil
        # delta = F.mse_loss(x, x_hat).detach()
        # if  delta.float() > 1e-20:
        #     raise ValueError("{} layer wise loss of brew {} bigger than 1e-20".format(module, delta.float()))
        epsil = epsil.detach()
    dfdx = dfdx.detach()

    return x, dfdx, epsil

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