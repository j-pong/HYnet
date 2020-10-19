import copy 

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from hynet.layers.brew_layer import linear_conv2d, linear_linear, linear_maxpool2d, grad_activation

lienar_layer = (nn.Conv2d, nn.Linear)
piece_wise_activation = (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)
piece_shrink_activation = (nn.MaxPool1d, nn.MaxPool2d)

def minimaxn(x, dim, max_b=1.0, min_a=0.0):
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    min_x = torch.min(x, dim=dim, keepdim=True)[0]

    norm = (max_x - min_x)
    norm[norm == 0.0] = 1.0

    x = (x - min_x) * (max_b - min_a)/ norm + min_a
    
    return x

def attn_norm(attn, max_b=1.0, min_a=0.0):
    # attention normalization
    b_sz, ch, in_h, in_w = attn.size()
    # attn normalization
    attn = attn.flatten(start_dim=2) 
    if isinstance(max_b, torch.Tensor):
        max_b = max_b.flatten(start_dim=2)
        min_a = min_a.flatten(start_dim=2)
    attn = minimaxn(attn, dim=-1, max_b=max_b, min_a=min_a)
    attn = attn.view(b_sz, ch, in_h, in_w).float()
    
    return attn

class BrewModuleList(nn.ModuleList):
    def __init__(self, modules = None):
        super().__init__(modules)

        self.aug_hat = {"a_hat":{}, "c_hat":{}}

class BrewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_size = None

        self.encoder = BrewModuleList([])
        self.decoder = BrewModuleList([])

    def backward_linear_impl(self, x, mlist):
        max_len = mlist.__len__()
        for idx in range(max_len):
            idx = max_len - idx - 1
            m = mlist.__getitem__(idx)
            if isinstance(m, lienar_layer):
                a = mlist.aug_hat["a_hat"][idx]
                if isinstance(m, nn.Conv2d):
                    x = linear_conv2d(x, m, a)
                elif isinstance(m, nn.Linear):
                    x = linear_linear(x, m, a)
            elif isinstance(m, piece_wise_activation):
                pass
            elif isinstance(m, piece_shrink_activation):
                a = mlist.aug_hat["a_hat"][idx]
                _, ind = m(a)
                x = linear_maxpool2d(x, m, ind)
            elif isinstance(m, nn.Flatten):
                x = x.view(self.attn_size[0], self.attn_size[1],
                           self.attn_size[2], self.attn_size[3])
            else:
                raise NotImplementedError

        return x

    def forward_linear_impl(self, x, mlist, bias_mask=False):
        max_len = mlist.__len__()
        for idx in range(max_len):
            m = mlist.__getitem__(idx)
            if isinstance(m, lienar_layer):
                x = m(x)
                a = mlist.aug_hat["a_hat"][idx]
                # c = self.aug_hat["c_hat"][idx]
                if a is not None:
                    x = a * x # + c
                else:
                    x = x
            elif isinstance(m, piece_wise_activation):
                pass
            elif isinstance(m, piece_shrink_activation):
                a = mlist.aug_hat["a_hat"][idx]
                _, ind = m(a)
                x_flat = x.flatten(start_dim=2)
                x = x_flat.gather(dim=2, index=ind.flatten(start_dim=2)).view_as(ind)
            elif isinstance(m, nn.Flatten):
                x = m(x)
            else:
                raise NotImplementedError

        return x

    def forward_impl(self, x, mlist, bias_mask=False):
        a_hat_cum = None
        c_hat_cum = None

        save_aug = False
        idx_lin = None

        max_len = mlist.__len__()
        for idx in range(max_len):
            m = mlist.__getitem__(idx)
            # make save activation flag
            if idx == (max_len - 1):
                if isinstance(m, lienar_layer) or isinstance(m, piece_wise_activation):
                    save_aug = True
                else:
                    save_aug = False
            elif isinstance(mlist.__getitem__(idx + 1), lienar_layer) or isinstance(mlist.__getitem__(idx + 1), piece_shrink_activation):
                if isinstance(m, piece_wise_activation):
                    save_aug = True
                else:
                    save_aug = False
            else:
                save_aug = False

            # forward processs
            if isinstance(m, lienar_layer):
                x = m(x)
                idx_lin = idx
            elif isinstance(m, piece_wise_activation):
                x, a_hat, c_hat = grad_activation(x, m, training=self.training)
                if a_hat_cum is not None:
                    a_hat_cum = a_hat * a_hat_cum
                    # c_hat_cum = a_hat * c_hat_cum  + c_hat
                else:
                    a_hat_cum = a_hat
                    # c_hat_cum = c_hat
            elif isinstance(m, piece_shrink_activation):
                x, a_hat, _ = grad_activation(x, m, training=self.training, shrink=True)
                mlist.aug_hat["a_hat"][idx] = a_hat
            elif isinstance(m, nn.Flatten):
                x = m(x)
            else:
                raise NotImplementedError

            if save_aug:
                # assert a_hat_cum is not Nones
                assert idx_lin is not None
                mlist.aug_hat["a_hat"][idx_lin] = a_hat_cum
                # self.aug_hat["c_hat"][idx_lin] = c_hat_cum
                a_hat_cum = None
        return x

    def backward_linear(self, x, y):
        # backward
        attn = self.backward_linear_impl(y, self.decoder)
        attn = self.backward_linear_impl(attn, self.encoder)

        # sign-field
        sign = torch.sign(x)
        attn = attn * sign
        
        return attn

    def forward(self, x, bias_mask=False, mode='none'):
        if mode == 'none':
            for m in self.encoder:
                if isinstance(m, piece_shrink_activation):
                    x, _ = m(x)
                else:
                    x = m(x)
            for m in self.decoder:
                if isinstance(m, piece_shrink_activation):
                    x, _ = m(x)
                else:
                    x = m(x)
            x_non_linear = x
        elif mode == 'brew':
            # regular forward pass
            x_non_linear = self.forward_impl(x, self.encoder)
            self.attn_size = x_non_linear.size()
            x_non_linear = self.forward_impl(x_non_linear, self.decoder)

            # linearization
            x_linear = self.forward_linear_impl(x, self.encoder)
            x_linear = self.forward_linear_impl(x_linear, self.decoder)

            # linearization error check
            loss_brew = F.mse_loss(x_non_linear, x_linear)
            if loss_brew > 1e-20:
                raise ValueError("loss of brew {} bigger than 1e-20".format(loss_brew))

        return x_non_linear
