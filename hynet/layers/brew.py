import copy 

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

lienar_layer = (nn.Conv2d, nn.Linear)
piece_wise_activation = (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)
piece_shrink_activation = (nn.MaxPool1d, nn.MaxPool2d)

class BrewModuleList(nn.ModuleList):
    def __init__(self, modules = None):
        super().__init__(modules)

        self.aug_hat = {"a_hat":{}, "c_hat":{}}

class BrewModule(nn.Module):
    def __init__(self):
        super().__init__()

    def grad_activation(self, x, module, training=True, shrink=False):
        # checkout inplace option for accuratly gradient
        if isinstance(module, nn.ReLU):
            assert module.inplace is False

        with torch.autograd.set_grad_enabled(True):
            x_base = x.requires_grad_()
            # if training:
            #     x_base = x
            # else:
            #     # validation mode has no backward graph for autograd 
            #     # Thus, we force variable to make graph locally 
            #     x_base = torch.autograd.Variable(x.data)
            #     x_base.requires_grad = True

            if shrink:
                x = module(x_base)[0]
            else:
                x = module(x_base)
            # caculate grad via auto grad respect to x_base
            dfdx = torch.autograd.grad(x.sum(), x_base, retain_graph=True)
            dfdx = dfdx[0].data#.double

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

    def linear_linear(self, x, m, a=None, gamma=1.0):
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

    def linear_conv2d(self, x, m, a=None, gamma=1.0):
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

    def linear_maxpool2d(self, x, m, ind=None):
        if isinstance(m, nn.MaxPool2d):
            x = F.max_unpool2d(x, ind, m.kernel_size, m.stride)
        else:
            raise AttributeError("This module is not approprate to this function.")
        
        return x

class BrewModel(BrewModule):
    def __init__(self):
        super(BrewModel, self).__init__()

        self.attn_size = None

        self.encoder = BrewModuleList([])
        self.decoder = BrewModuleList([])

        self.loss_brew = 0.0

    def backward_linear_impl(self, x, mlist):
        max_len = mlist.__len__()
        for idx in range(max_len):
            idx = max_len - idx - 1
            m = mlist.__getitem__(idx)
            if isinstance(m, lienar_layer):
                a = mlist.aug_hat["a_hat"][idx]
                # c = mlist.aug_hat["c_hat"][idx]
                if isinstance(m, nn.Conv2d):
                    x = self.linear_conv2d(x, m, a)
                    # if c is not None:
                    #     x += self.linear_conv2d(c, m)
                elif isinstance(m, nn.Linear):
                    x = self.linear_linear(x, m, a)
                    # if c is not None:
                    #     x += self.linear_linear(c, m)
            elif isinstance(m, piece_wise_activation):
                pass
            elif isinstance(m, piece_shrink_activation):
                a = mlist.aug_hat["a_hat"][idx]
                _, ind = m(a)
                x = self.linear_maxpool2d(x, m, ind)
            elif isinstance(m, nn.Flatten):
                x = x.view(self.attn_size[0], self.attn_size[1],
                           self.attn_size[2], self.attn_size[3])
            else:
                raise NotImplementedError

        return x

    def forward_linear_impl(self, x, mlist):
        max_len = mlist.__len__()
        for idx in range(max_len):
            m = mlist.__getitem__(idx)
            if isinstance(m, lienar_layer):
                x = m(x.float())
                a = mlist.aug_hat["a_hat"][idx]
                c = mlist.aug_hat["c_hat"][idx]
                if a is not None:
                    x = a * x + c
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

    def forward_impl(self, x, mlist):
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
                x, a_hat, c_hat = self.grad_activation(x, m, training=self.training)
                if a_hat_cum is not None:
                    a_hat_cum = a_hat * a_hat_cum
                    c_hat_cum = a_hat * c_hat_cum + c_hat
                else:
                    a_hat_cum = a_hat
                    c_hat_cum = c_hat
            elif isinstance(m, piece_shrink_activation):
                x, a_hat, _ = self.grad_activation(x, m, training=self.training, shrink=True)
                mlist.aug_hat["a_hat"][idx] = a_hat
            elif isinstance(m, nn.Flatten):
                x = m(x)
            else:
                raise NotImplementedError

            if save_aug:
                # assert a_hat_cum is not Nones
                assert idx_lin is not None
                mlist.aug_hat["a_hat"][idx_lin] = a_hat_cum
                mlist.aug_hat["c_hat"][idx_lin] = c_hat_cum
                a_hat_cum = None
                c_hat_cum = None
        return x

    def backward_linear(self, x, y):
        with torch.no_grad():
            # backward
            attn = self.backward_linear_impl(y, self.decoder)
            attn = self.backward_linear_impl(attn, self.encoder)

            # sign-field
            sign = torch.sign(x)
            attn = attn * sign
        
        return attn

    def forward(self, x, mode='none'):
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

            with torch.no_grad():
                # linearization
                x_linear = self.forward_linear_impl(x, self.encoder)
                x_linear = self.forward_linear_impl(x_linear, self.decoder)

                # linearization error check
                self.loss_brew = F.mse_loss(x_non_linear, x_linear).detach()
                if self.loss_brew.float() > 1e-18:
                    raise ValueError("loss of brew {} bigger than 1e-18".format(self.loss_brew))

        return x_non_linear
