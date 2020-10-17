import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from hynet.layers.brew_layer import linear_conv2d, linear_linear, linear_maxpool2d, grad_activation

lienar_layer = (nn.Conv2d, nn.Linear)
piece_wise_activation = (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)
piece_shrink_activation = (nn.MaxPool1d, nn.MaxPool2d)

class LRPModel(nn.Module):
    @torch.enable_grad()
    def backward_lrp_impl(self, x, mlist, ratio):
        max_len = mlist.__len__()
        for idx in range(max_len):
            m = mlist.__getitem__(max_len - idx - 1)
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
                rat = ratio.pop()
                rat = (rat.data).requires_grad_(True)
                if isinstance(m, nn.MaxPool2d):
                    z, _ = m(rat)
                elif isinstance(m, nn.Conv2d):
                    w = F.relu(m.weight)
                    # w = m.weight
                    b = m.bias
                    z = F.conv2d(rat, w, bias=b, stride=m.stride, padding=m.padding)
                elif isinstance(m, nn.Linear):
                    w = F.relu(m.weight)
                    # w = m.weight
                    b = m.bias
                    z = F.linear(rat, w, b)
                else:
                    z = m(rat)
                z = z + 1e-9 + 0.25 * ((z ** 2).mean()**.5).data
                s = (x / z).data
                if isinstance(m, nn.Linear):
                    c = F.linear(s, w.t())
                else:
                    c = torch.autograd.grad((z * s).sum(), 
                                            rat, 
                                            retain_graph=True,
                                            create_graph=True)[0]
                x = (rat * c).data
        assert len(ratio) == 0

        return x

    def forward_lrp_impl(self, x, mlist, ratio):
        max_len = mlist.__len__()
        for idx, m in enumerate(mlist):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                x = m(x)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                x = m(x)
                if isinstance(mlist.__getitem__(idx + 1), (nn.Conv2d, nn.Linear, nn.MaxPool2d)):
                    ratio.append(x.data.clone())
            elif isinstance(m, nn.MaxPool2d):
                x = m(x)
                if m.return_indices:
                    x, _ = x
                if not (max_len - 1) == idx:
                    ratio.append(x.data.clone())
            elif isinstance(m, nn.Flatten):
                x = m(x)
                # waring; this just for encoder and decoder model with cnn-dnn
                ratio.append(x.data.clone())
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError
            else:
                raise NotImplementedError
            
        return x

    def forward_lrp(self, x):
        ratios = [[], []]

        ratios[0].append(x.detach())
        x = self.forward_lrp_impl(x, self.encoder, ratios[0])
        x = self.forward_lrp_impl(x, self.decoder, ratios[1])
        
        return x, ratios


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
                a_hat = mlist.aug_hat["a_hat"][idx]
                # c_hat = self.aug_hat["c_hat"][idx]
                if isinstance(m, nn.Conv2d):
                    x = linear_conv2d(x, m, a_hat)
                elif isinstance(m, nn.Linear):
                    x = linear_linear(x, m, a_hat)
            elif isinstance(m, piece_wise_activation):
                pass
            elif isinstance(m, piece_shrink_activation):
                a_hat = mlist.aug_hat["a_hat"][idx]
                x = linear_maxpool2d(x, m, a_hat)
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
                a_hat = mlist.aug_hat["a_hat"][idx]
                # c_hat = self.aug_hat["c_hat"][idx]
                if a_hat is not None:
                    x = m(x) * a_hat # + c_hat
                else:
                    x = m(x)
            elif isinstance(m, piece_wise_activation):
                pass
            elif isinstance(m, piece_shrink_activation):
                a_hat = mlist.aug_hat["a_hat"][idx]
                x_flat = x.flatten(start_dim=2)
                x = x_flat.gather(dim=2, index=a_hat.flatten(start_dim=2)).view_as(a_hat)
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
                if bias_mask:
                    a = mlist.aug_hat["a_hat"][idx]
                    w = m.weight
                    b = m.bias
                    if isinstance(m, nn.Conv2d):
                        if a is not None:
                            b = b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            b = b.repeat(x.size(0), 1, x.size(2), x.size(3))
                            b = a * b
                            x = F.conv2d(x, w, stride=m.stride, padding=m.padding)
                            x = x + b
                        else:
                            x = F.conv2d(x, w, bias=b, stride=m.stride, padding=m.padding)
                    elif isinstance(m, nn.Linear):
                        if a is not None:
                            b = b.unsqueeze(0)
                            b = b.repeat(x.size(0), 1)
                            b = a * b
                            x = F.linear(x, w)
                        else:
                            x = F.linear(x, w, b)   
                else:
                    x = m(x)
                idx_lin = idx
            elif isinstance(m, piece_wise_activation):
                x, a_hat, c_hat = grad_activation(x, m, training=self.training)
                if a_hat_cum is not None:
                    a_hat_cum = a_hat * a_hat_cum
                    # c_hat_cum = c_hat * c_hat_cum + c_hat
                else:
                    a_hat_cum = a_hat
                    # c_hat_cum = c_hat
            elif isinstance(m, piece_shrink_activation):
                x, a_hat = m(x)
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

    def backward_linear(self, y):
        # backward
        attn = self.backward_linear_impl(y, self.decoder)
        attn = self.backward_linear_impl(attn, self.encoder)
        
        return attn

    def forward_linear(self, x):
        x = self.forward_linear_impl(x, self.encoder)
        x = self.forward_linear_impl(x, self.decoder)

        return x

    def forward(self, x, bias_mask=False):
        x = self.forward_impl(x, self.encoder, bias_mask)
        self.attn_size = x.size()
        x = self.forward_impl(x, self.decoder, bias_mask)

        return x
