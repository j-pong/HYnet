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

    # ==============================
    # LRP layers
    # ==============================
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

    # ==============================
    # BREW layers
    # ==============================
    def backward_linear_impl(self, x, mlist, ratio, b_hat=None):
        rats = []
        max_len = mlist.__len__()
        for idx in range(max_len):
            m = mlist.__getitem__(max_len - idx - 1)
            if isinstance(m, nn.Conv2d):
                if len(rats) == 0:
                    rat = None
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
        for m in mlist:
            if isinstance(m, nn.Conv2d):
                x = m(x)
                b = m.bias
                if b_hat is not None:
                    b_hat = m(b_hat)
                if b is not None:
                    b = b.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    b = b.repeat(x.size(0), 1, x.size(2), x.size(3))
                    if b_hat is None:
                        b_hat = b
                    x = x - b
            elif isinstance(m, nn.Linear):
                x = m(x)  # change this line wiht matmul
                b = m.bias
                if b_hat is not None:
                    b_hat = m(b_hat)
                if b is not None:
                    b = b.unsqueeze(0)
                    b = b.repeat(x.size(0), 1)
                    if b_hat is None:
                        b_hat = b
                    x = x - b
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                rat = ratio.pop(0)
                x = x * rat
                if b_hat is not None:
                    b_hat = b_hat * rat
            elif isinstance(m, nn.MaxPool2d):
                rat = ratio.pop(0)
                x_flat = x.flatten(start_dim=2)
                x = x_flat.gather(dim=2, index=rat.flatten(start_dim=2)).view_as(rat)
                if b_hat is not None:
                    b_hat_flat = b_hat.flatten(start_dim=2)
                    b_hat = b_hat_flat.gather(dim=2, index=rat.flatten(start_dim=2)).view_as(rat)
            elif isinstance(m, nn.Flatten):
                x = m(x)
                if b_hat is not None:
                    b_hat = m(b_hat)
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError
            else:
                raise NotImplementedError

        return x, b_hat

    def forward_impl(self, x, mlist, ratio):
        for m in mlist:
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Flatten)):
                x = m(x)
            elif isinstance(m, (nn.ReLU, nn.PReLU, nn.Tanh, nn.Dropout)):
                x, rat = calculate_ratio(x, m, training=self.training)
                ratio.append(rat)
            elif isinstance(m, nn.MaxPool2d):
                x, rat = m(x)
                ratio.append(rat)
            elif isinstance(m, nn.BatchNorm2d):
                raise NotImplementedError
            else:
                raise NotImplementedError
            
        return x

    def forward_linear(self, x, ratios):
        x, b_hat = self.forward_linear_impl(x, self.encoder, ratios[0])
        assert len(ratios[0]) == 0
        x, b_hat = self.forward_linear_impl(x, self.decoder, ratios[1], b_hat=b_hat)
        assert len(ratios[1]) == 0

        return x, b_hat

    def forward(self, x, return_ratios=False):
        ratios = [[], []]

        x = self.forward_impl(x, self.encoder, ratios[0])
        x = self.forward_impl(x, self.decoder, ratios[1])

        if return_ratios:
            return x, ratios
        else:
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