#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn

import numpy as np


def initialize(model, init_type="xavier_uniform"):
    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()


class NetLoss(nn.Module):
    def __init__(self, ignore_val, criterion=nn.MSELoss(reduce=None)):
        super(NetLoss, self).__init__()
        self.criterion = criterion
        self.ignore_val = ignore_val

    def forward(self, x, y):
        mask = y == self.ignore_val
        denom = (~mask).float().sum()
        loss = self.criterion(input=x, target=y)
        return loss.masked_fill(mask, 0).sum() / denom


class Net(nn.Module):
    def __init__(self, idim, odim, args, reporter):
        super(Net, self).__init__()
        # utils
        self.reporter = reporter

        # network hyperparameter
        self.hdim = args.hdim
        self.ignore_val = args.ignore_val

        # freq distribution design
        weight = self._hg_kernel(np.arange(0, self.hdim, dtype=np.float32), mu=0.0, sigma=430)
        weight = np.concatenate([1.0 - weight, weight], axis=0).T  # binomial distribution needs for hidden mask
        self.freq = torch.from_numpy(weight)

        # simple network add
        self.fc1 = nn.Linear(idim, self.hdim)
        self.fc2 = nn.Linear(self.hdim, odim)

        # network training related
        self.criterion = NetLoss(ignore_val=self.ignore_val)

        # initialize parameter
        self.reset_parameters()

    def _hg_kernel(self, x, mu, sigma):
        denom = 1 / (sigma * np.sqrt(2 * np.pi))
        y = denom * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y = y / np.max(y)  # normalizing with max so that first elements of hidden units is always fire
        y = np.expand_dims(y, axis=0)

        return y

    def _lin_kernel(self, x):
        # ToDo(j-pong): inverse linear space base for kernel
        y = x
        y = y / np.max(y)
        y = np.expand_dims(y, axis=0)

        return y

    def freq_mask(self, x):
        mask = torch.multinomial(self.freq, num_samples=x.size(0), replacement=True)  # [B * T, C]
        mask = mask.to(x.device).T

        return x * mask

    def reset_parameters(self):
        initialize(self)

    def forward(self, x, y):
        x_size = x.size()
        y_size = y.size()

        x = x.view(-1, x_size[-1])
        y = y.view(-1, y_size[-1])

        x = self.freq_mask(self.fc1(x))
        x = self.fc2(x)

        loss = self.criterion(x, y)

        return loss, x
