#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F

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
    def __init__(self, ignore_in, ignore_out, criterion=nn.MSELoss(reduce=None)):
        super(NetLoss, self).__init__()
        self.criterion = criterion
        self.ignore_in = ignore_in
        self.ignore_out = ignore_out

    def forward(self, x, y):
        if np.isnan(self.ignore_out):
            mask = torch.isnan(y)
            y = y.masked_fill(mask, self.ignore_in)
        else:
            mask = y == self.ignore_out
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
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out

        # freq distribution design
        weight = self._hg_kernel(np.arange(0, self.hdim, dtype=np.float32), mu=0.0, sigma=430)
        weight = np.concatenate([1.0 - weight, weight], axis=0).T  # binomial distribution needs for hidden mask
        self.freq = torch.from_numpy(weight)

        # simple network add
        self.fc1 = nn.Linear(idim, self.hdim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hdim, odim)

        # network training related
        self.criterion = NetLoss(ignore_in=self.ignore_in, ignore_out=self.ignore_out)

        # initialize parameter
        self.reset_parameters()

    def _hg_kernel(self, x, mu, sigma):
        denom = 1 / (sigma * np.sqrt(2 * np.pi))
        y = denom * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y = y / np.max(y)  # normalizing with max so that first elements of hidden units is always fire
        y = np.expand_dims(y, axis=0)  # for binomial distribution

        return y

    def _lin_kernel(self, x):
        y = np.flip(x)
        y = y / np.max(y)  # normalizing with max so that first elements of hidden units is always fire
        y = np.expand_dims(y, axis=0)  # for binomial distribution

        return y

    def freq_mask(self, x):
        mask = torch.multinomial(self.freq, num_samples=x.size(0), replacement=True)  # [B * T, C]
        mask = mask.to(x.device).T

        return x * mask

    def pad_for_shift(self, key, query):
        idim_k = key.size(-1)
        idim_q = query.size(-1)
        assert idim_q >= idim_k
        pad = idim_q - 1

        key_pad_trunk = []
        key_pad = F.pad(key, pad=(pad, pad))  # (B, Tmax, idim_k + pad * 2)
        for i in six.moves.range(idim_k + idim_q - 1):
            key_pad_trunk.append(key_pad[:, :, i:i + idim_q].unsqueeze(-2))
        key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_k + pad * 2)
        return key_pad_trunk

    def simiality(self, key_pad_trunk, query, sim_type='cos'):
        """Measuring similarity of each other tensor

        :param torch.Tensor key_pad_trunk: batch of padded source sequences (B, Tmax, idim_k + idim_q - 1, idim_k)
        :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)
        :param string sim_type:

        :return: max similarity value of sequence (B, Tmax)
        :rtype: torch.Tensor
        :return: max similarity index of sequence  (B, Tmax)
        :rtype: torch.Tensor
        """
        if sim_type == 'cos':
            # ToDo: cos similarity is normalized from comparision scope
            denom = torch.norm(query.unsqueeze(-2), dim=-1) * torch.norm(key_pad_trunk, dim=-1)
            sim = torch.sum(query.unsqueeze(-2) * key_pad_trunk, dim=-1) / denom  # (B, Tmax, idim_k + idim_q - 1)
            sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax)
        else:
            raise AttributeError('{} is not support yet'.format(sim_type))

        return sim_max, sim_max_idx

    def argaug(self, x, y, sim_type='cos'):
        """Find augmentation parameter using grid search

        :param torch.Tensor x: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor y: batch of padded target sequences (B, Tmax, idim)
        :param string sim_type:
        :param dictionary aug_type:

        :return: augmented source sequences (B, Tmax, idim)
        :rtype: torch.Tensor
        :return: value of augmentation parameter (B, Tmax)
        :rtype: torch.Tensor
        """
        sim_max_global = torch.zeros(size=(x.size(0), x.size(1))).to(device=x.device)
        x_aug = None
        for theta in np.linspace(0.5, 1.5, 9):
            # scale
            x_s = F.interpolate(x, scale_factor=theta)
            # shift
            # delta = y.size(-1) - x_s.size(-1)
            # if delta >= 0:
            #     x_s_pad_trunk = self.pad_for_shift(key=x_s, query=y)
            #     sim_max, sim_max_idx = self.simiality(key_pad_trunk=x_s_pad_trunk, query=y)
            # elif delta < 0:
            #     y_pad_trunk = self.pad_for_shift(key=y, query=x_s)
            #     sim_max, sim_max_idx = self.simiality(key_pad_trunk=y_pad_trunk, query=y)
            x_s_pad_trunk = self.pad_for_shift(key=x_s, query=y)
            sim_max, sim_max_idx = self.simiality(key_pad_trunk=x_s_pad_trunk, query=y)
            # maximum scale select
            mask = sim_max_global < sim_max
            sim_max_global[mask] = sim_max[mask]
            # maximum shift select
            kernel_size = kernel.size()
            kernel = kernel.view(-1, kernel_size[-2], kernel_size[-1])
            x_shifted = torch.cat([kernel[i, idxs] for i, idxs in enumerate(sim_max_idx.view(-1))], dim=0)
            if x_aug is None:
                x_aug = x_shifted
            else:
                x_aug[mask.view(-1)] = x_shifted[mask.view(-1)]

        print(x_aug.size())
        exit()

        return x_aug, sim_max_global

    def reset_parameters(self):
        initialize(self)

    def forward(self, x, y):
        x, _ = self.argaug(x, y)

        x_size = x.size()
        y_size = y.size()
        x = x.view(-1, x_size[-1])
        y = y.view(-1, y_size[-1])

        x_ = self.freq_mask(self.fc1(x))
        x = self.fc2(x_) + x  # residual component add to end of network because the network just infer diff.

        loss = self.criterion(x, y)

        return loss, x
