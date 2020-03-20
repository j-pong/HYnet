#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torch import nn


class SeqLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduce=None)):
        super(SeqLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, mask, reduce='mean'):
        denom = (~mask).float().sum()
        loss = self.criterion(input=x, target=y)
        loss = loss.masked_fill(mask, 0) / denom
        if reduce == 'mean':
            return loss.sum()
        elif reduce == None:
            return loss
        else:
            raise ValueError("{} type reduce is not defined".format(reduce))


class SeqEnergyLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduce=None)):
        super(SeqEnergyLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, seq_mask, feat_dim, theta_opt, energy_th, reduce='mean'):
        move_energy = torch.abs(theta_opt - feat_dim + 1).view(-1, 1) + 1.0
        move_mask = torch.abs(theta_opt - feat_dim + 1).view(-1, 1) > energy_th

        denom = (~seq_mask).float().sum()
        loss = self.criterion(input=x, target=y)
        loss = loss.masked_fill(seq_mask, 0.0).masked_fill(move_mask, 0.0) / move_energy / denom
        if reduce == 'mean':
            return loss.sum()
        elif reduce == None:
            return loss
        else:
            raise ValueError("{} type reduce is not defined".format(reduce))
