#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torch import nn


class SeqLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduction='none')):
        super(SeqLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, seq_mask, reduction='mean'):
        denom = (~seq_mask).float().sum()
        loss = self.criterion(input=x, target=y)
        loss = loss.masked_fill(seq_mask, 0) / denom
        if reduction == 'mean':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("{} type reduction is not defined".format(reduction))


class SeqEnergyLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduction='none')):
        super(SeqEnergyLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, seq_mask, feat_dim, theta_opt, energy_th, reduction='mean'):
        move_energy = torch.abs(theta_opt - feat_dim + 1).view(-1, 1) + 1.0
        move_energy = move_energy.repeat(1,seq_mask.size(-1))
        move_mask = torch.abs(theta_opt - feat_dim + 1).view(-1, 1) > energy_th
        move_mask = move_mask.repeat(1, seq_mask.size(-1))

        denom = (~seq_mask).float().sum()
        loss = self.criterion(input=x, target=y)
        loss = loss.masked_fill(seq_mask, 0.0).masked_fill(move_mask, 0.0) / move_energy / denom
        if reduction == 'mean':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("{} type reduction is not defined".format(reduction))
