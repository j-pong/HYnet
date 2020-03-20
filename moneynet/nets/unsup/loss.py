#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from torch import nn


class SeqLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduction='none')):
        super(SeqLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, mask, reduction='mean'):
        denom = (~mask).float().sum()
        loss = self.criterion(input=x, target=y)
        loss = loss.masked_fill(mask, 0) / denom
        if reduction == 'mean':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("{} type reduction is not defined".format(reduction))


class SeqMultiMaskLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduction='none')):
        super(SeqMultiMaskLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, masks, reduction='mean'):
        pass
