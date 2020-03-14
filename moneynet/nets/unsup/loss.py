#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        elif reduce is None:
            return loss
        else:
            raise ValueError("{} type reduction is not defined".format(reduce))
