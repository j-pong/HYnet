import torch
from torch import nn

class MultiMaskLoss(nn.Module):
    def __init__(
        self, 
        criterion=nn.CrossEntropyLoss(reduction='none')):
        super().__init__()
        self.criterion = criterion

    def forward(self, x, y, masks=None, reduction='mean'):
        if masks is None:
            loss = self.criterion(input=x, target=y)
            denorm = 0
            for sz in y.size():
                denorm *= sz
            loss /= denorm
        else:
            accum_mask = None
            for mask in masks:
                if accum_mask is not None:
                    accum_mask = mask | accum_mask
                else:
                    accum_mask = mask
            loss = self.criterion(input=x, target=y)
            denom = (~accum_mask).float().sum()
            loss = loss.masked_fill(accum_mask, 0) / denom
        if reduction == 'mean':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError("{} type reduction is not defined".format(reduction))
