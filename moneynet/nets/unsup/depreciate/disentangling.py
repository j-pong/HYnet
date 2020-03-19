#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F

from torch import nn

from moneynet.nets.unsup.utils import pad_for_shift, select_with_ind
from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqLoss


class Disentangling(nn.Module):
    def __init__(self, idim, odim, args):
        super(Disentangling, self).__init__()
        # network hyperparameter
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

        # training hyperparameter
        self.energy_th = args.energy_threshold

        # next frame predictor
        self.encoder_type = args.encoder_type
        if self.encoder_type == 'conv1d':
            self.input_extra = idim
            self.output_extra = odim
            self.encoder = nn.Linear(idim + self.input_extra, self.hdim)
            self.decoder_src = nn.Linear(self.hdim, odim + self.output_extra)
            self.decoder_self = nn.Linear(self.hdim, idim + self.input_extra)
        elif self.encoder_type == 'linear':
            self.encoder = nn.Linear(idim, self.hdim)
            self.decoder_src = nn.Linear(self.hdim, odim)
            self.decoder_self = nn.Linear(self.hdim, idim)

        # network training related
        self.criterion = SeqLoss()

        # initialize parameter
        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    def relay(self, h, theta):
        # :param torch.Tensor h: batch of padded source sequences (B, Tmax, hdim)
        h_energy = h.pow(2)
        indices = torch.topk(h_energy, k=self.cdim, dim=-1)[1]  # (B, T, cdim)

        move_mask = torch.abs(theta - self.idim + 1) > self.energy_th  # (B, T)
        cum_move_indices = move_mask.float().cumsum(-1).long()
        indices = [torch.cat([ind[0:1], ind[m]], dim=0) if ind[m].size(0) > 0 else ind[0:1]
                   for m, ind in zip(move_mask, indices)]  # list (B,) with (T_b, cdim)
        indices = [indices[i][ind, :] for i, ind in enumerate(cum_move_indices)]
        indices = torch.stack(indices, dim=0)  # (B, T, cdim)

        mask = F.one_hot(indices, num_classes=self.hdim).float().sum(-2)  # (B, T, hdim)

        return mask

    def hsr(self, h, mask_prev, seq_mask, theta):
        if mask_prev is None:
            mask_cur = self.relay(h, theta)
            mask_prev = mask_cur
            loss_h = None
        else:
            assert mask_prev is not None
            # intersection of prev and current hidden space
            mask_cur = self.relay(h, theta)
            mask_intersection = mask_prev * mask_cur
            seq_mask = seq_mask.prod(-1).unsqueeze(-1).repeat(1, 1, self.hdim).bool()
            # loss define
            h_ = h.clone()
            h_.retain_grad()
            target_mask = (1.0 - mask_intersection)
            loss_h = self.criterion(h_.view(-1, self.hdim),
                                    target_mask.view(-1, self.hdim),
                                    seq_mask.view(-1, self.hdim),
                                    reduction='none')
            loss_h = loss_h.masked_fill(target_mask.view(-1, self.hdim).bool(), 0.0).sum()
            # eliminate fired hidden nodes
            h[mask_prev.bool()] = 0.0
            mask_cur = self.relay(h, theta)
            mask_prev = mask_prev + mask_cur

        h = h.masked_fill(~(mask_cur.bool()), 0.0)

        return h, mask_prev, loss_h

    def forward(self, x, mask_prev, seq_mask, theta, decoder='self'):
        if self.encoder_type == 'conv1d':
            x, _ = pad_for_shift(key=x, pad=self.input_extra,
                                 window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
            h = self.encoder(x)  # (B, Tmax, *, hdim)
            # max pooling along shift size
            h_ind = torch.max(h.pow(2).sum(-1), dim=-1)[1]  # (B, Tmax)
            h = select_with_ind(h, h_ind)  # (B, Tmax, hdim)
            # hidden space regularization
            h, mask_prev, loss_h = self.hsr(h, mask_prev, seq_mask=seq_mask, theta=theta)
            # target trunk along feature side with window
            assert self.idim == self.odim
            if decoder == 'self':
                x_ext = self.decoder_self(h)
            elif decoder == 'src':
                x_ext = self.decoder_src(h)
            else:
                raise AttributeError
            x = torch.stack(
                [select_with_ind(x_ext, x_ext.size(-1) - 1 - h_ind - i) for i in torch.arange(self.idim).flip(0)],
                dim=-1)
        elif self.encoder_type == 'linear':
            h = self.encoder(x)
            h, mask_prev, loss_h = self.hsr(h, mask_prev, seq_mask=seq_mask, theta=theta)
            if decoder == 'self':
                x = self.decoder_self(h)
            elif decoder == 'src':
                x = self.decoder_src(h)
            else:
                raise AttributeError

        return x, mask_prev, loss_h
