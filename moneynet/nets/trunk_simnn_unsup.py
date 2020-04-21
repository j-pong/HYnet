#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from moneynet.nets.unsup.utils import pad_for_shift, select_with_ind
from moneynet.nets.unsup.attention import attention
from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqMultiMaskLoss


class InferenceNet(nn.Module):
    def __init__(self, idim, odim, args):
        super(InferenceNet, self).__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

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

    @staticmethod
    def energy_pooling(x, dim=-1):
        energy = x.pow(2).sum(dim)
        x_ind = torch.max(energy, dim=-1)[1]  # (B, *)
        x = select_with_ind(x, x_ind)  # (B, *, hdim)
        return x, x_ind

    @staticmethod
    def energy_pooling_mask(x, part_size, share=False):
        energy = x.pow(2)
        if share:
            indices = torch.topk(energy, k=part_size * 2, dim=-1)[1]  # (B, *, cdim*2)
        else:
            indices = torch.topk(energy, k=part_size, dim=-1)[1]  # (B, *, cdim)
        mask = F.one_hot(indices[:, :, :part_size], num_classes=x.size(-1)).float().sum(-2)  # (B, *, hdim)
        mask_share = F.one_hot(indices, num_classes=x.size(-1)).float().sum(-2)  # (B, *, hdim)
        return mask, mask_share

    def hidden_exclude_activation(self, h, mask_prev):
        if mask_prev is None:
            mask_cur, mask_cur_share = self.energy_pooling_mask(h, self.cdim, share=True)
            mask_prev = mask_cur
        else:
            assert mask_prev is not None
            h[mask_prev.bool()] = 0.0
            mask_cur, mask_cur_share = self.energy_pooling_mask(h, self.cdim, share=True)
            mask_prev = mask_prev + mask_cur
        h = h.masked_fill(~(mask_cur_share.bool()), 0.0)
        return h, mask_prev

    def forward(self, x, mask_prev, decoder_type, exclude=True):
        if self.encoder_type == 'conv1d':
            x, _ = pad_for_shift(key=x, pad=self.input_extra,
                                 window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
            h = self.encoder(x)  # (B, Tmax, *, hdim)
            # max pooling along shift size
            h, h_ind = self.energy_pooling(h)
            if exclude:
                # max pooling along hidden size
                h, mask_prev = self.hidden_exclude_activation(h, mask_prev)
            # feedforward decoder
            assert self.idim == self.odim
            if decoder_type == 'self':
                x_ext = self.decoder_self(h)
            elif decoder_type == 'src':
                x_ext = self.decoder_src(h)
            # output trunk along feature side with window
            x_ext = [select_with_ind(x_ext, x_ext.size(-1) - 1 - h_ind - i) for i in torch.arange(self.idim).flip(0)]
            x = torch.stack(x_ext, dim=-1)
        elif self.encoder_type == 'linear':
            h = self.encoder(x)
            # max pooling along hidden size
            h, mask_prev = self.hidden_exclude_activation(h, mask_prev)
            # feedforward decoder
            assert self.idim == self.odim
            if decoder_type == 'self':
                x = self.decoder_self(h)
            elif decoder_type == 'src':
                x = self.decoder_src(h)

        return x, mask_prev


class Net(nn.Module):
    def __init__(self, idim, odim, args, reporter):
        super(Net, self).__init__()
        # utils
        self.reporter = reporter

        # network hyperparameter
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out
        self.selftrain = args.self_train

        self.inference = InferenceNet(idim, odim, args)

        # training hyperparameter
        self.measurement = args.similarity
        self.temper = args.temperature
        self.energy_th = args.energy_threshold

        # network training related
        self.criterion = SeqMultiMaskLoss(criterion=nn.MSELoss(reduction='none'))

        # initialize parameter
        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    @staticmethod
    def max_variance(p, dim=-1):
        mean = torch.max(p, dim=dim, keepdim=True)[0]  # (B, T, 1)
        # get normalized confidence
        numer = (mean - p).pow(2)
        denom = p.size(dim) - 1
        return torch.sum(numer, dim=-1) / denom

    @staticmethod
    def prob(x, temper, pseudo_zero=1e-6):
        score = torch.abs(x)
        score[score < pseudo_zero] = pseudo_zero
        score = score / score.sum(dim=-1, keepdim=True)
        score = torch.exp(torch.log(score) / temper)
        p = score / score.sum(dim=-1, keepdim=True)
        return p

    def forward(self, x, ys):
        """
        :parm torch.Tensor x: (B,T,C)
        :parm torch.Tensor y: (B,T,r,C)
        """
        # prepare data
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(ys[:, :, 0, :])
            ys = ys.masked_fill(seq_mask.unsqueeze(-2), self.ignore_in)
        else:
            seq_mask = ys[:, :, 0, :] == self.ignore_out
        buffs = {'loss': [], 'x_ele': []}

        # start iteration for superposition
        x_res = x.clone()
        hidden_mask_self = None
        hidden_mask_src = None
        for _ in six.moves.range(int(self.hdim / self.cdim) - 1):
            x_aug, _ = pad_for_shift(key=x_res, pad=self.odim - 1, window=self.odim)

            sim = torch.sum(y * x, dim=-1)  # (B, Tmax, *)
            sim[torch.isnan(sim)] = 0.0
            sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax)

            x_ele, hidden_mask_self = self.inference(x_res, hidden_mask_self, decoder_type='self')  # (B,T,C)

            masks = [seq_mask.view(-1, self.idim)]
            loss_local_self = self.criterion(x_ele.view(-1, self.idim), x_res.view(-1, self.idim), masks)
            loss = loss_local_self

            x_res = (x_res - x_ele).detach()

            buffs['loss'].append(loss)
            buffs['x_ele'].append(x_ele)

        # 5. total loss compute
        loss = torch.stack(buffs['loss'], dim=-1).mean()

        # give information of data to reporter
        self.reporter.report_dict['loss'] = float(loss)
        x_hyp = torch.stack(buffs['x_ele'], dim=-1)
        loss_x = self.criterion(x_hyp.sum(-1).view(-1, self.idim),
                                x.view(-1, self.idim),
                                [seq_mask.view(-1, self.odim)])
        self.reporter.report_dict['loss_x'] = float(loss_x)

        return loss
