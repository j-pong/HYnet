#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import six

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import chainer
from chainer import reporter

from moneynet.nets.unsup.utils import pad_for_shift, reverse_pad_for_shift, selector, select_with_ind
from moneynet.nets.unsup.attention import attention
from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqLoss, SeqMultiMaskLoss


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        """Report at every step."""
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)


class InferenceNet(nn.Module):
    def __init__(self, idim, odim, args):
        super(InferenceNet, self).__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

        # next frame predictor
        self.encoder_type = args.etype
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
        x_ind = torch.max(energy, dim=-1)[1]  # (B, Tmax)
        x = select_with_ind(x, x_ind)  # (B, Tmax, hdim)
        return x, x_ind

    @staticmethod
    def energy_pooling_mask(x, part_size, share=False):
        energy = x.pow(2)
        if share:
            indices = torch.topk(energy, k=part_size * 2, dim=-1)[1]  # (B, T, cdim*2)
        else:
            indices = torch.topk(energy, k=part_size, dim=-1)[1]  # (B, T, cdim)
        mask = F.one_hot(indices[:, :, :part_size], num_classes=x.size(-1)).float().sum(-2)  # (B, T, hdim)
        mask_share = F.one_hot(indices, num_classes=x.size(-1)).float().sum(-2)  # (B, T, hdim)
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

    def forward(self, x, mask_prev, decoder_type):
        if self.encoder_type == 'conv1d':
            x, _ = pad_for_shift(key=x, pad=self.input_extra,
                                 window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
            h = self.encoder(x)  # (B, Tmax, *, hdim)
            # max pooling along shift size
            h, h_ind = self.energy_pooling(h)
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
    @staticmethod
    def add_arguments(parser):
        """Add arguments"""
        group = parser.add_argument_group("simnn setting")
        group.add_argument("--etype", default="conv1d", type=str)
        group.add_argument("--hdim", default=160, type=int)
        group.add_argument("--cdim", default=16, type=int)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        super(Net, self).__init__()
        # network hyperparameter
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim
        self.ignore_id = ignore_id
        self.subsample = [1]

        # reporter for monitoring
        self.reporter = Reporter()

        self.inference = InferenceNet(idim, odim, args)

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

    def forward(self, xs_pad, ilens, ys_pad):
        # prepare data
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(y)
            y = y.masked_fill(seq_mask, self.ignore_in)
        else:
            seq_mask = y == self.ignore_out
        buffs = {'y_ele': [], 'x_ele': [], 'x_res': [], 'y_res': [],
                 'theta_opt': [], 'sim_opt': [],
                 'loss': [], 'attn': [], 'hs': []}

        # start iteration for superposition
        x_res = x.clone()
        y_res = y.clone()
        hidden_mask_src = None
        hidden_mask_self = None
        attn_prev = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # self 1. action for self feature (current version just pad action support)
            x_aug, _ = pad_for_shift(key=x_res, pad=self.odim - 1,
                                     window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)

            # self 2. selection of action
            y_align_opt, sim_opt, theta_opt = selector(x_aug, y_res, measurement=self.measurement)

            # self 3. attention to target feature with selected feature
            attn = attention(y_align_opt, y_res, temper=self.temper)
            if attn_prev is not None:
                consistency_sim_th = 0.80
                consistency_var_th = 1e-6
                # check similarity of each data frame
                denom = torch.norm(attn, dim=-1) * torch.norm(attn_prev, dim=-1)
                sim = torch.sum(attn * attn_prev, dim=-1) / denom  # (B, T)
                sim_mask = sim > consistency_sim_th
                assert torch.isnan(sim).sum() == 0.0

                # compute similarity with
                var = self.max_variance(attn, dim=-1)
                var_mask = var.squeeze() > consistency_var_th
                denom = var_mask.float().sum()
                if denom > 1e-6:
                    sim = sim.masked_select(var_mask) / denom
                    sim = sim.sum()
                else:
                    sim = sim.mean()

                if sim > consistency_sim_th:
                    break
                y_align_opt_attn = y_align_opt * attn * sim_mask.float().unsqueeze(-1)
            else:
                y_align_opt_attn = y_align_opt * attn
            attn_prev = attn

            # self 4. reverse action
            x_align_opt_attn = reverse_pad_for_shift(key=y_align_opt_attn, pad=self.odim - 1, window=self.odim,
                                                     theta=theta_opt)

            if self.selftrain:
                # self 5 inference
                x_ele, hidden_mask_self = self.inference(x_align_opt_attn, hidden_mask_self,
                                                         decoder_type='self')
                x_ele += x_align_opt_attn

                # self 6 loss
                masks = [seq_mask.view(-1, self.idim),
                         torch.abs(theta_opt - self.idim + 1).unsqueeze(-1).repeat(1, 1, self.idim).view(-1,
                                                                                                         self.idim) > self.energy_th]
                loss_local_self = self.criterion(x_ele.view(-1, self.idim), x_res.view(-1, self.idim), masks)

                # source 1.action with inference feature that concern relation of pixel of frame
                x_aug, _ = pad_for_shift(key=x_ele, pad=self.odim - 1,
                                         window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)

                # source 2. selection of action
                y_align_opt, sim_opt, theta_opt = selector(x_aug, y_res, measurement=self.measurement)
                y_align_opt_attn = y_align_opt
            else:
                x_ele = x_align_opt_attn
            # 2. feedforward for src estimation
            y_ele, hidden_mask_src = self.inference(y_align_opt_attn, hidden_mask_src,
                                                    decoder_type='src')
            # source 3. inference
            masks = [seq_mask.view(-1, self.odim),
                     torch.abs(theta_opt - self.odim + 1).unsqueeze(-1).repeat(1, 1, self.odim).view(-1,
                                                                                                     self.odim) > self.energy_th]
            loss_local_src = self.criterion(y_ele.view(-1, self.odim), y_res.view(-1, self.odim), masks)

            # source 4. loss
            if self.selftrain:
                loss = loss_local_src.sum() + loss_local_self.sum()
            else:
                loss = loss_local_src.sum()

            # compute residual feature
            y_res = (y_res - y_ele).detach()
            x_res = (x_res - x_ele).detach()

            # buffering
            if not self.training:
                buffs['theta_opt'].append(theta_opt[0])
                buffs['sim_opt'].append(sim_opt[0])
                buffs['attn'].append(attn[0])
                buffs['x_res'].append(x_res)
                buffs['y_res'].append(y_res)
            buffs['loss'].append(loss)
            buffs['x_ele'].append(x_ele)
            buffs['y_ele'].append(y_ele)

        # 5. total loss compute
        loss = torch.stack(buffs['loss'], dim=-1).mean()

        return loss
