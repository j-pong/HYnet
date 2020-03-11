#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F

from torch import nn

import numpy as np

from moneynet.nets.utils import pad_for_shift, reverse_pad_for_shift, sim_argmax
from moneynet.nets.attention import attention
from moneynet.nets.initialization import initialize
from moneynet.nets.loss import SeqLoss


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

        # training hyperparameter
        self.measurement = args.similarity
        self.temper = args.temperature
        self.energy_th = args.energy_threshold

        # next frame predictor
        self.encoder_type = args.encoder_type
        if self.encoder_type == 'conv1d':
            self.input_extra = idim
            self.output_extra = odim
            self.encoder = nn.Linear(idim + self.input_extra, self.hdim)
            self.decoder = nn.Linear(self.hdim, odim + self.output_extra)
            self.decoder_self = nn.Linear(self.hdim, idim + self.input_extra)
        elif self.encoder_type == 'linear':
            self.encoder = nn.Linear(idim, self.hdim)
            self.decoder = nn.Linear(self.hdim, odim)
            self.decoder_self = nn.Linear(self.hdim, idim)

        # network training related
        self.criterion = SeqLoss(criterion=nn.MSELoss(reduce=None))
        self.criterion_h = SeqLoss(criterion=nn.MSELoss(reduce=None))

        # initialize parameter
        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    def hsr(self, h, mask_prev, seq_mask, mask=True):
        if mask:
            energy_h = h.pow(2)
            if mask_prev is None:
                indices_cur = torch.topk(energy_h, k=self.cdim, dim=-1)[1]
                mask_cur = F.one_hot(indices_cur, num_classes=self.hdim).float().sum(-2)
                mask_prev = mask_cur
                loss_h = None
            else:
                assert mask_prev is not None
                # intersection of prev and current hidden space
                indices_cur = torch.topk(energy_h, k=self.cdim, dim=-1)[1]
                mask_cur = F.one_hot(indices_cur, num_classes=self.hdim).float().sum(-2)
                mask_intersection = mask_prev * mask_cur
                seq_mask = seq_mask.prod(-1).unsqueeze(-1).repeat(1, 1, self.hdim).bool()
                # loss define
                h_ = h.clone()
                h_.retain_grad()
                loss_h = self.criterion_h(h_.view(-1, self.hdim),
                                          (1.0 - mask_intersection).view(-1, self.hdim),
                                          mask=seq_mask.view(-1, self.hdim),
                                          reduce=None)
                loss_h = loss_h.masked_fill(~(mask_intersection.view(-1, self.hdim).bool()), 0.0).sum()
                # eliminate fired hidden nodes
                h[mask_prev.bool()] = 0.0
                energy_h = h.pow(2)
                indices_cur = torch.topk(energy_h, k=self.cdim, dim=-1)[1]
                mask_cur = F.one_hot(indices_cur, num_classes=self.hdim).float().sum(-2)
                mask_prev = mask_prev + mask_cur
        else:
            pass
        h = h.masked_fill(~(mask_cur.bool()), 0.0)
        return h, mask_prev, loss_h

    def forward(self, x, y):
        # 0. prepare data
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(y)
            y = y.masked_fill(seq_mask, self.ignore_in)
        else:
            seq_mask = y == self.ignore_out

        buffs = {'y_dis': [], 'x_dis': [],
                 'theta_opt': [], 'sim_opt': [],
                 'loss': [], 'attn': [], 'hs': []}

        # iterative method for subtraction
        y_res = y.clone()
        x_res = x.clone()
        mask_prev_src = None
        mask_prev_self = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # 1. attention x_ele and y_res matching with transform for src disentangling
            x_aug, _ = pad_for_shift(key=x_res, pad=self.odim - 1,
                                     window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
            y_align_opt, sim_opt, theta_opt = sim_argmax(x_aug, y_res, measurement=self.measurement)
            attn = attention(y_align_opt, y_res, temper=self.temper)
            y_align_opt_attn = y_align_opt * attn
            x_ele = reverse_pad_for_shift(key=y_align_opt_attn, pad=self.odim - 1, window=self.odim, theta=theta_opt)

            if self.selftrain:
                # 1.1
                if self.encoder_type == 'conv1d':
                    b_size = x_ele.size(0)
                    t_size = x_ele.size(1)
                    x_ele, _ = pad_for_shift(key=x_ele,
                                             pad=self.input_extra,
                                             window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
                    h_self = self.encoder(x_ele).view(b_size * t_size, -1, self.hdim)  # (B * Tmax, *, hdim)
                    h_self_ind = torch.max(h_self.pow(2).sum(-1), dim=-1)[1]  # (B * Tmax)
                    h_self = h_self[torch.arange(h_self.size(0)), h_self_ind].view(b_size, t_size,
                                                                                   -1)  # (B, Tmax, hdim)
                    h_self, mask_prev_self, loss_h_self = self.hsr(h_self, mask_prev_self, seq_mask=seq_mask)
                    x_ele_ext = self.decoder_self(h_self).view(b_size * t_size, -1)
                    x_ele = torch.stack(
                        [x_ele_ext[torch.arange(x_ele_ext.size(0)), h_self_ind + i] for i in
                         six.moves.range(self.idim)],
                        dim=-1).view(b_size, t_size, -1)
                elif self.encoder_type == 'linear':
                    h_self = self.encoder(x_ele)  # (B * T, *, hdim)
                    h_self, mask_prev_self, loss_h_self = self.hsr(h_self, mask_prev_self, seq_mask=seq_mask)
                    x_ele = self.decoder_self(h_self)
                # 1.2
                move_energy = torch.abs(theta_opt - self.idim + 1).view(-1, 1) + 1.0
                move_mask = torch.abs(theta_opt - self.idim + 1).view(-1, 1) > self.energy_th
                loss_local_self = self.criterion(x_ele.view(-1, self.idim),
                                                 x_res.view(-1, self.idim),
                                                 mask=seq_mask.view(-1, self.idim), reduce=None)
                loss_local_self = loss_local_self.masked_fill(move_mask, 0.0) / move_energy
                # 1.3
                x_aug, _ = pad_for_shift(key=x_ele, pad=self.odim - 1,
                                         window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
                y_align_opt, sim_opt, theta_opt = sim_argmax(x_aug, y_res, measurement=self.measurement)
                # attn = attention(y_align_opt, y_res, temper=self.temper)
                y_align_opt_attn = y_align_opt  # * attn

            # 2. feedforward for src estimation
            if self.encoder_type == 'conv1d':
                b_size = y_align_opt.size(0)
                t_size = y_align_opt.size(1)
                y_align_opt_attn_pad, _ = pad_for_shift(key=y_align_opt_attn,
                                                        pad=self.input_extra,
                                                        window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
                h_src = self.encoder(y_align_opt_attn_pad).view(b_size * t_size, -1, self.hdim)  # (B * Tmax, *, hdim)
                h_src_ind = torch.max(h_src.pow(2).sum(-1), dim=-1)[1]  # (B * Tmax)
                h_src = h_src[torch.arange(h_src.size(0)), h_src_ind].view(b_size, t_size, -1)  # (B, Tmax, hdim)
                h_src, mask_prev_src, loss_h_src = self.hsr(h_src, mask_prev_src, seq_mask=seq_mask)
                y_ele_ext = self.decoder(h_src).view(b_size * t_size, -1)
                y_ele = torch.stack(
                    [y_ele_ext[torch.arange(y_ele_ext.size(0)), h_src_ind + i] for i in six.moves.range(self.odim)],
                    dim=-1).view(b_size, t_size, -1)
            elif self.encoder_type == 'linear':
                h_src = self.encoder(x_ele)  # (B * T, *, hdim)
                h_src, mask_prev_src, loss_h_src = self.hsr(h_src, mask_prev_src, seq_mask=seq_mask)
                y_ele = self.decoder(h_src)

            # 3. compute src estimation loss
            move_energy = torch.abs(theta_opt - self.odim + 1).view(-1, 1) + 1.0
            move_mask = torch.abs(theta_opt - self.odim + 1).view(-1, 1) > self.energy_th
            loss_local_src = self.criterion(y_ele.view(-1, self.odim),
                                            y_res.view(-1, self.odim),
                                            mask=seq_mask.view(-1, self.odim), reduce=None)
            loss_local_src = loss_local_src.masked_fill(move_mask, 0.0) / move_energy

            # 4. aggregate all loss
            if self.selftrain:
                if loss_h_src is not None:
                    loss = loss_local_src.sum() + loss_local_self.sum() + loss_h_src + loss_h_self
                else:
                    loss = loss_local_src.sum() + loss_local_self.sum()
            else:
                if loss_h_src is not None:
                    loss = loss_local_src.sum() + loss_h_src
                else:
                    loss = loss_local_src.sum()

            # 5. compute residual feature
            y_res = (y_res - y_ele).detach()
            x_res = (x_res - x_ele).detach()

            # buffering
            buffs['theta_opt'].append(theta_opt[0])
            buffs['sim_opt'].append(sim_opt[0])
            buffs['attn'].append(attn[0])
            buffs['x_dis'].append(x_ele)
            buffs['y_dis'].append(y_ele)
            buffs['loss'].append(loss)

        # 5. total loss compute
        loss = torch.stack(buffs['loss'], dim=-1).mean()
        self.reporter.report_dict['loss'] = float(loss)

        # appendix. for reporting some value or tensor
        # batch side sum needs for check
        x_dis = torch.stack(buffs['x_dis'], dim=-1)
        y_dis = torch.stack(buffs['y_dis'], dim=-1)
        loss_x = self.criterion(x_dis.sum(-1).view(-1, self.idim),
                                x.view(-1, self.idim),
                                mask=seq_mask.view(-1, self.odim))
        loss_y = self.criterion(y_dis.sum(-1).view(-1, self.idim),
                                y.view(-1, self.idim),
                                mask=seq_mask.view(-1, self.odim))
        self.reporter.report_dict['loss_x'] = float(loss_x)
        self.reporter.report_dict['loss_y'] = float(loss_y)
        self.reporter.report_dict['pred_y'] = y_dis.sum(-1)[0].detach().cpu().numpy()
        self.reporter.report_dict['pred_x'] = x_dis.sum(-1)[0].detach().cpu().numpy()
        self.reporter.report_dict['res_x'] = x_res[0].detach().cpu().numpy()
        self.reporter.report_dict['target'] = y[0].detach().cpu().numpy()

        # just one sample at batch should be check
        theta_opt = torch.stack(buffs['theta_opt'], dim=-1)
        sim_opt = torch.stack(buffs['sim_opt'], dim=-1)
        self.reporter.report_dict['theta_opt'] = theta_opt.detach().cpu().numpy()
        self.reporter.report_dict['sim_opt'] = sim_opt.detach().cpu().numpy()

        # disentangled hidden space check by attention disentangling
        attns = torch.stack(buffs['attn'], dim=-1)
        self.reporter.report_dict['attn0'] = attns[:, :, 0].detach().cpu().numpy()
        self.reporter.report_dict['attn1'] = attns[:, :, 1].detach().cpu().numpy()
        self.reporter.report_dict['attn2'] = attns[:, :, 2].detach().cpu().numpy()
        self.reporter.report_dict['attn3'] = attns[:, :, 3].detach().cpu().numpy()
        self.reporter.report_dict['attn4'] = attns[:, :, 4].detach().cpu().numpy()

        return loss
