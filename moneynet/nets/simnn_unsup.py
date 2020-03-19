#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F

from torch import nn

import numpy as np

from moneynet.nets.unsup.utils import pad_for_shift, reverse_pad_for_shift, selector, select_with_ind
from moneynet.nets.unsup.attention import attention
from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqLoss


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

    def disentangle(self, x, mask_prev, seq_mask, decoder):
        if self.encoder_type == 'conv1d':
            x, _ = pad_for_shift(key=x, pad=self.input_extra,
                                 window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
            h = self.encoder(x)  # (B, Tmax, *, hdim)
            # max pooling along shift size
            h_ind = torch.max(h.pow(2).sum(-1), dim=-1)[1]  # (B, Tmax)
            h = select_with_ind(h, h_ind)  # (B, Tmax, hdim)
            # hidden space regularization
            h, mask_prev, loss_h = self.hsr(h, mask_prev, seq_mask=seq_mask)
            # target trunk along feature side with window
            assert self.idim == self.odim
            x_ext = decoder(h)
            x = torch.stack(
                [select_with_ind(x_ext, x_ext.size(-1) - 1 - h_ind - i) for i in torch.arange(self.idim).flip(0)],
                dim=-1)
        elif self.encoder_type == 'linear':
            h = self.encoder(x)
            h, mask_prev, loss_h = self.hsr(h, mask_prev, seq_mask=seq_mask)
            x = decoder(h)

        return x, mask_prev, loss_h

    def energy_loss(self, x, y, feat_dim, seq_mask, theta_opt):
        move_energy = torch.abs(theta_opt - feat_dim + 1).view(-1, 1) + 1.0
        move_mask = torch.abs(theta_opt - feat_dim + 1).view(-1, 1) > self.energy_th
        loss_local = self.criterion(x.view(-1, feat_dim),
                                    y.view(-1, feat_dim),
                                    mask=seq_mask.view(-1, feat_dim), reduce=None)
        return loss_local.masked_fill(move_mask, 0.0) / move_energy

    def forward(self, x, y):
        # prepare data
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(y)
            y = y.masked_fill(seq_mask, self.ignore_in)
        else:
            seq_mask = y == self.ignore_out
        buffs = {'y_dis': [], 'x_dis': [], 'x_res': [], 'y_res': [],
                 'theta_opt': [], 'sim_opt': [],
                 'loss': [], 'attn': [], 'hs': []}

        # start iteration for superposition
        y_res = y.clone()
        x_res = x.clone()
        mask_prev_src = None
        mask_prev_self = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # self 1. action for self feature (current version just pad action support)
            x_aug, _ = pad_for_shift(key=x_res, pad=self.odim - 1,
                                     window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)

            # self 2. selection of action
            y_align_opt, sim_opt, theta_opt = selector(x_aug, y_res, measurement=self.measurement)

            # self 3. attention to target feature with selected feature
            attn = attention(y_align_opt, y_res, temper=self.temper)
            y_align_opt_attn = y_align_opt * attn

            # self 4. reverse action
            x_ele = reverse_pad_for_shift(key=y_align_opt_attn, pad=self.odim - 1, window=self.odim, theta=theta_opt)

            if self.selftrain:
                # self 5 inference
                x_ele, mask_prev_self, loss_h_self = self.disentangle(x_ele, mask_prev_self, seq_mask,
                                                                      decoder=self.decoder_self)

                # self 6 loss
                loss_local_self = self.energy_loss(x_ele, x_res, self.idim, seq_mask, theta_opt)

                # source 1.action with inference feature that concern relation of pixel of frame
                x_aug, _ = pad_for_shift(key=x_ele, pad=self.odim - 1,
                                         window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)

                # source 2. selection of action
                y_align_opt, sim_opt, theta_opt = selector(x_aug, y_res, measurement=self.measurement)
                y_align_opt_attn = y_align_opt
            # 2. feedforward for src estimation
            y_ele, mask_prev_src, loss_h_src = self.disentangle(y_align_opt_attn, mask_prev_src, seq_mask,
                                                                decoder=self.decoder)
            # source 3. inference
            loss_local_src = self.energy_loss(y_ele, y_res, self.odim, seq_mask, theta_opt)

            # source 4. loss
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

            # compute residual feature
            y_res = (y_res - y_ele).detach()
            x_res = (x_res - x_ele).detach()

            # buffering
            if not self.training:
                buffs['theta_opt'].append(theta_opt[0])
                buffs['sim_opt'].append(sim_opt[0])
                buffs['attn'].append(attn[0])
                buffs['x_dis'].append(x_ele)
                buffs['y_dis'].append(y_ele)
                buffs['x_res'].append(x_res)
                buffs['y_res'].append(y_res)
            buffs['loss'].append(loss)

        # 5. total loss compute
        loss = torch.stack(buffs['loss'], dim=-1).mean()
        self.reporter.report_dict['loss'] = float(loss)

        if not self.training:
            # appendix. for reporting some value or tensor
            # batch side sum needs for check
            x_dis = torch.stack(buffs['x_dis'], dim=-1)
            y_dis = torch.stack(buffs['y_dis'], dim=-1)
            loss_x = self.criterion(x_dis.sum(-1).view(-1, self.idim),
                                    x.view(-1, self.idim),
                                    seq_mask.view(-1, self.odim))
            loss_y = self.criterion(y_dis.sum(-1).view(-1, self.idim),
                                    y.view(-1, self.idim),
                                    seq_mask.view(-1, self.odim))
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

            # # disentangled hidden space check by attention disentangling
            attns = torch.stack(buffs['attn'], dim=-1)
            self.reporter.report_dict['attn0'] = attns[:, :, 0].detach().cpu().numpy()
            self.reporter.report_dict['attn1'] = attns[:, :, 1].detach().cpu().numpy()
            self.reporter.report_dict['attn2'] = attns[:, :, 2].detach().cpu().numpy()
            self.reporter.report_dict['attn3'] = attns[:, :, 3].detach().cpu().numpy()
            self.reporter.report_dict['attn4'] = attns[:, :, 4].detach().cpu().numpy()

            self.reporter.report_dict['x_dis0'] = x_dis[0, :, :, 0].detach().cpu().numpy()
            self.reporter.report_dict['x_dis1'] = x_dis[0, :, :, 1].detach().cpu().numpy()
            self.reporter.report_dict['x_dis2'] = x_dis[0, :, :, 2].detach().cpu().numpy()
            self.reporter.report_dict['x_dis3'] = x_dis[0, :, :, 3].detach().cpu().numpy()
            self.reporter.report_dict['x_dis4'] = x_dis[0, :, :, 4].detach().cpu().numpy()

            self.reporter.report_dict['x_res0'] = buffs['x_res'][0][0].detach().cpu().numpy()
            self.reporter.report_dict['x_res1'] = buffs['x_res'][1][0].detach().cpu().numpy()
            self.reporter.report_dict['x_res2'] = buffs['x_res'][2][0].detach().cpu().numpy()
            self.reporter.report_dict['x_res3'] = buffs['x_res'][3][0].detach().cpu().numpy()
            self.reporter.report_dict['x_res4'] = buffs['x_res'][4][0].detach().cpu().numpy()

            # self.reporter.report_dict['y_res'] = buffs['y_res'][0][0].detach().cpu().numpy()
            # self.reporter.report_dict['y_res'] = buffs['y_res'][1][0].detach().cpu().numpy()
            # self.reporter.report_dict['y_res'] = buffs['y_res'][2][0].detach().cpu().numpy()
            # self.reporter.report_dict['y_res'] = buffs['y_res'][3][0].detach().cpu().numpy()
            # self.reporter.report_dict['y_res'] = buffs['y_res'][4][0].detach().cpu().numpy()

        return loss
