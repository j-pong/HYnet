#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
from torch import nn

import numpy as np

from moneynet.nets.unsup.utils import pad_for_shift, reverse_pad_for_shift, selector, select_with_ind
from moneynet.nets.unsup.attention import attention
from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqLoss


class InferenceNet(nn.Module):
    def __init__(self, idim, odim, args):
        super(InferenceNet, self).__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim

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
        x_ind = torch.max(energy, dim=-1)[1]  # (B, Tmax)
        x = select_with_ind(x, x_ind)  # (B, Tmax, hdim)
        return x, x_ind

    def forward(self, x, seq_mask, decoder_type):
        if self.encoder_type == 'conv1d':
            x, _ = pad_for_shift(key=x, pad=self.input_extra,
                                 window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
            h = self.encoder(x)  # (B, Tmax, *, hdim)
            # max pooling along shift size
            h, h_ind = self.energy_pooling(h)
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
            # feedforward decoder
            assert self.idim == self.odim
            if decoder_type == 'self':
                x = self.decoder_self(h)
            elif decoder_type == 'src':
                x = self.decoder_src(h)

        return x


class Net(nn.Module):
    def __init__(self, idim, odim, args, reporter):
        super(Net, self).__init__()
        # utils
        self.reporter = reporter

        # network hyperparameter
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out
        self.selftrain = args.self_train

        self.thinkiter = args.think_iter

        self.inference = InferenceNet(idim, odim, args)

        # training hyperparameter
        self.measurement = args.similarity
        self.temper = args.temperature
        self.energy_th = args.energy_threshold

        # network training related
        self.criterion = SeqLoss(criterion=nn.MSELoss(reduction='none'))

        # initialize parameter
        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    def forward(self, x, y):
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
        y_res = y.clone()
        x_res = x.clone()
        # mask_prev_src = None
        # mask_prev_self = None
        for _ in six.moves.range(self.thinkiter):
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
                x_ele = self.inference(x_ele, seq_mask,
                                       decoder_type='self')

                # self 6 loss
                loss_local_self = self.criterion(x_ele, x_res, seq_mask)

                # source 1.action with inference feature that concern relation of pixel of frame
                x_aug, _ = pad_for_shift(key=x_ele, pad=self.odim - 1,
                                         window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)

                # source 2. selection of action
                y_align_opt, sim_opt, theta_opt = selector(x_aug, y_res, measurement=self.measurement)
                y_align_opt_attn = y_align_opt
            # 2. feedforward for src estimation
            y_ele = self.inference(y_align_opt_attn, seq_mask,
                                   decoder_type='src')
            # source 3. inference
            loss_local_src = self.criterion(y_ele, y_res, seq_mask)

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
        self.reporter.report_dict['loss'] = float(loss)
        x_hyp = torch.stack(buffs['x_ele'], dim=-1)
        y_hyp = torch.stack(buffs['y_ele'], dim=-1)
        loss_x = self.criterion(x_hyp.sum(-1).view(-1, self.idim),
                                x.view(-1, self.idim),
                                seq_mask.view(-1, self.odim))
        loss_y = self.criterion(y_hyp.sum(-1).view(-1, self.idim),
                                y.view(-1, self.idim),
                                seq_mask.view(-1, self.odim))
        self.reporter.report_dict['loss_x'] = float(loss_x)
        self.reporter.report_dict['loss_y'] = float(loss_y)

        if not self.training:
            # appendix. for reporting some value or tensor
            # batch side sum needs for check
            self.reporter.report_dict['pred_x'] = x_hyp.sum(-1)[0].detach().cpu().numpy()
            self.reporter.report_dict['pred_y'] = y_hyp.sum(-1)[0].detach().cpu().numpy()
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

            self.reporter.report_dict['x_ele0'] = x_hyp[0, :, :, 0].detach().cpu().numpy()
            self.reporter.report_dict['x_ele1'] = x_hyp[0, :, :, 1].detach().cpu().numpy()
            self.reporter.report_dict['x_ele2'] = x_hyp[0, :, :, 2].detach().cpu().numpy()
            self.reporter.report_dict['x_ele3'] = x_hyp[0, :, :, 3].detach().cpu().numpy()
            self.reporter.report_dict['x_ele4'] = x_hyp[0, :, :, 4].detach().cpu().numpy()

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
