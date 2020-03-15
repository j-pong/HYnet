#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch

from torch import nn

import numpy as np

from moneynet.nets.unsup.utils import pad_for_shift, reverse_pad_for_shift, selector, select_with_ind
from moneynet.nets.unsup.attention import attention
from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqLoss, SeqEnergyLoss

from moneynet.nets.unsup.disentangling import Disentangling


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

        # disentangling network
        self.disentangle = Disentangling(idim, odim, args)

        # training hyperparameter
        self.measurement = args.similarity
        self.temper = args.temperature
        self.energy_th = args.energy_threshold

        # network training related
        self.criterion = SeqLoss()
        self.criterion_energy = SeqEnergyLoss()

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

            if not self.training:
                buffs['theta_opt'].append(theta_opt[0])
                buffs['sim_opt'].append(sim_opt[0])
                buffs['attn'].append(attn[0])

            # self 4. reverse action
            x_align_opt_attn = reverse_pad_for_shift(key=y_align_opt_attn, pad=self.odim - 1, window=self.odim,
                                                     theta=theta_opt)

            if self.selftrain:
                # self 5 inference
                x_ele, mask_prev_self, loss_h_self = self.disentangle(x_align_opt_attn, mask_prev_self, seq_mask,
                                                                      theta=theta_opt,
                                                                      decoder='self')
                x_ele = x_ele + x_align_opt_attn

                # self 6 loss
                loss_local_self = self.criterion_energy(x_ele.view(-1, self.idim),
                                                        x_res.view(-1, self.idim),
                                                        seq_mask.view(-1, self.odim),
                                                        self.idim, theta_opt, self.energy_th)

                # source 1.action with inference feature that concern relation of pixel of frame
                x_aug, _ = pad_for_shift(key=x_ele, pad=self.odim - 1,
                                         window=self.odim)  # (B, Tmax, idim_k + idim_q - 1, idim_q)

                # source 2. selection of action
                y_align_opt, sim_opt, theta_opt = selector(x_aug, y_res, measurement=self.measurement)
                y_align_opt_attn = y_align_opt + y_align_opt_attn
            else:
                x_ele = x_align_opt_attn

            # source 3. inference
            y_ele, mask_prev_src, loss_h_src = self.disentangle(y_align_opt_attn, mask_prev_src, seq_mask,
                                                                theta=theta_opt,
                                                                decoder='src')

            # source 4. loss
            loss_local_src = self.criterion_energy(y_ele.view(-1, self.odim),
                                                   y_res.view(-1, self.odim),
                                                   seq_mask.view(-1, self.odim),
                                                   self.odim, theta_opt, self.energy_th)

            # aggregate all loss
            if self.selftrain:
                if loss_h_src is not None:
                    loss = loss_local_src + loss_local_self + loss_h_src + loss_h_self
                else:
                    loss = loss_local_src + loss_local_self
            else:
                if loss_h_src is not None:
                    loss = loss_local_src + loss_h_src
                else:
                    loss = loss_local_src

            # compute residual feature
            y_res = (y_res - y_ele).detach()
            x_res = (x_res - x_ele).detach()

            # buffering
            if not self.training:
                # buffs['theta_opt'].append(theta_opt[0])
                # buffs['sim_opt'].append(sim_opt[0])
                # buffs['attn'].append(attn[0])
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
