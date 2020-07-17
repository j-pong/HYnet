#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import chainer
from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from moneynet.nets.pytorch_backend.unsup.initialization import initialize
from moneynet.nets.pytorch_backend.unsup.loss import SeqMultiMaskLoss
from moneynet.nets.pytorch_backend.unsup.inference import Inference

from moneynet.nets.pytorch_backend.unsup.plot import PlotImageReport


def gaussian_func(x, m=0.0, sigma=1.0):
    norm = np.sqrt(2 * np.pi * sigma ** 2)
    dist = (x - m) ** 2 / (2 * sigma ** 2)
    return 1 / norm * np.exp(-dist)


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, buffs):
        """Report at every step."""
        for buff_key in buffs.keys():
            reporter.report({buff_key: buffs[buff_key]}, self)


class NetTransform(nn.Module):
    @staticmethod
    def add_arguments(parser):
        """Add arguments"""
        group = parser.add_argument_group("simnn setting")
        # task related
        group.add_argument("--tnum", default=10, type=int)

        # optimization related
        group.add_argument("--lr", default=0.001, type=float)
        group.add_argument("--momentum", default=0.9, type=float)
        group.add_argument("--warmup-steps", default=25000, type=float)

        # model related
        group.add_argument("--hdim", default=512, type=int)
        group.add_argument("--bias", default=1, type=int)

        # task related
        group.add_argument("--field-var", default=13.0, type=float)
        group.add_argument("--target-type", default='residual', type=str)
        group.add_argument("--cutting_gauge", default=0, type=int)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        super().__init__()
        # network hyperparameter
        self.idim = idim - 3
        self.odim = idim - 3
        self.hdim = args.hdim
        self.tnum = args.tnum
        self.ignore_id = ignore_id
        self.subsample = [1]

        # related task
        self.target_type = args.target_type
        self.cutting_gauge = args.cutting_gauge
        ## instance task
        self.task1 = False
        self.task2 = False
        self.task3 = False
        self.task4 = True

        # self.field_var = args.field_var
        # space = np.linspace(0, self.odim - 1, self.odim)
        # self.field = np.expand_dims(
        #     np.stack([gaussian_func(space, i, self.field_var) for i in range(self.odim)], axis=0), 0)
        # self.field = torch.from_numpy(2.0 - self.field / np.amax(self.field))

        # inference part with action and selection
        self.engine = Inference(idim=self.idim, odim=self.idim, args=args)

        # network training related
        self.criterion = SeqMultiMaskLoss(criterion=nn.MSELoss(reduction='none'))
        self.criterion_kernel = SeqMultiMaskLoss(criterion=nn.BCEWithLogitsLoss(reduction='none'))

        # reporter for monitoring
        self.reporter = Reporter()
        self.reporter_buffs = {}

        # initialize parameter
        initialize(self)

    @staticmethod
    def mvn(x):
        m = torch.mean(x, dim=-1, keepdim=True)
        v = torch.mean(torch.pow(x - m, 2), dim=-1, keepdim=True)
        x = (x - m) / v
        return x

    @staticmethod
    def minimaxn(x):
        max_x = torch.max(x)
        min_x = torch.min(x)
        if (max_x - min_x) == 0.0:
            logging.warning('Divided by the zero with max-min value : {}, Thus return None'.format((max_x - min_x)))
            x = None
        else:
            x = (x - min_x) / (max_x - min_x)

        return x

    @staticmethod
    def fb(x, mode='batch'):
        # TODO(j-pong): multi-target energy
        with torch.no_grad():
            if mode == 'batch':
                x = x[:, 0]
                bsz, tsz = x.size()
                xs = x.unsqueeze(-1).repeat(1, 1, tsz)
                ret = []
                for x in xs:
                    ones = torch.ones(tsz, tsz).to(x.device)

                    m_f = torch.triu(ones, diagonal=1)
                    x_f = torch.cumprod(torch.tril(x) + m_f, dim=-2)
                    x_f = torch.cat([ones[0:1], x_f[:-1]], dim=0) - m_f

                    ret.append(x_f)
                xs = torch.stack(ret, dim=0).unsqueeze(1)  # B, 1, T, T
            else:
                xs = []
                for i in range(x.size(-1)):
                    if i != 0:
                        x_f = torch.cumprod(x[:, :, i:], dim=-1)
                        x_b = torch.cumprod(x[:, :, :i].flip(dims=[-1]), dim=-1).flip(dims=[-1])
                        xs.append(torch.cat([x_b, x_f], dim=-1))
                    else:
                        xs.append(torch.cumprod(x[:, :, :], dim=-1))
                xs = torch.stack(xs, dim=-1)  # B, tnum, T, T

        if torch.isnan(xs.sum()):
            raise ValueError("kernel target value has nan")

        return xs

    def clustering(self, xs):
        """
        xs: (B, tnum, Tmax)

        return: (B, 1, Tmax)
        """
        xs = 1.0 - xs
        with torch.no_grad():
            ms = []
            for x in xs:
                if self.tnum == 1:
                    num_neg = int(x[0].sum(-1))
                    indices = torch.topk(x, k=num_neg, dim=-1)[-1]
                else:
                    raise AttributeError('Number of the target > 1 is not support yet!')
                m = torch.zeros_like(x[0])
                m[indices] = 1.0
                ms.append(m.unsqueeze(0).bool())
            ms = torch.stack(ms, dim=0)

        return ms

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad, buffering=False):
        # 0. prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens)].unsqueeze(1)  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens)].transpose(1, 2)
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)
        seq_mask_kernel = (1 - torch.matmul((~seq_mask).unsqueeze(-1).float(),
                                            (~seq_mask).unsqueeze(1).float())).bool()
        tsz = seq_mask_kernel.size(1)

        xs_pad_in_m = torch.mean(xs_pad_in, dim=-1, keepdim=True)
        xs_pad_in_v = torch.mean(torch.pow(xs_pad_in - xs_pad_in_m, 2), dim=-1, keepdim=True)

        # initialization buffer
        self.reporter_buffs = {}

        # 1. transform for checking similarity
        xs_pad_out_hat, ratio_t = self.engine(xs_pad_in, self.engine.transform)
        if self.target_type == 'mvn':
            xs_pad_out_hat_ = self.mvn(xs_pad_out_hat)
            xs_pad_out_hat_ = xs_pad_out_hat_ * xs_pad_in_v + xs_pad_in_m
        elif self.target_type == 'residual':
            xs_pad_out_hat_ = xs_pad_out_hat + xs_pad_in

        loss_g = self.criterion(xs_pad_out_hat_.view(-1, self.idim),
                                xs_pad_out.contiguous().view(-1, self.idim),
                                [seq_mask.view(-1, 1)],
                                reduction='none')

        # 2. calculate energy of sequence
        # Todo(j-pong): one step similarity test
        flows = self.calculate_energy(start_state=xs_pad_in.contiguous(), end_state=xs_pad_out_hat.contiguous(),
                                      func=self.engine.transform,
                                      ratio=ratio_t,
                                      flows={'e': None})
        energy_t = self.minimaxn(flows['e']).detach()

        # 3. embedding task
        h, ratio_e = self.engine(xs_pad_in, self.engine.embed)  # B, 1, Tmax, idim
        # if self.brewing:
        #     flows = self.calculate_energy(start_state=xs_pad_in.contiguous(), end_state=h.contiguous(),
        #                                   func=self.engine.embed,
        #                                   ratio=ratio_e,
        #                                   flows={'e': None})
        #     energy_e = flows['e']

        # get energy group mask

        # clustering with energy
        energy_t_high_mask = self.clustering(energy_t)
        # causal assumption
        if self.cutting_gauge:
            energy_t[energy_t_high_mask] = 0.0
        energy_t_cumprod = self.fb(energy_t)

        # calculate similarity matrix
        h_agg = torch.matmul(energy_t_cumprod, h) / \
                energy_t_cumprod.sum(-1, keepdim=True)  # [B, 1, T', T] x [B, 1, T, hdim] -> B, 1, T', hdim
        h_agg = h_agg.transpose(-2, -1)
        # h_agg = h.transpose(-2, -1)
        kernel = torch.matmul(h, h_agg.detach()) + torch.matmul(h.detach(),
                                                                h_agg)  # [B, 1, T, hdim] x [B, 1, hdim, T'] -> B, 1, T, T'
        kernel = kernel / 2.0 / np.sqrt(h.size(-1))
        if torch.isnan(kernel.sum()):
            raise ValueError("kernel value has nan")

        if self.task1:
            # large segmentation
            loss_e = self.criterion_kernel(kernel.view(-1, tsz, tsz),
                                           energy_t_cumprod.view(-1, tsz, tsz),
                                           [seq_mask_kernel, ~(energy_t_cumprod.view(-1, tsz, tsz) > 0.0)],
                                           reduction='none')
        elif self.task2:
            # wav2vec
            prediction = kernel.view(-1, tsz, tsz).diagonal(offset=1, dim1=-2, dim2=-1)  # B * tnum, (Tmax - offset)

            if prediction.size(-1) != tsz:
                seq_mask = seq_mask[:, :prediction.size(-1)].contiguous()
                energy_t_high_mask = energy_t_high_mask[:, :, :prediction.size(-1)].contiguous()
            neg_target = energy_t_high_mask.float().view(-1, 1)

            lam = neg_target.sum() / (1 - neg_target).sum()
            loss_e = lam * self.criterion_kernel(prediction.reshape(-1, 1),
                                                 (1 - neg_target).view(-1, 1),
                                                 [seq_mask.view(-1, 1), energy_t_high_mask.view(-1, 1)],
                                                 reduction='none') + \
                     self.criterion_kernel(-prediction.reshape(-1, 1),
                                           neg_target.view(-1, 1),
                                           [seq_mask.view(-1, 1), ~energy_t_high_mask.view(-1, 1)],
                                           reduction='none')
        elif self.task3:
            # wav2vec no negative
            prediction = kernel.view(-1, tsz, tsz).diagonal(offset=1, dim1=-2, dim2=-1)  # B * tnum, (Tmax - offset)
            if prediction.size(-1) != tsz:
                seq_mask = seq_mask[:, :prediction.size(-1)].contiguous()
            neg_target = energy_t_high_mask.float().view(-1, 1)
            loss_e = self.criterion_kernel(prediction.reshape(-1, 1),
                                           (1 - neg_target).view(-1, 1),
                                           [seq_mask.view(-1, 1)],
                                           reduction='none')
        elif self.task4:
            # small segmentation
            prediction = kernel.view(-1, tsz, tsz).diagonal(offset=1, dim1=-2, dim2=-1)  # B * tnum, (Tmax - offset)
            if prediction.size(-1) != tsz:
                seq_mask = seq_mask[:, :prediction.size(-1)].contiguous()
                energy_t = energy_t[:, :, :prediction.size(-1)].contiguous()
            loss_e = self.criterion_kernel(prediction.reshape(-1, 1),
                                           energy_t.view(-1, 1),
                                           [seq_mask.view(-1, 1)],
                                           reduction='none')

        # summary all task
        loss = loss_g.sum() + loss_e.sum()
        if not torch.isnan(loss):
            self.reporter.report({'loss': float(loss),
                                  'loss_g': float(loss_g.sum()),
                                  'loss_e': float(loss_e.sum())})
        else:
            logging.warning("loss (=%f) is not correct", float(loss))

        if buffering:
            self.reporter_buffs['out'] = xs_pad_out_hat_
            self.reporter_buffs['kernel'] = torch.sigmoid(kernel)

            # self.reporter_buffs['energy_e'] = energy_e[:, 0]
            self.reporter_buffs['energy_t'] = energy_t[:, 0]
            self.reporter_buffs['energy_t_cumprod'] = energy_t_cumprod[:, 0]

        return loss

    def calculate_energy(self, start_state, end_state, func, ratio, flows={'e': None}, split_dim=None):
        with torch.no_grad():
            # prepare data
            p_hat = self.engine.brew_(module_list=func, ratio=ratio, split_dim=split_dim)
            w_hat = p_hat[0]
            b_hat = p_hat[1]

            start_dim = start_state.size(-1)
            bsz, tnsz, tsz, end_dim = end_state.size()
            start_state = start_state.view(-1, start_dim)
            end_state = end_state.view(-1, end_dim) - b_hat

            # make data-pair field for tracing positive-negative
            sign_pair = torch.matmul(torch.sign(start_state.unsqueeze(-1)),
                                     torch.sign(end_state.unsqueeze(-2)))
            w_hat_x = w_hat * sign_pair
            w_hat_x_p = torch.relu(w_hat_x)
            w_hat_x_n = torch.relu(-w_hat_x)

            # relation with sign : how much state change in amplitude space
            energy = (torch.abs(start_state) * w_hat_x_n.sum(-1)).mean(-1, keepdim=True) / \
                     (torch.abs(start_state) * w_hat_x_p.sum(-1)).mean(-1, keepdim=True)
            num_nan = torch.isnan(energy).float().mean()
            if num_nan > 0.1:
                logging.warning("Energy sequence impose the nan {}".format(num_nan))
            energy[torch.isnan(energy)] = 1.0

            if self.target_type == 'mvn':
                energy = torch.pow(energy, 2)
            elif self.target_type == 'residual':
                energy = 1.0 - torch.pow(1.0 - energy, 2)

            # calculate energy
            for key in flows.keys():
                if key == 'e':
                    flows[key] = energy.view(bsz, tnsz, tsz)
                else:
                    raise AttributeError("'{}' type of augmentation factor is not defined!".format(key))

        return flows

    """
    Evaluation related
    """

    @property
    def images_plot_class(self):
        return PlotImageReport

    def calculate_images(self, xs_pad_in, xs_pad_out, ilens, ys_pad):
        with torch.no_grad():
            self.forward(xs_pad_in, xs_pad_out, ilens, ys_pad, buffering=True)

        dump = dict()
        for key in self.reporter_buffs.keys():
            dump[key] = self.reporter_buffs[key].cpu().numpy()

        return dump

    def recognize(self, x):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        return x
