#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy as np

import torch
from torch import nn
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

        # model related
        group.add_argument("--hdim", default=512, type=int)
        group.add_argument("--bias", default=1, type=int)

        # task related
        group.add_argument("--brewing", default=0, type=int)
        group.add_argument("--field_var", default=13.0, type=float)

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
        self.brewing = args.brewing
        # self.field_var = args.field_var
        # space = np.linspace(0, self.odim - 1, self.odim)
        # self.field = np.expand_dims(
        #     np.stack([gaussian_func(space, i, self.field_var) for i in range(self.odim)], axis=0), 0)
        # self.field = torch.from_numpy(2.0 - self.field / np.amax(self.field))

        # inference part with action and selection
        self.engine = Inference(idim=self.idim, odim=self.idim, args=args)

        # network training related
        self.criterion = SeqMultiMaskLoss(criterion=nn.MSELoss(reduction='none'))
        self.criterion_h = nn.BCELoss(reduction='none')

        # monitoring buffer
        self.buffs = {'out': None,
                      'seq_energy': None,
                      'kernel': None}

        # reporter for monitoring
        self.reporter = Reporter()

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
            raise ValueError('Divided by the zero with max-min value : {}'.format((max_x - min_x)))
        else:
            x = (x - min_x) / (max_x - min_x)

        return x

    def fb(self, x, mode='batch'):
        if mode == 'batch':
            # TODO(j-pong): multi-target energy
            x = 1.0 - self.minimaxn(x[:, 0])
            bsz, tsz = x.size()
            xs = x.unsqueeze(-1).repeat(1, 1, tsz)
            ret = []
            for x in xs:
                m_f = torch.triu(torch.ones(tsz, tsz)).to(x.device)
                x_f = torch.cumprod(torch.tril(x) + m_f, dim=-2) - m_f
                m_b = torch.tril(torch.ones(tsz, tsz)).to(x.device)
                x_b = torch.cumprod((torch.triu(x) + m_b).flip(dims=[-2]), dim=-2).flip(dims=[-2]) - m_b
                ret.append(x_b + x_f)
            xs = torch.stack(ret, dim=0).unsqueeze(1) / 2  # B, 1, T, T
        else:
            x = 1.0 - self.minimaxn(x[:, 0:1])
            xs = []
            for i in range(x.size(-1)):
                if i != 0:
                    x_f = torch.cumprod(x[:, :, i:], dim=-1)
                    x_b = torch.cumprod(x[:, :, :i].flip(dims=[-1]), dim=-1).flip(dims=[-1])
                    xs.append(torch.cat([x_b, x_f], dim=-1))
                else:
                    xs.append(torch.cumprod(x[:, :, :], dim=-1))
            xs = torch.stack(xs, dim=-1)  # B, tnum, T, T

        return xs

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad, buffering=False):
        # 0. prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens)].unsqueeze(1)  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens)].transpose(1, 2)
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)

        xs_pad_in_m = torch.mean(xs_pad_in, dim=-1, keepdim=True)
        xs_pad_in_v = torch.mean(torch.pow(xs_pad_in - xs_pad_in_m, 2), dim=-1, keepdim=True)
        xs_pad_out_m = torch.mean(xs_pad_out, dim=-1, keepdim=True)
        xs_pad_out_v = torch.mean(torch.pow(xs_pad_out - xs_pad_out_m, 2), dim=-1, keepdim=True)

        # initialization buffer
        self.buffs = {'out': None,
                      'seq_energy': None,
                      'kernel': None}

        # 1. embedding task
        h, ratio = self.engine(xs_pad_in, self.engine.encoder)  # B, 1, Tmax, idim
        if self.brewing:
            flows = self.calculate_energy(start_state=xs_pad_in.contiguous(), end_state=h,
                                          func=self.engine.encoder,
                                          ratio=ratio,
                                          flows={'e': None})
            energy = flows['e']
            if buffering:
                self.buffs['seq_energy_enc'] = energy
        # normalization feature from input space mean-variance
        h = self.mvn(h)
        h = h * xs_pad_in_v + xs_pad_in_m
        # split for low-high decoder
        h_h = h[:, :, :, :self.odim]
        h_l = h[:, :, :, self.odim:].repeat(1, self.tnum, 1, 1)
        # calculate similarity matrix
        kernel = torch.sigmoid(torch.matmul(h_h, h_h.transpose(-2, -1)))  # B, 1, T, T
        if buffering:
            self.buffs['kernel'] = kernel

        # 2. decoder
        xs_pad_out_l_hat, ratio_l = self.engine(h_l, self.engine.decoder_low)
        xs_pad_out_h_hat, ratio_h = self.engine(h_h, self.engine.decoder_high)
        if self.brewing:
            flows = self.calculate_energy(start_state=h_l.contiguous(), end_state=xs_pad_out.contiguous(),
                                          func=self.engine.decoder_low,
                                          ratio=ratio_l,
                                          flows={'e': None})
            energy_d_l = flows['e']
            kernel_target_l = self.fb(energy_d_l)
            flows = self.calculate_energy(start_state=h_h.contiguous(), end_state=xs_pad_out[:, 0:1].contiguous(),
                                          func=self.engine.decoder_high,
                                          ratio=ratio_h,
                                          flows={'e': None})
            energy_d_h = flows['e']
            kernel_target_h = self.fb(energy_d_h)
            if buffering:
                self.buffs['seq_energy_dec'] = energy_d_h
                self.buffs['kernel_target_l'] = kernel_target_l
                self.buffs['kernel_target_h'] = kernel_target_h

            # # compute loss of hidden space
            # seq_mask_kernel = (1 - torch.matmul((~seq_mask).unsqueeze(-1).float(),
            #                                     (~seq_mask).unsqueeze(1).float())).bool()
            #
            # tsz = seq_mask_kernel.size(1)
            # loss_e = self.criterion_h(input=kernel, target=kernel_target.detach())
            # loss_e = loss_e.view(-1, tsz, tsz).masked_fill(seq_mask_kernel, 0).mean()
            # if torch.isnan(kernel_target.sum()):
            #     raise ValueError("kernel target value has nan")
            # if torch.isnan(kernel.sum()):
            #     raise ValueError("kernel value has nan")
            loss_e = 0.0
        else:
            loss_e = 0.0
        xs_pad_out_l_hat = self.mvn(xs_pad_out_l_hat)
        xs_pad_out_l_hat = xs_pad_out_l_hat * xs_pad_out_v[:, 0:1] + xs_pad_out_m[:, 0:1]
        xs_pad_out_h_hat = self.mvn(xs_pad_out_h_hat)
        xs_pad_out_h_hat = xs_pad_out_h_hat * xs_pad_out_v[:, 0:1] + xs_pad_out_m[:, 0:1]
        if buffering:
            self.buffs['out'] = xs_pad_out_h_hat

        # compute loss of total network
        loss_l_g = self.criterion(xs_pad_out_l_hat.view(-1, self.idim),
                                  xs_pad_out.contiguous().view(-1, self.idim),
                                  [seq_mask.unsqueeze(1).repeat(1, self.tnum, 1).view(-1, 1)],
                                  'none')
        if self.brewing:
            kernel_weights = []
            for i in range(self.tnum):
                kernel_weight = torch.diagonal(kernel_target_h, i, dim1=-2, dim2=-1)
                if i > 0:
                    padding = torch.zeros_like(kernel_weight[:, :, 0:i])
                    kernel_weight = torch.cat([kernel_weight, padding], dim=-1)
                kernel_weights.append(kernel_weight)
            kernel_weights = torch.cat(kernel_weights, dim=1)
            loss_l_g = loss_l_g * kernel_weights.view(-1, 1)
            loss_l_g = loss_l_g.sum()
        loss_h_g = self.criterion(xs_pad_out_h_hat.view(-1, self.idim),
                                  xs_pad_out[:, 0:1].contiguous().view(-1, self.idim),
                                  [seq_mask.view(-1, 1)]).sum()
        loss = loss_l_g + loss_h_g + loss_e
        if not torch.isnan(loss):
            self.reporter.report({'loss': float(loss),
                                  'e_loss': float(loss_e),
                                  'l_loss': float(loss_l_g),
                                  'h_loss': float(loss_h_g)})
        else:
            print("loss (=%f) is not c\orrect", float(loss))
            logging.warning("loss (=%f) is not correct", float(loss))

        return loss

    def calculate_energy(self, start_state, end_state, func, ratio, flows={'e': None}):
        with torch.no_grad():
            # prepare data
            p_hat = self.engine.brew_(module_list=func, ratio=ratio)
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
            energy[torch.isnan(energy)] = 1.0
            # energy[torch.isinf(energy)] = 0.0
            energy = 1.0 - energy

            # calculate energy
            for key in flows.keys():
                if key == 'e':
                    flows[key] = torch.relu(energy).view(bsz, tnsz, tsz)
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

        ret = dict()

        ret['kernel'] = self.buffs['kernel'].cpu().numpy()
        ret['out'] = self.buffs['out'].cpu().numpy()
        if self.brewing:
            ret['seq_energy_enc'] = self.buffs['seq_energy_enc'][:, 0, :].cpu().numpy()
            ret['seq_energy_dec'] = self.buffs['seq_energy_dec'][:, 0, :].cpu().numpy()
            ret['kernel_target_h'] = self.buffs['kernel_target_h'][:, 0].cpu().numpy()
            ret['kernel_target_l'] = self.buffs['kernel_target_l'][:, 0].cpu().numpy()
        return ret

    def recognize(self, x):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        return x
