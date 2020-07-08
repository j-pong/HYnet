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
    def fb(x):
        xs = []
        for i in range(x.size(-1)):
            if i != 0:
                x_f = torch.cumsum(x[:, :, i:], dim=-1)
                x_b = torch.cumsum(x[:, :, :i].flip(dims=[-1]), dim=-1).flip(dims=[-1])
                xs.append(torch.cat([x_b, x_f], dim=-1))
            else:
                xs.append(torch.cumsum(x[:, :, :], dim=-1))
        xs = torch.stack(xs, dim=-1)  # B, tnum, T, T
        norm = torch.max(xs)
        if  norm == 0.0:
            raise ValueError('Energy maximum value is {}'.format(torch.max(xs)))
        else:
            xs = 1.0 - xs / norm

        return xs

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
            energy[torch.isnan(energy)] = 0.0
            # energy[torch.isinf(energy)] = 0.0
            energy = 1.0 - energy

            # calculate energy
            for key in flows.keys():
                if key == 'e':
                    flows[key] = torch.relu(energy).view(bsz, tnsz, tsz)
                else:
                    raise AttributeError("'{}' type of augmentation factor is not defined!".format(key))

        return flows

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad, buffering=False):
        # prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens)].unsqueeze(1)  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens)].transpose(1, 2)
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)
        masks = [seq_mask.unsqueeze(1).repeat(1, self.tnum, 1).view(-1, 1)]

        xs_pad_in_m = torch.mean(xs_pad_in, dim=-1, keepdim=True)
        xs_pad_in_v = torch.mean(torch.pow(xs_pad_in - xs_pad_in_m, 2), dim=-1, keepdim=True)
        xs_pad_out_m = torch.mean(xs_pad_out, dim=-1, keepdim=True)
        xs_pad_out_v = torch.mean(torch.pow(xs_pad_out - xs_pad_out_m, 2), dim=-1, keepdim=True)

        # monitoring buffer
        self.buffs = {'out': [],
                      'seq_energy': None,
                      'kernel': None}

        # embedding task
        h, ratio = self.engine(xs_pad_in, self.engine.encoder)  # B, 1, Tmax, idim
        if self.brewing:
            flows = self.calculate_energy(start_state=xs_pad_in.contiguous(), end_state=h,
                                          func=self.engine.encoder,
                                          ratio=ratio,
                                          flows={'e': None})  # B, Tmax, 1
            energy = flows['e']
            if buffering:
                self.buffs['seq_energy_enc'] = energy[:, :, :400]
        h = self.mvn(h)
        h = h * xs_pad_in_v + xs_pad_in_m

        # evaluate hidden space similarity
        kernel = torch.matmul(h, h.transpose(-2, -1))  # B, 1, T, T

        # long time prediction task
        h = h.repeat(1, self.tnum, 1, 1)
        xs_pad_out_hat, ratio = self.engine(h, self.engine.decoder)
        if self.brewing:
            flows = self.calculate_energy(start_state=h, end_state=xs_pad_out.contiguous(),
                                          func=self.engine.decoder,
                                          ratio=ratio,
                                          flows={'e': None})  # B, Tmax, 1
            energy = flows['e']
            kernel_target = self.fb(energy)
            if buffering:
                self.buffs['seq_energy_dec'] = kernel_target[:, :, :400, :400]

            # compute loss of hidden space
            seq_mask_kernel = (1 - torch.matmul((~seq_mask).unsqueeze(-1).float(),
                                                (~seq_mask).unsqueeze(1).float())).bool()
            tsz = seq_mask_kernel.size(1)
            loss_e = self.criterion_h(input=torch.sigmoid(kernel), target=kernel_target[:, 0:1].detach())
            loss_e = loss_e.view(-1, tsz, tsz).masked_fill(seq_mask_kernel, 0).mean()
            if torch.isnan(kernel_target.sum()):
                print("target nan")
            if torch.isnan(kernel.sum()):
                print("output nan")
            discontinuity = 0.0
        else:
            loss_e = 0.0
            discontinuity = 0.0
        xs_pad_out_hat = self.mvn(xs_pad_out_hat)
        xs_pad_out_hat = xs_pad_out_hat * xs_pad_out_v + xs_pad_out_m
        if buffering:
            self.buffs['out'].append(xs_pad_out_hat)
            self.buffs['kernel'] = kernel

        # compute loss of total network
        loss_g = self.criterion(xs_pad_out_hat.view(-1, self.idim),
                                xs_pad_out.contiguous().view(-1, self.idim),
                                masks).mean()
        loss = loss_g + loss_e
        if not torch.isnan(loss):
            self.reporter.report({'loss': float(loss),
                                  'e_loss': float(loss_e),
                                  'g_loss': float(loss_g),
                                  'discontinuity': float(discontinuity)})
        else:
            print("loss (=%f) is not c\orrect", float(loss))
            logging.warning("loss (=%f) is not correct", float(loss))

        return loss

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
        ret['out'] = self.buffs['out'][0].cpu().numpy()
        if self.brewing:
            ret['seq_energy_enc'] = self.buffs['seq_energy_enc'][:, 0, :].cpu().numpy()
            ret['seq_energy_dec'] = self.buffs['seq_energy_dec'][:, 0, :].cpu().numpy()
        return ret

    def recognize(self, x):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        return x
