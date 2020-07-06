#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import six

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

    def report(self, loss, e_loss, discontinuity):
        """Report at every step."""
        reporter.report({"loss": loss}, self)
        reporter.report({"e_loss": e_loss}, self)
        reporter.report({"discontinuity": discontinuity}, self)


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
        self.field_var = args.field_var
        space = np.linspace(0, self.odim - 1, self.odim)
        self.field = np.expand_dims(
            np.stack([gaussian_func(space, i, self.field_var) for i in range(self.odim)], axis=0), 0)
        self.field = torch.from_numpy(self.field / np.amax(self.field))

        # inference part with action and selection
        self.engine = Inference(idim=self.idim, odim=self.idim, args=args)

        # network training related
        self.criterion = SeqMultiMaskLoss(criterion=nn.MSELoss(reduction='none'))

        # reporter for monitoring
        self.reporter = Reporter()

        # initialize parameter
        initialize(self)

    @staticmethod
    def max_variance(p, dim=-1):
        mean = torch.max(p, dim=dim, keepdim=True)[0]  # (B, T, 1)
        # get normalized confidence
        numer = (mean - p).pow(2)
        denom = p.size(dim) - 1
        return torch.sum(numer, dim=-1) / denom

    def calculate_energy(self, start_state, end_state, func, ratio):
        with torch.no_grad():
            # prepare data
            p_hat = self.engine.brew_(module_list=func, ratio=ratio)
            w_hat = p_hat[0]
            b_hat = p_hat[1]

            start_dim = start_state.size(-1)
            bsz, tnsz, tsz, end_dim = end_state.size()
            start_state = start_state.view(-1, start_dim)
            end_state = end_state.view(-1, end_dim) - b_hat

            # calculate movement of each node
            move_flow = (torch.abs(w_hat) * (1 - self.field.to(w_hat.device))).mean(-1).mean(-1).unsqueeze(-1)

            # relation with sign
            sign_pair = torch.matmul(torch.sign(start_state.unsqueeze(-1)),
                                     torch.sign(end_state.unsqueeze(-2)))
            w_hat_x = w_hat * sign_pair
            w_hat_x_p = torch.relu(w_hat_x)
            w_hat_x_n = torch.relu(-w_hat_x)
            time_flow = (w_hat_x_p.sum(-1).sum(-1) / w_hat_x_n.sum(-1).sum(-1)).unsqueeze(-1)
            time_flow[torch.isnan(time_flow)] = 0.0

            # calculate energy
            seq_energy_mask = move_flow / (torch.abs(time_flow - 1.0) + 1e-6) * \
                              torch.abs(start_state).mean(-1).unsqueeze(-1)
            seq_energy_mask = torch.clamp(seq_energy_mask, 0.0, 0.5)

            return seq_energy_mask.view(bsz, tnsz, tsz)

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad, buffering=False):
        # prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens)]  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens)].transpose(1, 2)
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)

        # monitoring buffer
        self.buffs = {'out': [],
                      'seq_energy': None,
                      'kernel': None}

        masks = [seq_mask.unsqueeze(1).repeat(1, self.tnum, 1).view(-1, 1)]
        # non-linearity network training
        h, _ = self.engine(xs_pad_in.unsqueeze(1), self.engine.encoder)  # B, 1, Tmax, idim -> B, tnum, Tmax, idim
        kernel = torch.matmul(h, h.transpose(-2, -1))  # [B, T, T]
        h = h.repeat(1, self.tnum, 1, 1)
        xs_pad_out_hat, ratio = self.engine(h, self.engine.decoder)

        if buffering:
            self.buffs['out'].append(xs_pad_out_hat)
            self.buffs['kernel'] = kernel

        # compute loss of total network
        if self.brewing:
            seq_energy_mask = self.calculate_energy(start_state=h, end_state=xs_pad_out.contiguous(),
                                                    func=self.engine.decoder, ratio=ratio)

            e_loss = seq_energy_mask.mean()
            discontinuity = 0.0

            if buffering:
                self.buffs['seq_energy'] = seq_energy_mask[:, :, :400]

        else:
            e_loss = 0.0
            discontinuity = 0.0

        # compute loss and filtering
        loss = self.criterion(xs_pad_out_hat.view(-1, self.idim),
                              xs_pad_out.contiguous().view(-1, self.idim),
                              masks).mean()
        if not torch.isnan(loss) or not torch.isnan(e_loss):
            self.reporter.report(float(loss), float(e_loss), float(discontinuity))
        else:
            print("loss (=%f) is not c\orrect", float(loss))
            logging.warning("loss (=%f) is not correct", float(loss))

        return loss

    def recognize(self, x):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        return x

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
            ret['seq_energy_{}'.format(0)] = self.buffs['seq_energy'][:, 0, :].cpu().numpy()
        return ret
