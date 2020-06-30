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
        group.add_argument("--iter", default=1, type=int)

        # optimization related
        group.add_argument("--lr", default=0.001, type=float)
        group.add_argument("--momentum", default=0.9, type=float)

        # model related
        group.add_argument("--hdim", default=512, type=int)
        group.add_argument("--cdim", default=128, type=int)
        group.add_argument("--embed_dim_high", default=1000, type=int)
        group.add_argument("--embed_dim_low", default=50, type=int)

        # task related
        group.add_argument("--bias", default=1, type=int)
        group.add_argument("--embed_mem", default=0, type=int)
        group.add_argument("--brewing", default=0, type=int)
        group.add_argument("--eth", default=0.7, type=float)
        group.add_argument("--field_var", default=13.0, type=float)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        super().__init__()
        # network hyperparameter
        self.idim = idim - 3
        self.odim = idim - 3
        self.iter = args.iter
        self.tnum = args.tnum
        self.ignore_id = ignore_id
        self.subsample = [1]

        # additive task
        self.embed_mem = args.embed_mem
        self.brewing = args.brewing
        self.e_th = args.eth

        self.field_var = args.field_var
        space = np.linspace(0, self.idim - 1, self.idim)
        self.field = np.expand_dims(
            np.stack([gaussian_func(space, i, self.field_var) for i in range(self.idim)], axis=0), 0)
        self.field = torch.from_numpy(self.field / np.amax(self.field))

        # inference part with action and selection
        self.transform_f = Inference(idim=self.idim, odim=self.idim, args=args)

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

    @staticmethod
    def clustering(x, embed):
        sim_prob = torch.matmul(x, embed.weight.t()) / torch.norm(embed.weight.t(), dim=0, keepdim=True).unsqueeze(0)
        score_idx = torch.argmax(sim_prob, dim=-1)  # B, Tmax

        anchor = embed(score_idx)  # B, Tmax, d
        anchor = torch.softmax(anchor * x, dim=-1) * anchor

        return anchor, score_idx, sim_prob

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad):
        # prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens)]  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens)].transpose(1, 2)
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)

        # monitoring buffer
        self.buffs = {'score_idx_h': [],
                      'score_idx_l': [],
                      'out': [],
                      'seq_energy': None,
                      'kernel': None}

        # non-linearity network training
        xs_pad_out_hat, h, ratio_enc, ratio_dec, kernel = self.transform_f(xs_pad_in.unsqueeze(1))  # B, 1, Tmax, idim

        if self.eval:
            self.buffs['out'].append(xs_pad_out_hat)
            self.buffs['kernel'] = kernel

        # compute loss of total network
        masks = [seq_mask.unsqueeze(1).repeat(1, self.tnum, 1).view(-1, 1)]
        if self.brewing:
            # brew deep neural network model
            with torch.no_grad():
                p_hat = self.transform_f.brew([ratio_enc, ratio_dec])
                w_hat = p_hat[0]
                b_hat = p_hat[1]

                # calculate movement of each node
                move_flow = (torch.abs(w_hat) * (1 - self.field.to(w_hat.device))).mean(-1).mean(-1).unsqueeze(-1)

                # relation with sign
                sign_pair = torch.matmul(torch.sign(h.view(-1, self.hdim).unsqueeze(-1)),
                                         torch.sign(xs_pad_out.contiguous().view(-1, self.odim).unsqueeze(
                                             -2) - b_hat.unsqueeze(-2)))
                w_hat_x = w_hat * sign_pair
                w_hat_x_p = torch.relu(w_hat_x)
                w_hat_x_n = torch.relu(-w_hat_x)
                time_flow = (w_hat_x_p.sum(-1).sum(-1) / w_hat_x_n.sum(-1).sum(-1)).unsqueeze(-1)
                time_flow[torch.isnan(time_flow)] = 0.0

                seq_energy_mask = move_flow / (torch.abs(time_flow - 1.0) + 1e-6) * torch.abs(
                    h.view(-1, self.hdim)).mean(-1).unsqueeze(-1)
                seq_energy_mask = torch.clamp(seq_energy_mask, 0.0, 0.5)
                # seq_energy_mask = torch.log(seq_energy_mask + 1e-6)
                # seq_energy_mask[torch.isinf(seq_energy_mask)] = 10.0

                e_loss = seq_energy_mask.mean()

                discontinuity = (seq_energy_mask < self.e_th).float().mean()
                # masks.append(seq_energy_mask)

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

        if self.embed_mem:
            bsz, tnsz, tsz, csz = anchors.size()
            self.buffs['seq_energy'] = seq_energy_mask.view(bsz, tnsz, tsz)[:, :,
                                       :400]  # 1 - seq_energy_mask[:, 0, :].float()

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
            self.forward(xs_pad_in, xs_pad_out, ilens, ys_pad)

        ret = dict()

        ret['kernel'] = self.buffs['kernel'].cpu().numpy()
        if self.embed_mem:
            for i in range(self.tnum):
                ret['seq_energy_{}'.format(i)] = self.buffs['seq_energy'][:, i, :].cpu().numpy()

        ret['out'] = self.buffs['out'][0].cpu().numpy()
        return ret
