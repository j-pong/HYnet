#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import six

import torch
from torch import nn

import chainer
from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list

from moneynet.nets.unsup.initialization import initialize
from moneynet.nets.unsup.loss import SeqMultiMaskLoss
from moneynet.nets.unsup.inference import InferenceNet


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss):
        """Report at every step."""
        reporter.report({"loss": loss}, self)


class Net(nn.Module):
    @staticmethod
    def add_arguments(parser):
        """Add arguments"""
        group = parser.add_argument_group("simnn setting")
        group.add_argument("--etype", default="conv1d", type=str)
        group.add_argument("--hdim", default=160, type=int)
        group.add_argument("--cdim", default=16, type=int)
        group.add_argument("--tnum", default=10, type=int)
        group.add_argument("--lr", default=0.001, type=float)
        group.add_argument("--momentum", default=0.9, type=float)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        super(Net, self).__init__()
        # network hyperparameter
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim
        self.tnum = args.tnum
        self.ignore_id = ignore_id
        self.subsample = [1]

        # reporter for monitoring
        self.reporter = Reporter()

        # inference part with action and selection
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
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        seq_mask = make_pad_mask((ilens - self.tnum).tolist()).to(xs_pad.device)

        xs_pad_s = xs_pad
        xs = []
        for x_pad, ilen in zip(xs_pad_s, ilens):
            xs.append(x_pad[:ilen][:-self.tnum])
        xs_pad_s = pad_list(xs, 0)  # B, Tmax, C

        xs_pad_t = xs_pad.clone()
        xs_pad_t.retain_grad()
        xs = []
        for x_pad, ilen in zip(xs_pad_t, ilens):
            xs.append(torch.stack([x_pad[:ilen][i + 1:-self.tnum + i + 1] if (-self.tnum + i + 1) != 0 else
                                   x_pad[:ilen][i + 1:] for i in range(self.tnum)], dim=-2))
        xs_pad_t = pad_list(xs, 0)  # B, Tmax, tnum, C

        # monitoring buffer
        buffs = {'loss': []}

        # start iteration for superposition
        hidden_mask = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # feedforward to inference network
            xs_ele_t, hidden_mask = self.inference(xs_pad_s, hidden_mask, decoder_type='src')
            xs_ele_t = xs_ele_t.unsqueeze(-2).repeat(1, 1, self.tnum, 1)  # B, Tmax, tnum, C

            # compute loss of total network
            masks = [seq_mask.view(-1, 1)]
            loss_local = self.criterion(xs_ele_t.mean(-2).view(-1, self.idim), xs_pad_t.mean(-2).view(-1, self.idim),
                                        masks)

            # compute residual feature
            xs_pad_t = (xs_pad_t - xs_ele_t).detach()

            # buffering
            buffs['loss'].append(loss_local.sum())

        # total loss compute
        loss = torch.stack(buffs['loss'], dim=-1).mean()

        if not torch.isnan(loss):
            self.reporter.report(float(loss))
        else:
            print("loss (=%f) is not correct", float(loss))
            exit()
            logging.warning("loss (=%f) is not correct", float(loss))

        return loss
