#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import six

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import chainer
from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from moneynet.nets.pytorch_backend.unsup.initialization import initialize
from moneynet.nets.pytorch_backend.unsup.loss import SeqMultiMaskLoss
from moneynet.nets.pytorch_backend.unsup.inference import ExcInference


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
        group.add_argument("--etype", default="linear", type=str)
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
        self.odim = idim
        self.hdim = args.hdim
        self.cdim = args.cdim
        self.tnum = args.tnum
        self.ignore_id = ignore_id
        self.subsample = [1]

        self.k = 1000

        # reporter for monitoring
        self.reporter = Reporter()

        # inference part with action and selection
        self.embed = torch.nn.Embedding(self.k, self.odim)
        self.inference = ExcInference(self.idim, self.odim, args)

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

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad):
        # prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens)]  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens)]
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)
        # print(xs_pad_in.size(), xs_pad_out.size(), seq_mask.size())
        # exit()

        # monitoring buffer
        buffs = {'loss': [], 'score_idx': []}

        # start iteration for superposition
        hidden_mask = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # find anchor with maximum similarity
            score = torch.matmul(xs_pad_in, self.embed.weight.t()) / \
                    torch.norm(self.embed.weight, dim=-1).view(1, 1, self.k)
            score_idx = torch.argmax(score, dim=-1)  # B, Tmax
            buffs['score_idx'].append(score_idx)
            anchor = self.embed(score_idx)

            # feedforward to inference network
            xs_ele_out, _, hidden_mask = self.inference(anchor, hidden_mask, decoder_type='src')
            xs_ele_out = xs_ele_out.unsqueeze(-2).repeat(1, 1, self.tnum, 1)  # B, Tmax, tnum, C

            # compute loss of total network
            masks = [seq_mask.view(-1, 1)]
            loss_local = self.criterion(xs_ele_out.mean(-2).view(-1, self.idim),
                                        xs_pad_out.mean(-2).view(-1, self.idim),
                                        masks)

            # compute residual feature
            xs_pad_out = (xs_pad_out - xs_ele_out).detach()

            # buffering
            buffs['loss'].append(loss_local.sum())

        # total loss compute
        loss = torch.stack(buffs['loss'], dim=-1).mean()

        if not torch.isnan(loss):
            self.reporter.report(float(loss))
        else:
            print("loss (=%f) is not correct", float(loss))
            logging.warning("loss (=%f) is not correct", float(loss))

        return loss

    def recognize(self, x):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        buffs = {'score_idx': []}

        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # find anchor with maximum similarity
            score = torch.matmul(x, self.embed.weight.t()) / \
                    torch.norm(self.embed.weight, dim=-1).view(1, 1, self.k)
            score_idx = torch.argmax(score, dim=-1)  # B, Tmax
            buffs['score_idx'].append(score_idx)
            anchor = self.embed(score_idx)

            # feedforward to inference network
            xs_ele_out, _, hidden_mask = self.inference(anchor, hidden_mask, decoder_type='src')
            xs_ele_out = xs_ele_out.unsqueeze(-2).repeat(1, 1, self.tnum, 1)  # B, Tmax, tnum, C

            # compute residual feature
            x = (x - xs_ele_out)

        out = torch.stack(buffs['score_idx'], dim=-1)

        return out
