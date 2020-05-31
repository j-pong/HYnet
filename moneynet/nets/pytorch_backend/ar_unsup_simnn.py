#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import six

import torch
import torch.nn.functional as F
from torch import nn

import chainer
from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from moneynet.nets.pytorch_backend.unsup.initialization import initialize
from moneynet.nets.pytorch_backend.unsup.loss import SeqMultiMaskLoss
from moneynet.nets.pytorch_backend.unsup.inference import Inference, ConvInference


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
        # task related
        group.add_argument("--tnum", default=5, type=int)
        group.add_argument("--iter", default=1, type=int)

        # optimization related
        group.add_argument("--lr", default=0.001, type=float)
        group.add_argument("--momentum", default=0.9, type=float)

        # model related
        group.add_argument("--hdim", default=512, type=int)
        group.add_argument("--cdim", default=128, type=int)
        group.add_argument("--inference_type", default='conv', type=str)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        super(Net, self).__init__()
        # network hyperparameter
        self.idim = idim
        self.odim = idim
        self.iter = args.iter
        self.tnum = args.tnum
        self.ignore_id = ignore_id
        self.subsample = [1]

        self.k = 1000

        # reporter for monitoring
        self.reporter = Reporter()

        # inference part with action and selection
        self.embed = torch.nn.Embedding(self.k, self.odim)
        if args.inference_type == 'linear':
            self.inference = Inference(idim=idim, odim=idim, args=args)
        elif args.inference_type == 'conv':
            self.inference = ConvInference(idim=idim, odim=idim, args=args)

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
        buffs = {'loss': [], 'score_idx': [], 'conservation_error': 0}

        # start iteration for superposition
        for _ in six.moves.range(self.iter):
            # find anchor with maximum similarity
            score = torch.matmul(xs_pad_in, self.embed.weight.t()) / \
                    torch.norm(self.embed.weight, dim=-1).view(1, 1, self.k)
            score_idx = torch.argmax(score, dim=-1)  # B, Tmax
            buffs['score_idx'].append(score_idx)
            anchor = self.embed(score_idx)

            # feedforward to inference network
            xs_ele_out = self.inference(anchor)
            sz = xs_ele_out.size()
            xs_ele_out = xs_ele_out.view(sz[0], sz[1], self.tnum, self.idim)  # B, Tmax, tnum, C

            # compute loss of total network
            masks = [seq_mask.unsqueeze(-1).repeat(1, 1, self.tnum).view(-1, 1)]
            loss_local = self.criterion(xs_ele_out.reshape(-1, self.idim),
                                        xs_pad_out.reshape(-1, self.idim),
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

        # find anchor with maximum similarity
        score = torch.matmul(x, self.embed.weight.t()) / \
                torch.norm(self.embed.weight, dim=-1).view(1, 1, self.k)
        score_idx = torch.argmax(score, dim=-1)
        anchor = self.embed(score_idx)

        out = anchor

        return out
