#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import six

import torch
from torch import nn
import torch.nn.functional as F

import chainer
from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from moneynet.nets.pytorch_backend.unsup.initialization import initialize
from moneynet.nets.pytorch_backend.unsup.loss import SeqMultiMaskLoss
from moneynet.nets.pytorch_backend.unsup.inference import Inference, ConvInference

from moneynet.nets.pytorch_backend.unsup.plot import PlotImageReport


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss, e_positive, e_negative):
        """Report at every step."""
        reporter.report({"loss": loss}, self)
        reporter.report({"e_positive": e_positive}, self)
        reporter.report({"e_negative": e_negative}, self)


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

        # task related
        group.add_argument("--bias", default=1, type=int)
        group.add_argument("--embed_mem", default=0, type=int)
        group.add_argument("--brewing", default=0, type=int)

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

        # additive task
        self.embed_mem = args.embed_mem
        self.brewing = args.brewing

        # reporter for monitoring
        self.reporter = Reporter()

        if self.embed_mem:
            # clustering configuration
            self.embed_dim_high = 1000
            self.embed_dim_low = 50
            self.low_freq = 2
            self.embed_feat_high = torch.nn.Embedding(self.embed_dim_high, self.odim - self.low_freq)
            self.embed_feat_low = torch.nn.Embedding(self.embed_dim_low, self.low_freq)
            # self.embed_w = torch.nn.Embedding(self.embed_dim, self.idim * self.odim)
        # inference part with action and selection
        if args.inference_type == 'linear':
            self.transform_f = Inference(idim=idim, odim=idim, args=args)
        elif args.inference_type == 'conv':
            self.transform_f = ConvInference(idim=idim, odim=idim, args=args)

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

    def clustering(self, x, embed):
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
        self.buffs = {'score_idx_h': [], 'score_idx_l': [], 'out': []}

        # For solving superposition state of the feature
        if self.embed_mem:
            anchors = []
            for _ in six.moves.range(self.iter):
                anchor_h, score_idx_h, _ = self.clustering(xs_pad_in[..., :-self.low_freq], self.embed_feat_high)
                anchor_l, score_idx_l, _ = self.clustering(xs_pad_in[..., -self.low_freq:], self.embed_feat_low)
                anchor = torch.cat([anchor_h, anchor_l], dim=-1)
                if self.iter != 1:
                    xs_pad_in = (xs_pad_in - anchor).detach()
                anchors.append(anchor)
                if self.eval:
                    self.buffs['score_idx_h'].append(score_idx_h)
                    self.buffs['score_idx_l'].append(score_idx_l)
            anchors = torch.stack(anchors, dim=1).unsqueeze(1).repeat(1, 1, self.tnum, 1,
                                                                      1)  # B, iter, tnum, Tmax, idim
        else:
            anchors = xs_pad_in.unsqueeze(1).unsqueeze(1).repeat(1, 1, self.tnum, 1, 1)

        # Inference via transform function with anchors
        xs_pad_out_hat, ratio_enc, ratio_dec = self.transform_f(anchors)  # B, iter, tnum, Tmax, odim
        xs_pad_out_hat = xs_pad_out_hat.mean(dim=1)  # B, iter, tnum, Tmax, odim
        if self.eval:
            self.buffs['out'].append(xs_pad_out_hat)

        if self.brewing:
            p_hat = self.transform_f.brew([ratio_enc, ratio_dec])
            p_hat = p_hat[0]

            w_hat_x = p_hat[0] * torch.sign(anchors.view(-1, self.idim).unsqueeze(-1).detach())
            w_hat_x_p = torch.relu(w_hat_x)
            w_hat_x_n = torch.relu(-w_hat_x)
            e_positive = torch.sum(torch.mean(w_hat_x_p, dim=0))
            e_negative = torch.sum(torch.mean(w_hat_x_n, dim=0))
        else:
            e_positive = 0
            e_negative = 0

        # compute loss of total network
        masks = [seq_mask.unsqueeze(1).repeat(1, self.tnum, 1).view(-1, 1)]
        loss = self.criterion(xs_pad_out_hat.reshape(-1, self.idim),
                              xs_pad_out.reshape(-1, self.idim),
                              masks).mean()

        if not torch.isnan(loss):
            self.reporter.report(float(loss), float(e_positive), float(e_negative))
        else:
            print("loss (=%f) is not correct", float(loss))
            logging.warning("loss (=%f) is not correct", float(loss))

        return loss

    def recognize(self, x):
        assert self.embed_mem
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)

        buffs = {'score_idx': []}

        # For solving superposition state of the feature
        for _ in six.moves.range(self.iter):
            anchor, score_idx, _ = self.clustering(x, self.embed_feat)
            x = (x - anchor).detach()
            buffs['score_idx'].append(score_idx)

        out = torch.stack(buffs['score_idx'], dim=-1)[0]

        return out

    @property
    def images_plot_class(self):
        return PlotImageReport

    def calculate_images(self, xs_pad_in, xs_pad_out, ilens, ys_pad):
        with torch.no_grad():
            self.forward(xs_pad_in, xs_pad_out, ilens, ys_pad)
        ret = dict()
        if self.embed_mem:
            ret['score_idx_h'] = F.one_hot(torch.stack(self.buffs['score_idx_h'], dim=1),
                                           num_classes=self.embed_dim_high).cpu().numpy()
            ret['score_idx_l'] = F.one_hot(torch.stack(self.buffs['score_idx_l'], dim=1),
                                           num_classes=self.embed_dim_low).cpu().numpy()
        ret['out'] = self.buffs['out'][0].cpu().numpy()
        return ret
