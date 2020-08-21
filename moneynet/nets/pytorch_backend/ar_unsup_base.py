#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import chainer
from chainer import reporter

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list

from moneynet.nets.pytorch_backend.unsup.initialization import initialize
from moneynet.nets.pytorch_backend.unsup.loss import SeqMultiMaskLoss
from moneynet.nets.pytorch_backend.unsup.inference import HirInference

from moneynet.nets.pytorch_backend.unsup.plot import PlotImageReport


# from sklearn.cluster import KMeans


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
        ## Fair task
        group.add_argument("--prediction-steps", default=1, type=int)
        group.add_argument("--num-negatives", default=1, type=int)
        group.add_argument("--cross-sample-negatives", default=0, type=int)
        group.add_argument("--sample-distance", default=10, type=int)
        group.add_argument("--dropout", default=0.2, type=float)
        group.add_argument("--balanced-classes", default=0, type=int)
        group.add_argument("--infonce", default=0, type=int)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        super().__init__()
        # network hyperparameter
        self.trunk = 3
        self.idim = idim - self.trunk
        self.odim = idim - self.trunk
        self.hdim = args.hdim
        self.tnum = args.tnum
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.min_value = float(
            np.finfo(torch.tensor(0, dtype=torch.float).numpy().dtype).min
        )

        # related task
        self.target_type = args.target_type
        ## calculate energy task
        self.field_var = args.field_var
        space = np.linspace(0, self.odim - 1, self.odim)
        self.field = np.expand_dims(
            np.stack([self.gaussian_func(space, i, self.field_var) for i in range(self.odim)], axis=0), 0)
        self.field = torch.from_numpy(self.field)
        ## energy post processing task
        self.energy_level = np.array([13, 8, 5, 3, 2, 1]) / 32

        # inference part with action and selection
        self.engine = HirInference(idim=self.idim, odim=self.idim, args=args)

        # network training related
        self.criterion = SeqMultiMaskLoss(criterion=nn.MSELoss(reduction='none'))
        self.criterion_kernel = SeqMultiMaskLoss(criterion=nn.BCEWithLogitsLoss(reduction='none'))

        # reporter for monitoring
        self.reporter = Reporter()
        self.reporter_buffs = {}

        # initialize parameter
        initialize(self)

    @staticmethod
    def gaussian_func(x, m=0.0, sigma=1.0):
        norm = np.sqrt(2 * np.pi * sigma ** 2)
        dist = (x - m) ** 2 / (2 * sigma ** 2)
        return 1 / norm * np.exp(-dist)

    @staticmethod
    def minimaxn(x):
        max_x = torch.max(x, dim=-1, keepdim=True)[0]
        min_x = torch.min(x, dim=-1, keepdim=True)[0]
        # if (max_x - min_x) == 0.0:
        #     logging.warning('Divided by the zero with max-min value : {}, Thus return None'.format((max_x - min_x)))
        #     x = None
        # else:
        x = (x - min_x) / (max_x - min_x)

        return x

    @staticmethod
    def fb(x, mode='batch', offset=1):
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
                    if offset > 0:
                        x_f = torch.cat([ones[0:offset], x_f[:-offset]], dim=0) - m_f
                    else:
                        x_f = x_f - m_f

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

    @staticmethod
    def attn(q, k, mask, min_value, mask_diag=False, T=None):
        """
        q : B, 1, T, C
        K : B, 1, T, C
        """
        tsz = q.size(-2)

        score = torch.matmul(k, q.transpose(-2, -1)) / np.sqrt(q.size(-1))
        if mask_diag:
            score[:, :, range(tsz), range(tsz)] = min_value
        score = score.masked_fill(mask, min_value)
        if T is not None:
            score = score / T.unsqueeze(-1)  # B, 1, T, T / B, 1, T, 1
        kernel = torch.softmax(score, dim=-1). \
            masked_fill(mask, 0.0)
        if mask_diag:
            kernel[:, :, range(tsz), range(tsz)] = 0.0
        c = torch.matmul(kernel, q)

        # check worst case of the kernel
        if mask_diag:
            assert torch.sum((kernel.diagonal(dim1=-2, dim2=-1) != 0.0).float()) == 0.0

        return c, kernel, score

    @staticmethod
    def energy_quantization(x, energy_level):
        x_past = None
        for el in energy_level:
            if x_past is None:
                x[x > el] = el
            else:
                x[(x_past > x) & (x > el)] = el
            x_past = el
        x[x_past > x] = x_past
        return x

    def forward(self, xs_pad_in, xs_pad_out, ilens, ys_pad, buffering=False):
        # 0. prepare data
        xs_pad_in = xs_pad_in[:, :max(ilens), :-self.trunk].unsqueeze(1)  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens), :, :-self.trunk].transpose(1, 2)

        bsz, tnum, tsz, fdim = xs_pad_in.size()

        # seq_mask related
        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)
        seq_mask_kernel = (1 - torch.matmul((~seq_mask).unsqueeze(-1).float(),
                                            (~seq_mask).unsqueeze(1).float())).bool()

        # initialization buffer
        self.reporter_buffs = {}

        # 1. preparation of key dictionary : must key is aligned respect to target seq.
        # first key almost goes to one-hot but second is aux-info.
        k, _ = self.engine(xs_pad_out, self.engine.encoder_k)
        kernel_kk = torch.matmul(k, k.transpose(-2, -1))

        # 2. query for causal case
        q, ratio_e_q = self.engine(xs_pad_in, self.engine.encoder_q)

        # 3. transformer
        agg_q, kernel_kq, score = self.attn(q, k, seq_mask_kernel.unsqueeze(1),
                                            min_value=self.min_value,
                                            mask_diag=True)
        xs_pad_out_hat, ratio_d = self.engine(agg_q, self.engine.decoder)

        # 4. brewing with attention respect to second attention
        with torch.no_grad():
            p_hat = self.engine.brew_(module_lists=[self.engine.encoder_q],
                                      ratio=ratio_e_q,
                                      split_dim=None)
            w = torch.matmul(kernel_kq.view(bsz, tnum, tsz, tsz), p_hat[0].view(bsz, tnum, tsz, -1)). \
                view(-1, fdim, self.hdim)
            b = torch.matmul(kernel_kq.view(bsz, tnum, tsz, tsz), p_hat[1].view(bsz, tnum, tsz, -1)). \
                view(-1, self.hdim)
            p_hat = (w, b)
            p_hat = self.engine.brew_(module_lists=[self.engine.decoder],
                                      ratio=ratio_d,
                                      split_dim=None,
                                      w_hat=p_hat[0],
                                      bias_hat=p_hat[1])

        # 5. post processing
        w, b = p_hat
        w = w.view(bsz, tnum, tsz, fdim, fdim)
        b = b.view(bsz, tnum, tsz, fdim)

        # 6. make target with energy
        # kernel_kq = torch.softmax(score / 0.01, dim=-1).masked_fill(seq_mask_kernel.unsqueeze(1), 0.0)
        energy_t = kernel_kq.diagonal(dim1=-2, dim2=-1, offset=1)
        energy_t = torch.cat([torch.ones_like(energy_t)[:, :, 0:1], 1 - energy_t], dim=-1)
        kernel_target = self.fb(energy_t)
        # energy_t = kernel_kq.diagonal(dim1=-2, dim2=-1, offset=-1)
        # kernel_target += self.fb(torch.cat([torch.ones_like(energy_t)[:, :, 0:1], energy_t], dim=-1))

        # 7. calculate similarity loss
        loss_k = self.criterion_kernel(torch.tril(kernel_kk.view(-1, tsz, tsz)),
                                       torch.tril(kernel_target.view(-1, tsz, tsz)),
                                       [seq_mask_kernel],
                                       reduction='none')

        # 8. calculate generative loss
        with torch.no_grad():
            xs_pad_out_hat_hat = torch.matmul(xs_pad_in.unsqueeze(-2), w).squeeze(-2) + b
            loss_test = self.criterion(xs_pad_out_hat_hat.view(-1, self.idim),
                                       xs_pad_out.contiguous().view(-1, self.idim),
                                       [seq_mask.view(-1, 1)],
                                       reduction='none')
            xs_pad_out_hat_hat = torch.matmul(torch.abs(w),
                                              torch.abs(xs_pad_out - b).unsqueeze(-2).transpose(-2, -1)).squeeze(-1)
        loss_g = self.criterion(xs_pad_out_hat.view(-1, self.idim),
                                xs_pad_out.contiguous().view(-1, self.idim),
                                [seq_mask.view(-1, 1)],
                                reduction='none')

        # summary all task
        loss = loss_g.sum() + loss_k.sum()
        if not torch.isnan(loss):
            self.reporter.report({'loss': float(loss),
                                  'loss_1': float(loss_g.sum()),
                                  'loss_2': float(loss_k.sum()),
                                  'loss_3': float(loss_test.sum())})
        else:
            logging.warning("loss (=%f) is not correct", float(loss))

        if buffering:
            self.reporter_buffs['out'] = xs_pad_out_hat_hat

            self.reporter_buffs['p_hat'] = w[:, 0, 100:101, :, :]

            self.reporter_buffs['energy_t'] = energy_t[:, 0, :200]

            self.reporter_buffs['kernel_kq'] = kernel_kq[:, :, :200, :200]
            self.reporter_buffs['kernel_kk'] = torch.sigmoid(kernel_kk)
            self.reporter_buffs['kernel_kk_target'] = kernel_target[:, :, :200, :200]

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

        dump = dict()
        for key in self.reporter_buffs.keys():
            dump[key] = self.reporter_buffs[key].cpu().numpy()

        return dump

    def recognize(self, x):
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)[:, :, :-self.trunk]

        x, _ = self.engine(x, self.engine.encoder_k)

        return x
