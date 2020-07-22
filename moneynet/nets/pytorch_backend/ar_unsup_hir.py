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
from moneynet.nets.pytorch_backend.unsup.inference import HirInference

from moneynet.nets.pytorch_backend.unsup.plot import PlotImageReport


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
    def mvn(x):
        m = torch.mean(x, dim=-1, keepdim=True)
        v = torch.mean(torch.pow(x - m, 2), dim=-1, keepdim=True)
        x = (x - m) / v
        return x

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

    def energy_quantization(self, x):
        x_past = None
        for el in self.energy_level:
            if x_past is None:
                x[x > el] = el
            else:
                x[(x_past > x) & (x > el)] = el
            x_past = el
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

    def clustering(self, xs, golden_ratio=3.14):
        """
        xs: (B, tnum, Tmax)

        return: (B, 1, Tmax)
        """
        with torch.no_grad():
            ms = []
            for x in xs:
                if self.tnum == 1:
                    num_neg = int(x[0].sum(-1) * golden_ratio)
                    if num_neg < 1:
                        num_neg = 1
                        logging.warning("num_neg has wrong value < 1")
                    elif num_neg > x.size(-1):
                        num_neg = x.size(-1)
                        logging.warning("num_neg has wrong value > {}".format(num_neg))
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
        xs_pad_in = xs_pad_in[:, :max(ilens), :-self.trunk].unsqueeze(1)  # for data parallel
        xs_pad_out = xs_pad_out[:, :max(ilens), :, :-self.trunk].transpose(1, 2)

        seq_mask = make_pad_mask((ilens).tolist()).to(xs_pad_in.device)
        tsz = seq_mask.size(-1)

        seq_mask_kernel = (1 - torch.matmul((~seq_mask).unsqueeze(-1).float(),
                                            (~seq_mask).unsqueeze(1).float())).bool()

        # initialization buffer
        self.reporter_buffs = {}

        # 1. transform
        h, ratio_e = self.engine(xs_pad_in, self.engine.encoder)
        p_hat = self.engine.brew_(module_lists=[self.engine.encoder], ratio=ratio_e,
                                  split_dim=None)
        # self attention based kerenl
        min_value = float(
            np.finfo(torch.tensor(0, dtype=torch.float).numpy().dtype).min
        )
        q, k = h.split(int(h.size(-1) / 2), dim=-1)
        kernel = torch.matmul(q, k.transpose(-2, -1))
        kernel = kernel.masked_fill(seq_mask_kernel.unsqueeze(1), min_value)
        kernel = torch.softmax(kernel, dim=-1).\
            masked_fill(seq_mask_kernel.unsqueeze(1), 0.0)
        h_ = torch.matmul(kernel, q)
        h = torch.cat([h_, k], dim=-1)
        # ratio_k = self.engine.calculate_ratio(h_, h)
        # h = h_
        # p_hat = self.engine.amp(ratio_k, p_hat[0], p_hat[1])

        # training key for major feature extraction
        kernel_explict = torch.matmul(k, k.transpose(-2, -1))

        xs_pad_out_hat, ratio_d = self.engine(h, self.engine.decoder)
        p_hat = self.engine.brew_(module_lists=[self.engine.decoder], ratio=ratio_d,
                                  split_dim=None,
                                  w_hat=p_hat[0],
                                  bias_hat=p_hat[1])
        if self.target_type == 'mvn':
            xs_pad_in_m = torch.mean(xs_pad_in, dim=-1, keepdim=True)
            xs_pad_in_v = torch.mean(torch.pow(xs_pad_in - xs_pad_in_m, 2), dim=-1, keepdim=True)
            xs_pad_out_hat_ = self.mvn(xs_pad_out_hat)
            xs_pad_out_hat_ = xs_pad_out_hat_ * xs_pad_in_v + xs_pad_in_m
        elif self.target_type == 'residual':
            xs_pad_out_hat_ = xs_pad_out_hat + xs_pad_in

        # 2. calculate energy
        flows = self.calculate_energy(start_state=xs_pad_in.contiguous(),
                                      end_state=xs_pad_out_hat.contiguous(),
                                      p_hat=p_hat,
                                      flows={'e': None})
        energy_t = flows['e'].detach()

        # 3. calculating similarity
        fc_mat = torch.matmul(xs_pad_out, xs_pad_out_hat.transpose(-2, -1))

        # 4. make target
        with torch.no_grad():
            # # task 0
            # fc_target = torch.ones_like(fc_mat).to(fc_mat.device)

            # th = []
            # # task1
            # for i, ke in enumerate(kernel.sort(dim=-2, descending=True)[0]):
            #     ind = ilens[i] * (1.0 - energy_t[i, 0])
            #     th.append(ke[:, ind.long(), range(tsz)])
            # th = torch.stack(th, dim=0).unsqueeze(1)
            # kernel_target = (kernel > th).float()

            # # task2
            # kernel_ = torch.sigmoid(kernel)
            # for i, ke in enumerate(kernel_.sort(dim=-2, descending=True)[0]):
            #     ke = torch.cumsum(ke, dim=-2)
            #     ke = ke / ke[:, -1, :].unsqueeze(1)
            #     ind = torch.argmin(torch.abs(ke - 1 + energy_t[i].unsqueeze(1)), dim=-2)[0]
            #     th.append(ke[:, ind.long(), range(tsz)])
            # th = torch.stack(th, dim=0).unsqueeze(1)
            # kernel_target = (kernel_ > th).float()

            # task 4
            energy_t_inv = (1.0 - self.energy_quantization(energy_t))
            fc_target = self.fb(energy_t_inv)

        # 5. calculate similarity loss
        loss_k = self.criterion_kernel(kernel_explict.view(-1, tsz, tsz),
                                       fc_target.view(-1, tsz, tsz),
                                       [seq_mask_kernel, fc_target.view(-1, tsz, tsz) < 0.1],
                                       reduction='none')
        loss_g = self.criterion_kernel(fc_mat.view(-1, tsz, tsz),
                                       fc_target.view(-1, tsz, tsz),
                                       [seq_mask_kernel, fc_target.view(-1, tsz, tsz) < 0.1],
                                       reduction='none')
        # summary all task
        loss = loss_g.sum() + loss_k.sum()
        if not torch.isnan(loss):
            self.reporter.report({'loss': float(loss),
                                  'loss_g': float(loss_g.sum())})
        else:
            logging.warning("loss (=%f) is not correct", float(loss))

        if buffering:
            self.reporter_buffs['out'] = xs_pad_out_hat_

            self.reporter_buffs['energy_t'] = energy_t[:, 0]

            self.reporter_buffs['kernel'] = torch.sigmoid(torch.cat([kernel, kernel_explict], dim=1))

            self.reporter_buffs['fc_target'] = fc_target
            self.reporter_buffs['fc_mat'] = torch.sigmoid(fc_mat)

        return loss

    def calculate_energy(self, start_state, end_state, p_hat, flows={'e': None}):
        with torch.no_grad():
            # prepare data
            w_hat = p_hat[0]
            b_hat = p_hat[1]

            start_dim = start_state.size(-1)
            bsz, tnsz, tsz, end_dim = end_state.size()
            start_state = start_state.view(-1, start_dim)
            end_state = end_state.view(-1, end_dim) - b_hat

            # distance
            distance = F.kl_div(input=torch.log_softmax(torch.abs(w_hat), dim=-1),
                                target=self.field.to(w_hat.device),
                                reduction='none').float()
            distance = distance.sum(-1)

            # time
            sign_pair = torch.matmul(torch.sign(start_state.unsqueeze(-1)),
                                     torch.sign(end_state.unsqueeze(-2)))
            w_hat_x = w_hat * sign_pair
            freq = (torch.relu(w_hat_x).sum(-1) / torch.abs(w_hat_x).sum(-1))
            num_nan = torch.isnan(freq).float().mean()
            if num_nan > 0.1:
                logging.warning("Energy sequence impose the nan {}".format(num_nan))
            freq[torch.isnan(freq)] = 0.5
            time = torch.abs(freq - 0.5) + 1

            if self.target_type == 'mvn':
                energy = torch.pow(time * torch.abs(start_state), 2)
            elif self.target_type == 'residual':
                energy = distance * torch.abs(start_state) * freq
                energy = energy.mean(-1)
                energy = energy.view(bsz, tnsz, tsz)

                # # make gaussian filter
                # variance = 3
                # space = np.linspace(0, tsz - 1, tsz)
                # lpass_filter = np.expand_dims(
                #     np.stack([self.gaussian_func(space, i, variance) for i in range(tsz)], axis=0), 0)
                # lpass_filter = torch.from_numpy(lpass_filter / np.amax(lpass_filter)).to(energy.device)
                # lpass_filter = torch.triu(lpass_filter).float()
                # # filtering with gaussain filter
                # energy = torch.matmul(energy.view(bsz, tnsz, tsz), lpass_filter)
                # # calculate energy difference
                # energy = torch.cat([energy[:, :, 1:] - energy[:, :, :-1],
                #                     torch.zeros_like(energy[:, :, 0:1]).to(energy.device)], dim=-1)
                # energy = torch.abs(energy)
                # # time step
                # freq = freq.mean(-1).view(bsz, tnsz, tsz)
                # freq = torch.cat([freq[:, :, 1:],
                #                   torch.ones_like(freq[:, :, 0:1]).to(energy.device)], dim=-1)
                # energy = energy / freq

            # calculate energy
            for key in flows.keys():
                if key == 'e':
                    # dump
                    flows[key] = energy
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
