#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F

from torch import nn

import numpy as np


def initialize(model, init_type="xavier_uniform"):
    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()


class NetLoss(nn.Module):
    def __init__(self, ignore_in, ignore_out, criterion=nn.MSELoss(reduce=None)):
        super(NetLoss, self).__init__()
        self.criterion = criterion
        self.ignore_in = ignore_in
        self.ignore_out = ignore_out

    def forward(self, x, y):
        if np.isnan(self.ignore_out):
            mask = torch.isnan(y)
            y = y.masked_fill(mask, self.ignore_in)
        else:
            mask = y == self.ignore_out
        denom = (~mask).float().sum()
        loss = self.criterion(input=x, target=y)
        return loss.masked_fill(mask, 0).sum() / denom


class Net(nn.Module):
    def __init__(self, idim, odim, args, reporter):
        super(Net, self).__init__()
        # utils
        self.reporter = reporter

        # network hyperparameter
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out

        # freq distribution design
        weight = self._hg_kernel(np.arange(0, self.hdim, dtype=np.float32), mu=0.0, sigma=430)
        weight = np.concatenate([1.0 - weight, weight], axis=0).T  # binomial distribution needs for hidden mask
        self.freq = torch.from_numpy(weight)

        # simple network add
        self.fc1 = nn.Linear(idim, self.hdim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hdim, odim)

        # network training related
        self.criterion = NetLoss(ignore_in=self.ignore_in, ignore_out=self.ignore_out)

        # initialize parameter
        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    def _hg_kernel(self, x, mu, sigma):
        denom = 1 / (sigma * np.sqrt(2 * np.pi))
        y = denom * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        y = y / np.max(y)  # normalizing with max so that first elements of hidden units is always fire
        y = np.expand_dims(y, axis=0)  # for binomial distribution

        return y

    def _lin_kernel(self, x):
        y = np.flip(x)
        y = y / np.max(y)  # normalizing with max so that first elements of hidden units is always fire
        y = np.expand_dims(y, axis=0)  # for binomial distribution

        return y

    def _pad_for_shift(self, key, query):
        """Padding to channel dim for convolution

        :param torch.Tensor key: batch of padded source sequences (B, Tmax, hdim, idim_k)
        :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)

        :return: padded and truncated tensor that matches to query dim (B, Tmax, hdim, idim_k + idim_q - 1, idim_k)
        :rtype: torch.Tensor
        :return: padded and truncated tensor that matches to query dim (B, Tmax, hdim, idim_k + idim_q - 1, idim_k)
        :rtype: torch.Tensor
        """
        idim_k = key.size(-1)
        idim_q = query.size(-1)
        pad = idim_q - 1

        key_pad = F.pad(key, pad=(pad, pad))  # (B, Tmax, hdim, idim_k + pad * 2)
        key_pad_trunk = []
        query_mask = []
        for i in six.moves.range(idim_k + idim_q - 1):
            # key pad trunk
            kpt = key_pad[..., i:i + idim_q]
            key_pad_trunk.append(kpt.unsqueeze(-2))
            # make mask for similarity
            kpt_mask = torch.zeros_like(kpt).to(kpt.device)
            end = -max(i - idim_q, 0)
            if end < 0:
                kpt_mask[..., -(i + 1):end] = 1
            else:
                kpt_mask[..., -(i + 1):] = 1
            query_mask.append(kpt_mask.unsqueeze(-2))
        key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, hdim ,idim_k + idim_q - 1, idim_q)
        query_mask = torch.cat(query_mask, dim=-2)  # (B, Tmax, hdim, idim_k + idim_q - 1, idim_q)
        return key_pad_trunk, query_mask

    def _simiality(self, key_pad_trunk, query, query_mask=None, measurement='cos', normalize=True):
        """Measuring similarity of each other tensor

        :param torch.Tensor key_pad_trunk: batch of padded source sequences (B, Tmax, hdim, idim_k + idim_q - 1, idim_k)
        :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)
        :param torch.Tensor query_mask: batch of padded source sequences (B, Tmax, hdim, idim_k + idim_q - 1, idim_k)
        :param string measurement:

        :return: max similarity value of sequence (B, Tmax, hdim)
        :rtype: torch.Tensor
        :return: max similarity index of sequence  (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        query = query.unsqueeze(-2).unsqueeze(-2)
        if query_mask is not None:
            query = query * query_mask
        if measurement == 'cos':
            denom = torch.norm(query, dim=-1) * torch.norm(key_pad_trunk, dim=-1)
            sim = torch.sum(query * key_pad_trunk, dim=-1) / denom  # (B, Tmax, idim_k + idim_q - 1)
            # nan filtering
            mask = torch.isnan(sim)
            sim[mask] = 0.0
            # optimal similarity point search
            sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax, hdim)
        else:
            raise AttributeError('{} is not support yet'.format(measurement))

        return sim_max, sim_max_idx

    def argaug(self, x, y, measurement='cos'):
        """Find augmentation parameter using grid search

        :param torch.Tensor x: batch of padded source sequences (B, Tmax, hdim, idim)
        :param torch.Tensor y: batch of padded target sequences (B, Tmax, hdim, idim)
        :param string measurement:

        :return: augmented source sequences (B, Tmax, hdim, idim)
        :rtype: torch.Tensor
        :return: value of augmentation parameter (B, Tmax, hdim, 2)
        :rtype: torch.Tensor
        :return: value of similarity (B, Tmax, hdim)
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        time_size = x.size(1)
        c1_size = x.size(2)  # hdim
        x_aug = None
        for theta in np.linspace(0.8, 1.2, 5):
            # scale
            x_s = F.interpolate(x, scale_factor=(1, theta))
            # shift
            x_s_pad_trunk, _ = self._pad_for_shift(key=x_s, query=y)  # (B, Tmax, hdim ,idim_k + idim_q - 1, idim_q)
            # similarity measuring
            sim_max, sim_max_idx = self._simiality(key_pad_trunk=x_s_pad_trunk, query=y, query_mask=None,
                                                   measurement=measurement)  # (B, Tmax, hdim)
            # aug parameters
            p_aug_shift = sim_max_idx.view(-1, 1).float()
            p_aug_scale = torch.tensor([theta]).view(1, 1).repeat(repeats=p_aug_shift.size()).to(p_aug_shift.device)
            p_augs = torch.cat([p_aug_shift, p_aug_scale], dim=-1)  # (B * Tmax * hdim, 2)
            # maximum shift select
            xspt_size = x_s_pad_trunk.size()
            x_s_pad_trunk = x_s_pad_trunk.view(-1,
                                               xspt_size[-2],
                                               xspt_size[-1])  # (B * Tmax * hdim, idim_k + idim_q - 1, idim_q)
            x_s_opt = x_s_pad_trunk[
                torch.arange(x_s_pad_trunk.size(0)), sim_max_idx.view(-1)]  # (B * Tmax * hdim, idim_q)
            # maximum scale select
            if x_aug is None:
                x_aug = x_s_opt
                sim_max_global = sim_max
                p_augs_global = p_augs
            else:
                mask = sim_max_global < sim_max
                x_aug[mask.view(-1)] = x_s_opt[mask.view(-1)]
                sim_max_global[mask] = sim_max[mask]
                p_augs_global[mask.view(-1)] = p_augs[mask.view(-1)]
        # recover shape
        x_aug = x_aug.view(batch_size, time_size, c1_size, -1)
        p_augs_global = p_augs_global.view(batch_size, time_size, c1_size, -1)
        return x_aug, sim_max_global, p_augs_global

    def freq_mask(self, x):
        batch_size = x.size(0)
        time_size = x.size(1)
        mask = torch.multinomial(self.freq, num_samples=batch_size * time_size, replacement=True)  # [B * T, C]
        mask = mask.to(x.device).T
        mask = mask.view(batch_size, time_size, -1)

        return x * mask, mask

    def disentangle(self, h, kernel_output=False, normalize=False):
        """Disentangle representation

        """
        # make mask for figurig out each node output
        m_h = torch.eye(self.hdim).to(h.device).unsqueeze(0).float()
        h_ = m_h * h.unsqueeze(-1)  # (B, Tmax, hdim, hdim)
        # decode each node
        x_ = self.fc2(h_)  # (B, Tmax, hdim, odim)
        if kernel_output:
            if normalize:
                denom = torch.norm(x_, dim=-1, keepdim=True)  # (B, Tmax, hdim, odim)
            else:
                denom = 1.0
            kernel = torch.matmul(x_ / denom, (x_ / denom).transpose(-1, -2))  # (B, Tmax, hdim, hdim)
        else:
            kernel = None

        return kernel, x_

    def forward(self, x, y):
        # # 0. prepare data
        # batch_size = x.size(0)
        # time_size = x.size(1)

        # 1. hidden space control via freq mask with some distribution
        h, m = self.freq_mask(self.fc1(x))

        # 2. disentangle by screening out hidden space without self-node
        # ToDo: disentangled feature constrained needs
        kernel, x_dis = self.disentangle(h)

        # 3. each disentangled feature search location of optimal scale and shift arguments
        x_dis, sim_max_global, p_augs_global = self.argaug(x_dis, y)

        # 4. feed forward each disentangled feature for matching to target feature
        h = self.fc1(x_dis)
        x = torch.sum(self.fc2(h), dim=-2)

        # 5. compute loss
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))
        loss = self.criterion(x, y)

        # appendix. for reporting some value or tensor
        self.reporter.report_dict['augs_p'] = p_augs_global[0, :, 5].detach().cpu().numpy()
        self.reporter.report_dict['augs_sim'] = sim_max_global[0, :, 5].detach().cpu().numpy()
        if kernel is not None:
            self.reporter.report_dict['distang'] = kernel[20].detach().cpu().numpy()

        return loss, x
