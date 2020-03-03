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
            elif init_type == "identity":
                pass
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

    def forward(self, x, y, mask):
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
        self.cdim = args.cdim
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out

        # simple network add
        self.fc1 = nn.Linear(idim, self.hdim)
        self.fc2 = nn.Linear(self.hdim, odim)
        self.q = nn.Linear(self.idim, self.odim, bias=False)

        # network training related
        self.criterion = NetLoss(ignore_in=self.ignore_in, ignore_out=self.ignore_out)

        # initialize parameter
        self.reset_parameters()

    def reset_parameters(self):
        initialize(self)

    @staticmethod
    def _pad_for_shift(key, query):
        """Padding to channel dim for convolution

        :param torch.Tensor key: batch of padded source sequences (B, Tmax, idim_k)
        :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)

        :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k + idim_q - 1, idim_k)
        :rtype: torch.Tensor
        :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k + idim_q - 1, idim_k)
        :rtype: torch.Tensor
        """
        idim_k = key.size(-1)
        idim_q = query.size(-1)
        pad = idim_q - 1

        key_pad = F.pad(key, pad=[pad, pad])  # (B, Tmax, idim_k + pad * 2)
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
        key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
        query_mask = torch.cat(query_mask, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
        return key_pad_trunk, query_mask

    @staticmethod
    def _reverse_pad_for_shift(key, query, p_augs_global):
        """

        :param torch.Tensor key: batch of padded source sequences (B, Tmax, idim_k)

        :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k + idim_q - 1, idim_k)
        :rtype: torch.Tensor
        """
        # ToDo: Don't use this function when scale to key yet

        idim_k = key.size(-1)
        idim_q = query.size(-1)
        pad = idim_q - 1

        key_pad = F.pad(key, pad=[pad, pad])  # (B, Tmax, idim_k + pad * 2)
        p_aug = p_augs_global.long().view(-1, 2)[:, 0]
        key_pad_trunk = []
        for i in six.moves.range(idim_k + idim_q - 1):
            kpt = key_pad[..., i:i + idim_q]
            key_pad_trunk.append(kpt.unsqueeze(-2))
        key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
        key_pad_trunk = key_pad_trunk.view(-1, idim_k + idim_q - 1, idim_q)
        key_pad_trunk = key_pad_trunk[torch.arange(key_pad_trunk.size(0)), 2 * idim_k - 2 - p_aug]

        return key_pad_trunk.view(key.size(0), key.size(1), idim_k)

    @staticmethod
    def _simiality(key_pad_trunk, query, query_mask=None, measurement='cos'):
        """Measuring similarity of each other tensor

        :param torch.Tensor key_pad_trunk: batch of padded source sequences (B, Tmax, idim_k + idim_q - 1, idim_k)
        :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)
        :param torch.Tensor query_mask: batch of padded source sequences (B, Tmax, idim_k + idim_q - 1, idim_k)
        :param string measurement:

        :return: max similarity value of sequence (B, Tmax)
        :rtype: torch.Tensor
        :return: max similarity index of sequence  (B, Tmax)
        :rtype: torch.Tensor
        """
        query = query.unsqueeze(-2)
        if query_mask is not None:
            query = query * query_mask
        if measurement == 'cos':
            denom = torch.norm(query, dim=-1) * torch.norm(key_pad_trunk, dim=-1)
            sim = torch.sum(query * key_pad_trunk, dim=-1) / denom  # (B, Tmax, idim_k + idim_q - 1)
            # nan filtering
            mask = torch.isnan(sim)
            sim[mask] = 0.0
            # optimal similarity point search
            sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax)
        else:
            raise AttributeError('{} is not support yet'.format(measurement))

        return sim_max, sim_max_idx

    def argaug(self, x, y, measurement='cos', scale=False):
        """Find augmentation parameter using grid search

        :param torch.Tensor x: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor y: batch of padded target sequences (B, Tmax, idim)
        :param string measurement:
        :param bool scale:

        :return: augmented source sequences (B, Tmax, idim)
        :rtype: torch.Tensor
        :return: value of augmentation parameter (B, Tmax, *)
        :rtype: torch.Tensor
        :return: value of similarity (B, Tmax)
        :rtype: torch.Tensor
        """
        global sim_max_global, p_augs_global
        batch_size = x.size(0)
        time_size = x.size(1)
        x_aug = None

        if scale:
            start = 0.8
            end = 1.2
            num = 5
        else:
            start = 1.0
            end = 1.0
            num = 1

        for theta in np.linspace(start, end, num):
            if scale:
                # scale
                x_s = F.interpolate(x, scale_factor=theta)
            else:
                x_s = x
            # shift
            x_s_pad_trunk, _ = self._pad_for_shift(key=x_s, query=y)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
            # similarity measuring
            sim_max, sim_max_idx = self._simiality(key_pad_trunk=x_s_pad_trunk, query=y, query_mask=None,
                                                   measurement=measurement)  # (B, Tmax)
            # aug parameters
            p_aug_shift = sim_max_idx.view(-1, 1).float()
            p_aug_scale = torch.tensor([theta]).view(1, 1).repeat(repeats=p_aug_shift.size()).to(p_aug_shift.device)
            p_augs = torch.cat([p_aug_shift, p_aug_scale], dim=-1)  # (B * Tmax, 2)
            # maximum shift select
            xspt_size = x_s_pad_trunk.size()
            x_s_pad_trunk = x_s_pad_trunk.view(-1,
                                               xspt_size[-2],
                                               xspt_size[-1])  # (B * Tmax, idim_k + idim_q - 1, idim_q)
            x_s_opt = x_s_pad_trunk[
                torch.arange(x_s_pad_trunk.size(0)), sim_max_idx.view(-1)]  # (B * Tmax, idim_q)
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
        x_aug = x_aug.view(batch_size, time_size, -1)
        p_augs_global = p_augs_global.view(batch_size, time_size, -1)
        return x_aug, sim_max_global, p_augs_global

    def forward(self, x, y):
        self.reporter.report_dict['target'] = y[0].detach().cpu().numpy()
        # 0. prepare data
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(y)
            y = y.masked_fill(seq_mask, self.ignore_in)
        else:
            seq_mask = y == self.ignore_out

        # display buffer
        y_dis = []
        x_dis = []
        p_augs_s = []
        sim_max_s = []
        # loss buffer
        loss_self = []
        loss_src = []

        # iterative method for subtraction
        y_res = y.clone()
        x_res = x.clone()
        for i in six.moves.range(int(self.hdim / self.cdim)):
            # 1. augment and search optimal aug-parameter
            # (B, Tmax, idim) -> (B, Tmax, idim)
            x_aug, sim_max_global, p_augs_global = self.argaug(x_res, y_res, scale=False)
            p_augs_s.append(p_augs_global.unsqueeze(-1))
            sim_max_s.append(sim_max_global.unsqueeze(-1))

            # 2. attention mask
            attn = torch.softmax(x_aug * y_res, dim=-1).detach()
            attn[torch.isnan(attn)] = 0.0
            x_attn = x_aug * attn
            y_attn = y_res * attn

            # aux1. reverse attention x_aug feature
            x_ele = self._reverse_pad_for_shift(x_attn, y_res, p_augs_global)

            # 3. subtract inference feature and residual re-match to new one (This function act like value scale)
            h = self.fc1(x_attn)
            # get mask
            mask = torch.zeros_like(h)
            mask[..., i * self.cdim:(i + 1) * self.cdim] = 1.0
            mask = mask.to(h.device).float()
            # apply mask
            y_ele = self.fc2(h * mask)

            # 4. compute loss of residual feature
            loss_src.append(self.criterion(y_ele.view(-1, self.odim),
                                           y_res.view(-1, self.odim),
                                           mask=seq_mask).unsqueeze(-1))

            # 5. compute residual feature
            y_res = y_res - y_ele
            x_res = x_res - x_ele

            # aux 2. inference result save as each nodes fire
            y_dis.append(y_ele.unsqueeze(-1))

        # 6. total loss compute
        loss = torch.cat(loss_src, dim=-1).mean()

        # appendix. for reporting some value or tensor
        p_augs_s = torch.cat(p_augs_s, dim=-1)
        sim_max_s = torch.cat(sim_max_s, dim=-1)
        self.reporter.report_dict['augs_p'] = p_augs_s[0, :, :, 1].detach().cpu().numpy()
        self.reporter.report_dict['augs_sim'] = sim_max_s[0, :, 1].detach().cpu().numpy()
        y_dis = torch.cat(y_dis, dim=-1)
        energy = y_dis.pow(2).sum(-2)
        self.reporter.report_dict['disentangle'] = np.log(energy[0].detach().cpu().numpy() + 1e-6)

        self.reporter.report_dict['loss'] = float(loss)
        self.reporter.report_dict['pred'] = y_dis.sum(-1)[0].detach().cpu().numpy()

        return loss

    def pretrain_forward(self, x):
        h, m = self.freq_mask(self.fc1(x))
        x_hat = self.fc2(h)

        loss = self.criterion(x_hat.view(-1, self.odim), x.view(-1, self.odim))

        return loss
