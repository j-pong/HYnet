#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F

from torch import nn

import numpy as np


def initialize(model, init_type="xavier_normal"):
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


class SeqLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss(reduce=None)):
        super(SeqLoss, self).__init__()
        self.criterion = criterion

    def forward(self, x, y, mask, reduce='mean'):
        denom = (~mask).float().sum()
        loss = self.criterion(input=x, target=y)
        loss = loss.masked_fill(mask, 0) / denom
        if reduce == 'mean':
            return loss.sum()
        elif reduce is None:
            return loss
        else:
            raise ValueError("{} type reduction is not defined".format(reduce))


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

        # training hyperparameter
        self.measurement = args.similarity
        self.temper = args.temperature

        # next frame predictor
        self.encoder = nn.Linear(idim, self.hdim)
        self.relu = nn.ReLU()
        self.decoder_src = nn.Linear(self.hdim, odim)
        self.decoder_self = nn.Linear(self.hdim, odim)
        # self.q = torch.nn.parameter.Parameter(torch.eye(odim, dtype=torch.float32))

        # network training related
        self.criterion = SeqLoss(criterion=nn.MSELoss(reduce=None))
        self.criterion_h = SeqLoss(criterion=nn.MSELoss(reduce=None))

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
    def _reverse_pad_for_shift(key, query, theta):
        """Reverse to padded data

        :param torch.Tensor key: batch of padded source sequences (B, Tmax, idim_k)
        :param torch.Tensor query: batch of padded source sequences (B, Tmax, idim_k)
        :param torch.Tensor theta: batch of padded source sequences (B, Tmax)

        :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k)
        :rtype: torch.Tensor
        """
        idim_k = key.size(-1)
        idim_q = query.size(-1)
        pad = idim_q - 1

        key_pad = F.pad(key, pad=[pad, pad])  # (B, Tmax, idim_k + pad * 2)
        theta = theta.long().view(-1)
        key_pad_trunk = []
        for i in six.moves.range(idim_k + idim_q - 1):
            kpt = key_pad[..., i:i + idim_q]
            key_pad_trunk.append(kpt.unsqueeze(-2))
        key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
        key_pad_trunk = key_pad_trunk.view(-1, idim_k + idim_q - 1, idim_q)
        key_pad_trunk = key_pad_trunk[torch.arange(key_pad_trunk.size(0)), 2 * idim_k - 2 - theta]

        return key_pad_trunk.view(key.size(0), key.size(1), idim_k)

    @staticmethod
    def _score(key_pad_trunk, query, query_mask=None, measurement='cos'):
        """Measuring similarity of each other tensor

        :param torch.Tensor key_pad_trunk: batch of padded source sequences (B, Tmax, ... , idim_k)
        :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)
        :param torch.Tensor query_mask: batch of padded source sequences (B, Tmax, ... , idim_k)
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
            denom = (torch.norm(query, dim=-1) * torch.norm(key_pad_trunk, dim=-1) + 1e-6)
            sim = torch.sum(query * key_pad_trunk, dim=-1) / denom  # (B, Tmax, ...)
            # nan filtering
            mask = torch.isnan(sim)
            sim[mask] = 0.0
            # optimal similarity point search
            sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax)
        else:
            raise AttributeError('{} is not support yet'.format(measurement))

        return sim_max, sim_max_idx

    @staticmethod
    def temp_softmax(x, T=10.0, dim=-1):
        x = x / T
        max_x = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - max_x)
        x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
        return x

    def sim_argmax(self, x, y, measurement='cos'):
        batch_size = x.size(0)
        time_size = x.size(1)

        # similarity measuring
        sim_max, sim_max_idx = self._score(key_pad_trunk=x, query=y, query_mask=None,
                                           measurement=measurement)  # (B, Tmax)

        # maximum shift select
        x = x.view(-1, x.size(-2), x.size(-1))  # (B * Tmax, idim_k + idim_q - 1, idim_q)
        x = x[torch.arange(x.size(0)), sim_max_idx.view(-1)]  # (B * Tmax, idim_q)

        # recover shape
        x = x.view(batch_size, time_size, -1)
        return x, sim_max, sim_max_idx

    def attention(self, x, y, temper):
        denom = (torch.norm(x, dim=-1, keepdim=True) * torch.norm(y, dim=-1, keepdim=True) + 1e-6)
        score = x * y / denom
        attn = self.temp_softmax(score, T=temper, dim=-1).detach()
        attn[torch.isnan(attn)] = 0.0
        return attn

    def hsr(self, h, mask_prev, seq_mask, mask=True):
        if mask:
            energy_h = h.pow(2)
            if mask_prev is None:
                indices_cur = torch.topk(energy_h, k=self.cdim, dim=-1)[1]
                mask_cur = F.one_hot(indices_cur, num_classes=self.hdim).float().sum(-2)
                mask_prev = mask_cur
                loss_h = None
            else:
                assert mask_prev is not None
                # intersection of prev and current hidden space
                indices_cur = torch.topk(energy_h, k=self.cdim, dim=-1)[1]
                mask_cur = F.one_hot(indices_cur, num_classes=self.hdim).float().sum(-2)
                mask_intersection = mask_prev * mask_cur
                seq_mask = seq_mask.prod(-1).unsqueeze(-1).repeat(1, 1, self.hdim).bool()
                # loss define
                h_ = h.clone()
                h_.retain_grad()
                loss_h = self.criterion_h(h_.view(-1, self.hdim),
                                          (1.0 - mask_intersection).view(-1, self.hdim),
                                          mask=seq_mask.view(-1, self.hdim),
                                          reduce=None)
                loss_h = loss_h.masked_fill(~(mask_intersection.view(-1, self.hdim).bool()), 0.0).sum()
                # eliminate fired hidden nodes
                h[mask_prev.bool()] = 0.0
                energy_h = h.pow(2)
                indices_cur = torch.topk(energy_h, k=self.cdim, dim=-1)[1]
                mask_cur = F.one_hot(indices_cur, num_classes=self.hdim).float().sum(-2)
                mask_prev = mask_prev + mask_cur
        else:
            pass
        h = h.masked_fill(~(mask_cur.bool()), 0.0)
        return h, mask_prev, loss_h

    def forward(self, x, y, pretrain=True):
        self.reporter.report_dict['target'] = y[0].detach().cpu().numpy()
        # 0. prepare data
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(y)
            y = y.masked_fill(seq_mask, self.ignore_in)
        else:
            seq_mask = y == self.ignore_out

        buffs = {'y_dis': [], 'x_dis': [],
                 'theta_opt': [], 'sim_opt': [],
                 'loss': [], 'attn': [], 'hs': []}

        # iterative method for subtraction
        y_res = y.clone()
        x_res = x.clone()
        mask_prev_self = None
        mask_prev_src = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            # 1. attention x_res and y_res matching with transform for self disentangling
            x_aug, _ = self._pad_for_shift(key=x_res, query=y_res)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
            x_opt, sim_opt, theta_opt = self.sim_argmax(x_aug, y_res, measurement=self.measurement)
            attn = self.attention(x_opt, y_res, temper=self.temper)
            x_attn = x_opt * attn
            x_ele = self._reverse_pad_for_shift(key=x_attn, query=y_res, theta=theta_opt)
            buffs['theta_opt'].append(theta_opt[0].unsqueeze(-1))
            buffs['sim_opt'].append(sim_opt[0].unsqueeze(-1))
            buffs['attn'].append(attn[0].unsqueeze(-1))

            # 2. feedforward for self estimation
            h_self = self.encoder(x_ele)
            h_self, mask_prev_self, loss_h_self = self.hsr(h_self, mask_prev_self, seq_mask=seq_mask)
            x_ele_relation = self.decoder_self(h_self)
            buffs['x_dis'].append(x_ele_relation.unsqueeze(-1))

            # 3. compute self estimation loss
            loss_local_self = self.criterion(x_ele_relation.view(-1, self.odim),
                                             x_res.view(-1, self.odim),
                                             mask=seq_mask.view(-1, self.odim))
            if loss_h_self is not None:
                loss = loss_local_self.unsqueeze(-1) + loss_h_self.unsqueeze(-1)
            else:
                loss = loss_local_self.unsqueeze(-1)

            # 4. attention x_ele and y_res matching with transform for src disentangling
            x_aug, _ = self._pad_for_shift(key=x_ele_relation, query=y_res)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
            x_ele_opt, sim_opt, theta_opt = self.sim_argmax(x_aug, y_res, measurement=self.measurement)
            attn = self.attention(x_ele_opt, y_res, temper=self.temper)
            x_attn = x_ele_opt * attn
            x_ele = self._reverse_pad_for_shift(key=x_attn, query=y_res, theta=theta_opt)

            # 5. feedforward for src estimation
            h_src = self.encoder(x_attn)
            h_src, mask_prev_src, loss_h_src = self.hsr(h_src, mask_prev_src, seq_mask=seq_mask)
            y_ele = self.decoder_self(h_src)
            buffs['y_dis'].append(y_ele.unsqueeze(-1))

            # 6. compute src estimation loss
            move_energy = torch.abs(theta_opt - self.odim + 1).view(-1, 1) + 1.0
            loss_local_src = self.criterion(y_ele.view(-1, self.odim),
                                            y_res.view(-1, self.odim),
                                            mask=seq_mask.view(-1, self.odim), reduce=None)
            loss_local_src = loss_local_src / move_energy
            if loss_h_src is not None:
                loss += loss_local_src.sum().unsqueeze(-1) + loss_h_src.unsqueeze(-1)
            else:
                loss += loss_local_src.sum().unsqueeze(-1)
            buffs['loss'].append(loss)

            # 7. compute residual feature
            y_res = (y_res - y_ele).detach()
            x_res = (x_res - x_ele).detach()

        # 6. total loss compute
        loss = torch.cat(buffs['loss'], dim=-1).mean()
        self.reporter.report_dict['loss'] = float(loss)

        # appendix. for reporting some value or tensor
        # batch side sum needs for check
        x_dis = torch.cat(buffs['x_dis'], dim=-1)
        y_dis = torch.cat(buffs['y_dis'], dim=-1)
        loss_x = self.criterion(x_dis.sum(-1).view(-1, self.idim),
                                x.view(-1, self.idim),
                                mask=seq_mask.view(-1, self.odim))
        loss_y = self.criterion(y_dis.sum(-1).view(-1, self.idim),
                                y.view(-1, self.idim),
                                mask=seq_mask.view(-1, self.odim))
        self.reporter.report_dict['loss_x'] = float(loss_x)
        self.reporter.report_dict['loss_y'] = float(loss_y)
        self.reporter.report_dict['pred_y'] = y_dis.sum(-1)[0].detach().cpu().numpy()
        self.reporter.report_dict['pred_x'] = x_dis.sum(-1)[0].detach().cpu().numpy()
        self.reporter.report_dict['res_x'] = x_res[0].detach().cpu().numpy()

        # just one sample at batch should be check
        theta_opt = torch.cat(buffs['theta_opt'], dim=-1)
        sim_opt = torch.cat(buffs['sim_opt'], dim=-1)
        # energy_y = y_dis.pow(2).sum(-2)

        self.reporter.report_dict['theta_opt'] = theta_opt.detach().cpu().numpy()
        self.reporter.report_dict['sim_opt'] = sim_opt.detach().cpu().numpy()
        # self.reporter.report_dict['energy_y'] = np.log(energy_y[0].detach().cpu().numpy() + 1e-6)
        #
        # """
        # New block for testing hidden space
        # """
        # hs = torch.cat(buffs['h_src'], dim=-1)
        # self.reporter.report_dict['h_src0'] = hs[:, :, 0].detach().cpu().numpy()
        # self.reporter.report_dict['h_src1'] = hs[:, :, 1].detach().cpu().numpy()
        # self.reporter.report_dict['h_src2'] = hs[:, :, 2].detach().cpu().numpy()
        # self.reporter.report_dict['h_src3'] = hs[:, :, 3].detach().cpu().numpy()
        # self.reporter.report_dict['h_src4'] = hs[:, :, 4].detach().cpu().numpy()
        #
        """
        New block for testing attention
        """
        attns = torch.cat(buffs['attn'], dim=-1)
        self.reporter.report_dict['attn0'] = attns[:, :, 0].detach().cpu().numpy()
        self.reporter.report_dict['attn1'] = attns[:, :, 1].detach().cpu().numpy()
        self.reporter.report_dict['attn2'] = attns[:, :, 2].detach().cpu().numpy()
        self.reporter.report_dict['attn3'] = attns[:, :, 3].detach().cpu().numpy()
        self.reporter.report_dict['attn4'] = attns[:, :, 4].detach().cpu().numpy()
        """
        Checker block
        """
        # seq_mask_ = ~(seq_mask)
        # seq_mask_ = (seq_mask_.float().sum(-1) > 0).unsqueeze(-1).repeat(1, 1, self.hdim).bool()
        # print(mask_prev_self[0][seq_mask_[0]].view(-1, mask_prev_self.size(2)).sum(-1),
        #       seq_mask_[0].sum(-1))
        # exit()

        return loss

    def forward_self(self, x, y):
        if np.isnan(self.ignore_out):
            seq_mask = torch.isnan(y)
        else:
            seq_mask = y == self.ignore_out

        buffs = {'y_dis': [], 'x_dis': [],
                 'theta_opt': [], 'sim_opt': [],
                 'loss': [], 'attn': [], 'hs': []}

        # iterative method for subtraction
        x_res = x.clone()
        mask_prev_self = None
        mask_prev_src = None
        for _ in six.moves.range(int(self.hdim / self.cdim)):
            attn = self.attention(x_res, x_res, temper=self.temper)
            buffs['attn'].append(attn[0].unsqueeze(-1))
            x_attn = x_res * attn

            h_self = self.encoder(x_attn)
            h_self, mask_prev_self, loss_h_self = self.hsr(h_self, mask_prev_self, seq_mask=seq_mask)
            x_ele = self.decoder_self(h_self)
            buffs['x_dis'].append(x_ele.unsqueeze(-1))
            loss_local_self = self.criterion(x_ele.view(-1, self.odim),
                                             x_res.view(-1, self.odim),
                                             mask=seq_mask.view(-1, self.odim))
            if loss_h_self is not None:
                loss = loss_local_self.unsqueeze(-1) + loss_h_self.unsqueeze(-1)
            else:
                loss = loss_local_self.unsqueeze(-1)

            x_res = (x_res - x_ele).detach()

            buffs['loss'].append(loss)

        # 6. total loss compute
        loss = torch.cat(buffs['loss'], dim=-1).mean()
        self.reporter.report_dict['loss'] = float(loss)
