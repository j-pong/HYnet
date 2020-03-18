#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import six

import torch
import torch.nn.functional as F


def select_with_ind(x, ind):
    # get size of tensors
    x_size = x.size()
    ind_size = ind.size()
    assert len(x_size) > len(ind_size)

    # ind type check and change
    if ind.type() is not torch.LongTensor:
        ind = ind.long()

    # get new size
    x_new_size = []
    x_return_size = []
    for i, ind_s in enumerate(ind_size):
        if i == 0:
            ind_s_sum = ind_s
        else:
            ind_s_sum *= ind_s
        x_return_size.append(ind_s)
    x_new_size.append(ind_s_sum)
    for i, x_s in enumerate(x_size[i + 1:]):
        x_new_size.append(x_s)
        if i != 0:
            x_return_size.append(x_s)

    # select with ind
    x = x.view(x_new_size)
    ind = ind.view(x_new_size[0])
    x = x[torch.arange(x_new_size[0]), ind].view(x_return_size)

    return x


def pad_for_shift(key, pad, window=None, mask=False):
    """Padding to channel dim for convolution

    :param torch.Tensor key: batch of padded source sequences (B, Tmax, idim_k)
    :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)

    :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k + idim_q - 1, idim_k)
    :rtype: torch.Tensor
    :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k + idim_q - 1, idim_k)
    :rtype: torch.Tensor
    """
    idim_k = key.size(-1)
    if window is None:
        window = pad + 1
    key_pad = F.pad(key, pad=(pad, pad))  # (B, Tmax, idim_k + pad * 2)
    key_pad_trunk = []
    trunk_mask = []
    for i in six.moves.range(idim_k + 2 * pad - window + 1):
        kpt = key_pad[..., i:i + window]
        if mask:
            kpt_mask = torch.zeros_like(kpt).to(kpt.device)
            end = -max(i - window, 0)
            if end < 0:
                kpt_mask[..., -(i + 1):end] = 1
            else:
                kpt_mask[..., -(i + 1):] = 1
            trunk_mask.append(kpt_mask.unsqueeze(-2))
        key_pad_trunk.append(kpt)
    key_pad_trunk = torch.stack(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
    if mask:
        trunk_mask = torch.stack(trunk_mask, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
    else:
        trunk_mask = None
    return key_pad_trunk, trunk_mask


def reverse_pad_for_shift(key, pad, theta, window=None):
    """Reverse to padded data

    :param torch.Tensor key: batch of padded source sequences (B, Tmax, idim_k)
    :param torch.Tensor query: batch of padded source sequences (B, Tmax, idim_k)
    :param torch.Tensor theta: batch of padded source sequences (B, Tmax)

    :return: padded and truncated tensor that matches to query dim (B, Tmax, idim_k)
    :rtype: torch.Tensor
    """
    idim_k = key.size(-1)
    # ToDo: other case of pad and window is not concerned at current reverse algorithm.
    assert pad == idim_k - 1 and window == idim_k
    key_pad = F.pad(key, pad=(pad, pad))  # (B, Tmax, idim_k + pad * 2)
    key_pad_trunk = []
    for i in six.moves.range(idim_k + 2 * pad - window + 1):
        kpt = key_pad[..., i:i + window]
        key_pad_trunk.append(kpt.unsqueeze(-2))
    key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
    key_pad_trunk = select_with_ind(key_pad_trunk, 2 * pad - theta)

    return key_pad_trunk.view(key.size(0), key.size(1), -1)


def selector(x, y, measurement='cos'):
    """Measuring similarity of each other tensor

        :param torch.Tensor x: batch of padded source sequences (B, Tmax, * , c)
        :param torch.Tensor y: batch of padded target sequences (B, Tmax, c)
        :param string measurement:

        :return: max similarity of x (B, Tmax, c)
        :rtype: torch.Tensor
        :return: max similarity value of sequence (B, Tmax)
        :rtype: torch.Tensor
        :return: max similarity index of sequence  (B, Tmax)
        :rtype: torch.Tensor
        """
    y = y.unsqueeze(-2)
    if measurement == 'cos':
        denom = (torch.norm(y, dim=-1) * torch.norm(x, dim=-1) + 1e-6)
        sim = torch.sum(y * x, dim=-1) / denom  # (B, Tmax, *)
        sim[torch.isnan(sim)] = 0.0
        sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax)
    else:
        raise AttributeError('{} is not support yet'.format(measurement))
    # maximum shift select
    x = select_with_ind(x, sim_max_idx)
    return x, sim_max, sim_max_idx


def temp_softmax(x, T=10.0, dim=-1):
    x = x / T
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return x
