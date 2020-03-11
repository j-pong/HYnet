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
    x = x[torch.arange(x_new_size[0]), ind.long()].view(x_return_size)

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
    key_pad = F.pad(key, pad=(pad, pad))  # (B, Tmax, idim_k + pad * 2)
    theta = theta.long().view(-1)
    key_pad_trunk = []
    for i in six.moves.range(idim_k + 2 * pad - window + 1):
        kpt = key_pad[..., i:i + window]
        key_pad_trunk.append(kpt.unsqueeze(-2))
    key_pad_trunk = torch.cat(key_pad_trunk, dim=-2)  # (B, Tmax, idim_k + idim_q - 1, idim_q)
    key_pad_trunk = key_pad_trunk.view(-1, idim_k + 2 * pad - window + 1, window)
    key_pad_trunk = key_pad_trunk[torch.arange(key_pad_trunk.size(0)), 2 * pad - theta]

    return key_pad_trunk.view(key.size(0), key.size(1), -1)


def score(key_pad_trunk, query, query_mask=None, measurement='cos'):
    """Measuring similarity of each other tensor

    :param torch.Tensor key_pad_trunk: batch of padded source sequences (B, Tmax, * , idim_k)
    :param torch.Tensor query: batch of padded target sequences (B, Tmax, idim_q)
    :param torch.Tensor query_mask: batch of padded source sequences (B, Tmax, * , idim_k)
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
        sim = torch.sum(query * key_pad_trunk, dim=-1) / denom  # (B, Tmax, *)
        # nan filtering
        mask = torch.isnan(sim)
        sim[mask] = 0.0
        # optimal similarity point search
        sim_max, sim_max_idx = torch.max(sim, dim=-1)  # (B, Tmax)
    else:
        raise AttributeError('{} is not support yet'.format(measurement))

    return sim_max, sim_max_idx


def temp_softmax(x, T=10.0, dim=-1):
    x = x / T
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return x


def sim_argmax(x, y, measurement='cos'):
    batch_size = x.size(0)
    time_size = x.size(1)

    # similarity measuring
    sim_max, sim_max_idx = score(key_pad_trunk=x, query=y, query_mask=None,
                                 measurement=measurement)  # (B, Tmax)

    # maximum shift select
    x = x.view(-1, x.size(-2), x.size(-1))  # (B * Tmax, idim_k + idim_q - 1, idim_q)
    x = x[torch.arange(x.size(0)), sim_max_idx.view(-1)]  # (B * Tmax, idim_q)

    # recover shape
    x = x.view(batch_size, time_size, -1)
    return x, sim_max, sim_max_idx
