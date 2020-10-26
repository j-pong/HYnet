#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import math
import torch
import torch.nn.functional as F

from torch import nn, einsum

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, n_feat, n_head, dropout_rate):
        super(EncoderLayer, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.d_v = n_feat // n_head
        self.d_u = 1
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, self.d_u * self.d_k)
        self.linear_v = nn.Linear(n_feat, self.d_u * self.d_v)
        self.norm_q = nn.BatchNorm1d(n_feat)
        self.norm_v = nn.BatchNorm1d(self.d_k * self.d_u)
        self.pos_emb = nn.Conv2d(self.d_u, self.d_k, 3, padding=1, bias=False)
        # self.pos_emb = nn.Linear(n_feat, n_feat)    # TODO: conv
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)
        self.local_contexts = False

    def forward(self, x, mask=None):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        query, key, value = (x, x, x)
        n_batch = query.size(0)
        q = self.norm_q(self.linear_q(query).view(-1, self.h * self.d_k)).view(n_batch, self.h, self.d_k, -1)
        k = self.linear_k(key).view(n_batch, self.d_u, self.d_k, -1)
        v = self.norm_v(self.linear_v(value).view(-1, self.d_u * self.d_v)).view(n_batch, self.d_u, self.d_v, -1)

        k = k.softmax(dim=-1)

        # TODO: plot λc
        if self.local_contexts:
            # TODO: Local Context
            return
        else:
            λc = einsum('b u k t, b u v t -> b k v', k, v)  # (batch, d_k, d_v)
            Yc = einsum('b h k t, b k v -> b h v t', q, λc)  # (batch, head, d_v, time1)

        λp = F.relu(self.pos_emb(v))  # (batch, d_k, d_v, time2)
        Yp = einsum('b h k t, b k v t -> b h v t', q, λp)

        Yc = Yc.contiguous().view(n_batch, self.h * self.d_k, -1)
        Yp = Yp.contiguous().view(n_batch, self.h * self.d_k, -1)
        Yc = Yc + Yp  # (batch, n_feat, time1)
        Yc = Yc.transpose(1,2)

        # TODO: dropout?
        return self.linear_out(Yc), mask  # (batch, time1, n_feat)
