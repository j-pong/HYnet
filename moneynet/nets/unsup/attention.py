#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from moneynet.nets.unsup.utils import temp_softmax


def attention(x, y, temper):
    denom = (torch.norm(x, dim=-1, keepdim=True) * torch.norm(y, dim=-1, keepdim=True) + 1e-6)
    score = x * y / denom
    attn = temp_softmax(score, T=temper, dim=-1).detach()
    attn[torch.isnan(attn)] = 0.0
    return attn
