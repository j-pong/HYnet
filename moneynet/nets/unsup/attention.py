#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

from moneynet.nets.unsup.utils import temp_softmax


def attention(x, y, temper, pseudo_zero=1e-6):
    denom = (torch.norm(x, dim=-1, keepdim=True) * torch.norm(y, dim=-1, keepdim=True) + 1e-6)
    score = x * y / denom
    attn = temp_softmax(score, T=temper, dim=-1).detach()
    attn[torch.isnan(attn)] = 0.0
    # energy = torch.pow(x, 2).sum(dim=-1, keepdim=True)
    # mask_trivial = energy < pseudo_zero
    # score = (x * y) / (x * y).sum(dim=-1, keepdim=True)
    # score = torch.exp(torch.log(score) / temper)
    # attn = score / score.sum(dim=-1, keepdim=True)
    # attn = attn.masked_fill(mask_trivial, 0.0)
    return attn
