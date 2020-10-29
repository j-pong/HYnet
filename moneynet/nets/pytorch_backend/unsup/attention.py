#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch


def attention(x, y, temper, pseudo_zero=1e-6):
    score = (x * y)
    score[score < pseudo_zero] = pseudo_zero
    score = score / score.sum(dim=-1, keepdim=True)
    score = torch.exp(torch.log(score) / temper)
    attn = score / score.sum(dim=-1, keepdim=True)
    return attn
