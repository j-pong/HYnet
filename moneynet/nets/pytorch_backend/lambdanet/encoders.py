#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Dong Hyun Kim
#  Apache 2.0  (http://idimw.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.repeat import repeat

from moneynet.nets.pytorch_backend.lambdanet.encoder_layer import EncoderLayer
from moneynet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling

class Encoder(nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, n_feat, n_head, num_blocks, dropout_rate):
        super(Encoder, self).__init__()
        self.embed = VGG2L(idim, n_feat)
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                n_feat, n_head, dropout_rate
            )
        )

    def forward(self, x, mask=None):
        x, mask = self.embed(x, mask)
        x, mask = self.encoders(x)
        return x
