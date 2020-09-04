import logging
import six
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn as nn
from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device


def get_same_pad(k, s, d):
    assert not (s > 1 and d > 1)
    if s > 1:
        return (k-s+1)//2
    return (k-1)*d//2


class CNN(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, lnorm, bnorm, typ="blstm"):
        super(CNN, self).__init__()
        bidir = typ[0] == "b"

        dim1_2d_conv = 128
        dim2_2d_conv = 256

        in_channel = 1
        self.in_channel = in_channel

        cdim_mid = int(dim2_2d_conv * idim / 8)

        cdim_1_1 = int(cdim/2)
        cdim_1_2 = int(cdim/4)
        cdim_1_3 = cdim - cdim_1_1 - cdim_1_2

        dilation1 = 1
        dilation2 = 4
        dilation3 = 7

        kernel_size1_1 = 5
        kernel_size1_2 = 5
        kernel_size1_3 = 5

        dropout1 = 0.1

        conv2d_1 = torch.nn.Conv2d(self.in_channel, dim1_2d_conv, 3, stride=1, padding=1)
        #bn2d_1 = nn.BatchNorm2d(dim1_2d_conv, momentum=0.05)
        relu2d_1 = nn.ReLU()
        do2d_1 = nn.Dropout(p=dropout1)
        self.conv2d_1 = nn.Sequential(conv2d_1, relu2d_1, do2d_1)
        #self.conv2d_1 = nn.Sequential(conv2d_1, relu2d_1, bn2d_1, do2d_1)

        conv2d_2 = torch.nn.Conv2d(dim1_2d_conv, dim2_2d_conv, 3, stride=1, padding=1)
        #bn2d_2 = nn.BatchNorm2d(dim2_2d_conv, momentum=0.05)
        relu2d_2 = nn.ReLU()
        do2d_2 = nn.Dropout(p=dropout1)
        self.conv2d_2 = nn.Sequential(conv2d_2, relu2d_2, do2d_2)
        #self.conv2d_2 = nn.Sequential(conv2d_2, relu2d_2, bn2d_2, do2d_2)

        conv2d_3 = torch.nn.Conv2d(dim2_2d_conv, dim2_2d_conv, 3, stride=(1,2), padding=1)
        #bn2d_3 = nn.BatchNorm2d(dim2_2d_conv, momentum=0.05)
        relu2d_3 = nn.ReLU()
        do2d_3 = nn.Dropout(p=dropout1) 
        self.conv2d_3 = nn.Sequential(conv2d_3, relu2d_3, do2d_3)
        #self.conv2d_3 = nn.Sequential(conv2d_3, relu2d_3, bn2d_3, do2d_3)

        conv2d_4 = torch.nn.Conv2d(dim2_2d_conv, dim2_2d_conv, 3, stride=(1,2), padding=1)
        #bn2d_4 = nn.BatchNorm2d(dim2_2d_conv, momentum=0.05)
        relu2d_4 = nn.ReLU()
        do2d_4 = nn.Dropout(p=dropout1)
        self.conv2d_4 = nn.Sequential(conv2d_4, relu2d_4, do2d_4)
        #self.conv2d_4 = nn.Sequential(conv2d_4, relu2d_4, bn2d_4, do2d_4)

        conv2d_5 = torch.nn.Conv2d(dim2_2d_conv, dim2_2d_conv, 3, stride=(1,2), padding=1)
        #bn2d_5 = nn.BatchNorm2d(dim2_2d_conv, momentum=0.05)
        relu2d_5 = nn.ReLU()
        do2d_5 = nn.Dropout(p=dropout1)
        self.conv2d_5 = nn.Sequential(conv2d_5, relu2d_5, do2d_5)
        #self.conv2d_5 = nn.Sequential(conv2d_5, relu2d_5, bn2d_5, do2d_5)

        self.fc1 = torch.nn.Linear(cdim, cdim) 
        self.l_last = torch.nn.Linear(cdim, hdim)

        self.conv1d_1_1 = torch.nn.Conv1d(in_channels=cdim_mid, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_1_2 = torch.nn.Conv1d(in_channels=cdim_mid, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_1_3 = torch.nn.Conv1d(in_channels=cdim_mid, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_2_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_2_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_2_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_3_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_3_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_3_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_4_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_4_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_4_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_5_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_5_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_5_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_6_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_6_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_6_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_7_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_7_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_7_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_8_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_8_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_8_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_9_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_9_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_9_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_10_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_10_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_10_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_11_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_11_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_11_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_12_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_12_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_12_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_13_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_13_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_13_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_14_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_14_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_14_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)
        
        self.conv1d_15_1 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_1, kernel_size=kernel_size1_1, padding=get_same_pad(kernel_size1_1, 1, dilation1), dilation=dilation1)
        self.conv1d_15_2 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_2, kernel_size=kernel_size1_2, padding=get_same_pad(kernel_size1_2, 1, dilation2), dilation=dilation2)
        self.conv1d_15_3 = torch.nn.Conv1d(in_channels=cdim, out_channels=cdim_1_3, kernel_size=kernel_size1_3, padding=get_same_pad(kernel_size1_3, 1, dilation3), dilation=dilation3)

        # Layer Normalization
        if lnorm:
            self.layer_norm = torch.nn.LayerNorm(hdim)
        if bnorm:
            self.batch_norm = torch.nn.BatchNorm1d(hdim)

        self.typ = typ
        self.lnorm = lnorm
        self.bnorm = bnorm

    def forward(self, xs_pad, ilens, prev_state=None):
        """CNN forward
        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))

        #if prev_state is not None and self.nbrnn.bidirectional:
        #    # We assume that when previous state is passed,
        #    # it means that we're streaming the input
        #    # and therefore cannot propagate backward BRNN state
        #    # (otherwise it goes in the wrong direction)
        #    prev_state = reset_backward_rnn_state(prev_state)

        # for vgg # x: utt x 1 (input channel num) x frame x dim # from [128, 243, 40] to [128, 1, 243, 40]
        xs_pad = xs_pad.view(
            xs_pad.size(0),
            xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel,
        ).transpose(1, 2)

        xs_pad = self.conv2d_1(xs_pad)
        xs_pad = self.conv2d_2(xs_pad)
        xs_pad = self.conv2d_3(xs_pad)
        xs_pad = self.conv2d_4(xs_pad)
        xs_pad = self.conv2d_5(xs_pad)
        xs_pad = xs_pad.transpose(1, 2)

        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )
        xs_pad = xs_pad.permute(0,2,1)

        ys_dil_1 = self.conv1d_1_1(xs_pad)
        ys_dil_2 = self.conv1d_1_2(xs_pad)
        ys_dil_3 = self.conv1d_1_3(xs_pad)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad)

        ys_dil_1 = self.conv1d_2_1(xs_pad_old)
        ys_dil_2 = self.conv1d_2_2(xs_pad_old)
        ys_dil_3 = self.conv1d_2_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_3_1(xs_pad_old)
        ys_dil_2 = self.conv1d_3_2(xs_pad_old)
        ys_dil_3 = self.conv1d_3_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_4_1(xs_pad_old)
        ys_dil_2 = self.conv1d_4_2(xs_pad_old)
        ys_dil_3 = self.conv1d_4_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_5_1(xs_pad_old)
        ys_dil_2 = self.conv1d_5_2(xs_pad_old)
        ys_dil_3 = self.conv1d_5_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_6_1(xs_pad_old)
        ys_dil_2 = self.conv1d_6_2(xs_pad_old)
        ys_dil_3 = self.conv1d_6_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_7_1(xs_pad_old)
        ys_dil_2 = self.conv1d_7_2(xs_pad_old)
        ys_dil_3 = self.conv1d_7_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_8_1(xs_pad_old)
        ys_dil_2 = self.conv1d_8_2(xs_pad_old)
        ys_dil_3 = self.conv1d_8_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_9_1(xs_pad_old)
        ys_dil_2 = self.conv1d_9_2(xs_pad_old)
        ys_dil_3 = self.conv1d_9_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)
        
        ys_dil_1 = self.conv1d_10_1(xs_pad_old)
        ys_dil_2 = self.conv1d_10_2(xs_pad_old)
        ys_dil_3 = self.conv1d_10_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_11_1(xs_pad_old)
        ys_dil_2 = self.conv1d_11_2(xs_pad_old)
        ys_dil_3 = self.conv1d_11_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_12_1(xs_pad_old)
        ys_dil_2 = self.conv1d_12_2(xs_pad_old)
        ys_dil_3 = self.conv1d_12_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)
        
        ys_dil_1 = self.conv1d_13_1(xs_pad_old)
        ys_dil_2 = self.conv1d_13_2(xs_pad_old)
        ys_dil_3 = self.conv1d_13_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_14_1(xs_pad_old)
        ys_dil_2 = self.conv1d_14_2(xs_pad_old)
        ys_dil_3 = self.conv1d_14_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        ys_dil_1 = self.conv1d_15_1(xs_pad_old)
        ys_dil_2 = self.conv1d_15_2(xs_pad_old)
        ys_dil_3 = self.conv1d_15_3(xs_pad_old)
        xs_pad = torch.cat([ys_dil_1, ys_dil_2, ys_dil_3], dim=1)
        xs_pad_old = F.relu(xs_pad_old + xs_pad)

        xs_pad = xs_pad_old.permute(0,2,1)
        xs_pad = F.relu(self.fc1(xs_pad))

        states=None

        projected = torch.tanh(
            self.l_last(xs_pad.contiguous().view(-1, xs_pad.size(2)))
        )

        xs_pad = projected.view(xs_pad.size(0), xs_pad.size(1), -1)

        return xs_pad, ilens, states


class Encoder(torch.nn.Module):
    """Encoder module

    :param str etype: type of encoder network
    :param int idim: number of dimensions of encoder network
    :param int elayers: number of layers of encoder network
    :param int eunits: number of lstm units of encoder network
    :param int eprojs: number of projection units of encoder network
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param int in_channel: number of input channels
    """

    def __init__(
        self, etype, idim, elayers, eunits, eprojs, subsample, dropout, lnorm, bnorm, in_channel=1
    ):
        super(Encoder, self).__init__()

        self.enc = torch.nn.ModuleList(
            [CNN(idim, elayers, eunits, eprojs, dropout, lnorm, bnorm, typ=typ)]
        )
        logging.info(typ.upper() + " without projection for encoder")

    def forward(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad, ilens, current_states


def encoder_for(args, idim, subsample):
    """Instantiates an encoder module given the program arguments

    :param Namespace args: The arguments
    :param int or List of integer idim: dimension of input, e.g. 83, or
                                        List of dimensions of inputs, e.g. [83,83]
    :param List or List of List subsample: subsample factors, e.g. [1,2,2,1,1], or
                                        List of subsample factors of each encoder.
                                         e.g. [[1,2,2,1,1], [1,2,2,1,1]]
    :rtype torch.nn.Module
    :return: The encoder module
    """
    num_encs = getattr(args, "num_encs", 1)  # use getattr to keep compatibility
    if num_encs == 1:
        # compatible with single encoder asr mode
        return Encoder(
            args.etype,
            idim,
            args.elayers,
            args.eunits,
            args.eprojs,
            subsample,
            args.dropout_rate,
            args.lnorm,
            args.bnorm,
        )
    elif num_encs >= 1:
        enc_list = torch.nn.ModuleList()
        for idx in range(num_encs):
            enc = Encoder(
                args.etype[idx],
                idim[idx],
                args.elayers[idx],
                args.eunits[idx],
                args.eprojs,
                subsample[idx],
                args.dropout_rate[idx],
                args.lnorm[idx],
                args.bnorm[idx],
            )
            enc_list.append(enc)
        return enc_list
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
