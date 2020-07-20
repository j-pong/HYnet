import logging
import six
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

from moneynet.nets.pytorch_backend.DEQ.nets_utils import RootFind, DEQModule

class RNN(torch.nn.Module):
    """RNN module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of final projection units
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = (
            torch.nn.LSTM(
                idim,
                cdim,
                1,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
            if "lstm" in typ
            else torch.nn.GRU(
                idim,
                cdim,
                1,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
        )
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        self.typ = typ

    def forward(self, xs_pad, ilens, prev_state=None):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        logging.debug(self.__class__.__name__ + " input lengths: " + str(ilens))
        xs_pack = pack_padded_sequence(xs_pad, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed,
            # it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state
            # (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(
            self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2)))
        )
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
        return xs_pad

class GRU(torch.nn.Module):
    def __init__(self, idim, elayers, cdim, hdim, dropout, typ="blstm"):
        super(GRU, self).__init__()

        # Reading parameters
        self.idim = idim
        self.bidir = typ.startswith('b')

        # List initialization
        self.wh = torch.nn.Linear(idim, cdim)
        self.uh = torch.nn.Linear(cdim, cdim, bias=False)

        self.wz = torch.nn.Linear(idim, cdim)  # Update Gate
        self.uz = torch.nn.Linear(cdim, cdim, bias=False)   # Update Gate

        self.wr = torch.nn.Linear(idim, cdim)  # Reset Gate
        self.ur = torch.nn.Linear(cdim, cdim, bias=False)  # Reset Gate

        if self.bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)

        self.dropout = dropout
        self.cdim = cdim
        self.odim = cdim + self.bidir * cdim

    def forward(self, xs_pad, ilens, prev_state=None):

        # Initial state and concatenation
        if self.bidir:
            raise NotImplementedError(
                "bidirection not yet supported"
            )
            # h_init = torch.zeros(2 * x.shape[0], self.gru_lay[i])
            # x = torch.cat([x, flip(x, 1)], 0)
        # else:
            # h_init = torch.zeros(xs_pad.shape[0], cdim)

        # h_init = h_init.cuda()

        # Feed-forward affine transformations (all steps in parallel)
        wh_out = self.wh(xs_pad)
        wz_out = self.wz(xs_pad)
        wr_out = self.wr(xs_pad)

        # Processing time steps
        ht = torch.zeros(xs_pad.shape[1], self.cdim).to(xs_pad.device)
        ys_pad = torch.zeros(xs_pad.size(0), xs_pad.size(1), self.odim).to(xs_pad.device)

        for k in range(xs_pad.shape[0]):

            # gru equation
            zt = torch.sigmoid(wz_out[k] + self.uz(ht))
            rt = torch.sigmoid(wr_out[k] + self.ur(ht))
            at = wh_out[k] + self.uh(rt * ht)
            hcand = F.dropout(torch.tanh(at), p=self.dropout)
            ht = zt * ht + (1 - zt) * hcand
            ys_pad[k] = ht

        projected = torch.tanh(
            self.l_last(ys_pad.contiguous().view(-1, ys_pad.size(2)))
        )
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)

        # # Bidirectional concatenations
        # if self.bidir:
        #     h_f = h[0: int(x.shape[0] / 2)]
        #     h_b = flip(h[int(x.shape[0] / 2): x.shape[0]].contiguous(), 0)
        #     h = torch.cat([h_f, h_b], 2)

        # Setup x for the next hidden layer
        return xs_pad


def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes

    Useful in processing of sliding windows over the inputs
    """
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.0
    else:
        states[1::2] = 0.0
    return states

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
        self, etype, idim, elayers, eunits, eprojs, subsample, dropout, in_channel=1
    ):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ["lstm", "gru", "blstm", "bgru"]:
            logging.error("Error: need to specify an appropriate encoder architecture")

        self.l_first = torch.nn.Linear(idim, 550)
        self.enc = RNN(550, elayers, eunits, eprojs, dropout, typ=typ)
        self.enc_copy = copy.deepcopy(self.enc)
        for params in self.enc_copy.parameters():
            params.requires_grad_(False)

        self.amolang = Amolang(self.enc, self.enc_copy)

        self.elayers = elayers

    def get_history(self, Z):
        """
        Get the history repackaging part
        :param Z: Hidden unit sequence of dimension (bsz x seq_len x hdim)
        """
        return Z[:, -1:, :]

    def copy(self, enc):
        self.enc_copy = copy.deepcopy(enc)
        for params in self.enc_copy.parameters():
            params.detach()

    def forward(self, xs_pad, ilens, prev_states=None):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """

        self.copy(self.enc)

        current_states = []
        xs_pad = self.l_first(xs_pad)
        xs_pad = self.enc(xs_pad, ilens, prev_state=None)
        self.pretrain_steps = 5000
        if os.path.isfile('./train_step.txt'):
            with open('./train_step.txt', 'r') as f:
                train_step = int(f.readlines()[0])
        else:
            train_step = 0

        # DEQ calculation
        if 0 <= train_step < self.pretrain_steps:
            for i in range(self.elayers):
                xs_pad = self.enc(xs_pad, ilens, prev_state=None)
        else:
            xs_pad = self.amolang(xs_pad, ilens, train_step)

        train_step += 1
        with open('./train_step.txt', 'w') as f:
            f.write(str(train_step))

        z0 = self.get_history(xs_pad)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), z0

class Amolang(DEQModule):
    def __init__(self, enc, enc_copy):
        super(Amolang, self).__init__(enc, enc_copy)
        self.enc = enc
        self.enc_copy = enc_copy
        self.training = True

    def forward(self, xs_pad, ilens, train_step):
        threshold = 50
        xs_pad = RootFind.apply(self.enc, xs_pad, ilens, None, threshold, train_step)
        if self.training:
            xs_pad = RootFind.f(self.enc, xs_pad, ilens, None)
            xs_pad = self.Backward.apply(self.enc_copy, xs_pad, ilens, None, threshold, train_step)

        return xs_pad

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
            )
            enc_list.append(enc)
        return enc_list
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
