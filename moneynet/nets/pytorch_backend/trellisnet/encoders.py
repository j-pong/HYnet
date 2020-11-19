import logging
import six
from distutils.util import strtobool

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device

from moneynet.nets.pytorch_backend.trellisnet.nets_utils import weight_norm, VariationalHidDropout, WeightDrop


class WeightShareConv1d(nn.Module):
    def __init__(self, idim, hdim, pdim, kernel_size, dropout=0.0):
        """
        The weight-tied 1D convolution used in TrellisNet.
        :param idim: The dim of original input
        :param hdim: The dim of hidden input
        :param pdim: The dim of the pre-activation (i.e. convolutional) output
        :param kernel_size: The size of the convolutional kernel
        :param dropout: Hidden-to-hidden dropout
        """
        super(WeightShareConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.idim = idim

        conv1 = nn.Conv1d(idim, pdim, kernel_size)
        self.weight1 = conv1.weight

        conv2 = nn.Conv1d(hdim, pdim, kernel_size)
        self.weight2 = conv2.weight
        self.bias2 = conv2.bias
        # self.init_weights() # TODO: change this if not work

        self.dict = dict()
        self.drop = VariationalHidDropout(dropout=dropout)

    def init_weights(self):
        bound = 0.01
        self.weight1.data.normal_(0, bound)
        self.weight2.data.normal_(0, bound)
        self.bias2.data.normal_(0, bound)

    def forward(self, z_pad, dilation, hid=None):
        k = self.kernel_size
        padding = (k - 1) * dilation    # To maintain causality constraint / TODO: ???

        # Input part
        x_1 = z_pad[:, :self.idim]

        # Hidden part
        z_1 = z_pad[:, self.idim:]
        device = x_1.get_device()

        # TODO: change below here
        # A linear transformation of the input sequence (and pre-computed once)
        if (dilation, device) not in self.dict or self.dict[(dilation, device)] is None:
            self.dict[(dilation, device)] = F.conv1d(x_1, self.weight1, dilation=dilation)
        a = F.conv1d(self.drop(z_1), self.weight2, self.bias2, dilation=dilation)
        # Input injection
        return self.dict[(dilation, device)] + F.conv1d(self.drop(z_1), self.weight2, self.bias2, dilation=dilation)


class TrellisNet(nn.Module):
    def __init__(self, idim, elayers, cdim, hdim,
                 dropout, lnorm, bnorm, wnorm=False, kernel_size=2, aux_frequency=20, dilation=[1], typ="trellis"):
        """
        Build a trellis network with LSTM-style gated activations
        :param idim: The input (e.g., embedding) dimension
        :param cdim: The hidden unit dimension (excluding the output dimension). In other words, if you want to build
                     a TrellisNet with hidden size 1000 and output size 400, you should set cdim = 1000-400 = 600.
                     (The reason we want to separate this is from Theorem 1.)
        :param hdim: The output dimension
        :param elayers: Number of layers
        :param kernel_size: Kernel size of the TrellisNet
        :param dropout: Hidden-to-hidden (VD-based) dropout rate
        :param wnorm: A boolean indicating whether to use weight normalization
        :param aux_frequency: Frequency of taking the auxiliary loss; (-1 means no auxiliary loss)
        :param dilation: The dilation of the convolution operation in TrellisNet
        """
        super(TrellisNet, self).__init__()

        self.idim = idim
        self.cdim = cdim
        self.hdim = hdim
        self.hsize = hsize = cdim + hdim
        self.dilation = dilation
        self.elayers = elayers
        self.fn = None
        self.last_output = None
        self.aux_frequency = aux_frequency

        self.kernel_size = kernel_size

        self.ln = torch.nn.Linear(idim + hsize, hsize * 4)
        if wnorm:
            print("Weight normalization applied")
            self.full_conv, self.fn = weight_norm(
                WeightShareConv1d(idim, hsize, 4 * hsize, kernel_size=kernel_size, dropout=dropout),
                names=['weight1', 'weight2'],
                dim=0)           # The weights to be normalized
        else:
            self.full_conv = WeightShareConv1d(idim, hsize, 4 * hsize, kernel_size=kernel_size, dropout=dropout)

    def transform_input(self, X):
        # X has dimension (N, ,idim, L)
        device = X.device
        batch_size = X.size(0)
        seq_len = X.size(1)
        hsize = self.hsize

        self.ht = torch.zeros(batch_size, seq_len, hsize).to(device)
        return torch.cat([X, self.ht], dim=2)     # "Injecting" input sequence at layer 1

    def step(self, Z, dilation=1, hc=None):
        idim = self.idim
        hsize = self.hsize
        (hid, cell) = hc
        # hc: (N, h_size, 1, N, h_size, 1)

        # Apply convolution
        # out: self.dict[(dilation, device)] +
        # F.conv1d(self.drop(z_1), self.weight2, self.bias2, dilation=dilation)
        # out = self.full_conv(Z, dilation=dilation, hid=hid)
        out = self.ln(Z)

        # Gated activations among channel groups
        ct_1 = F.pad(self.ct, (dilation, 0))[:, :, :-dilation]  # Dimension (N, hsize, L)
        ct_1[:, :, :dilation] = cell.repeat(1, 1, dilation)

        it = torch.sigmoid(out[:, :hsize])
        ot = torch.sigmoid(out[:, hsize: 2 * hsize])
        gt = torch.tanh(out[:, 2 * hsize: 3 * hsize])
        ft = torch.sigmoid(out[:, 3 * hsize: 4 * hsize])
        ct = ft * ct_1 + it * gt
        ht = ot * torch.tanh(ct)

        # Put everything back to form Z (i.e., injecting input to hidden unit)
        Z = torch.cat([Z[:, :, :idim], ht], dim=1)
        self.ct = ct
        return Z

    def forward(self, xs_pad, ilens, state, aux):
        idim = self.idim
        hdim = self.hdim
        xs_pad = xs_pad.view(-1, xs_pad.size(2), xs_pad.size(1))
        Z = self.transform_input(xs_pad)
        aux_outs = []
        dilation_cycle = self.dilation

        # xs_pad: N, L, idim
        # Z: N, L, idim + h_size
        # state: (N, h_size, 1, N, h_size, 1)

        if self.fn is not None:
            # Recompute weight normalization weights
            self.fn.reset(self.full_conv)
        for key in self.full_conv.dict:
            # Clear the pre-computed computations
            if key[1] == xs_pad.get_device():
                self.full_conv.dict[key] = None
        self.full_conv.drop.reset_mask(Z[:, idim:])

        # Feed-forward layers
        for i in range(self.elayers):
            d = dilation_cycle[i % len(dilation_cycle)]
            Z = self.step(Z, dilation=d, hc=state)
            if aux and i % self.aux_frequency == (self.aux_frequency-1):
                aux_outs.append(Z[:, :, -hdim:].unsqueeze(0))

        out = Z[:, :, -hdim:]              # Dimension (N, L, hdim)
        state = (Z[:, :, idim:], self.ct[:, -1:, :])     # Dimension (N, hsize, L)
        if aux:
            aux_outs = torch.cat(aux_outs, dim=0).transpose(0, 1).transpose(2, 3) ### ???
        else:
            aux_outs = None
        return out, state, aux_outs

    def init_hidden(self, batch_size):
        """Sets backward BRNN states to zeroes

        Useful in processing of sliding windows over the inputs
        """
        hsize = self.cdim + self.hdim
        state = (torch.zeros(batch_size, hsize, 1), torch.zeros(batch_size, hsize, 1))
        return state

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
    # TODO: wnorm
    def __init__(
        self, etype, idim, elayers, eunits, eprojs, dropout, lnorm, bnorm, wnorm=False,
            temporalwdrop=True, aux=False, aux_frequency=1e4):

        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        if typ not in ["trellis"]:
            logging.error("Error: need to specify an appropriate encoder architecture")

        if etype.startswith("trellis"):
            self.enc = torch.nn.ModuleList(
                [TrellisNet(idim, elayers, eunits, eprojs,
                 dropout, lnorm, bnorm, wnorm=wnorm, kernel_size=3, aux_frequency=aux_frequency,
                            dilation=[1], typ="trellis")]
            )
            logging.info(typ.upper() + " without projection for encoder")

        reg_term = '_v' if wnorm else ''
        for module in self.enc:
            self.network = WeightDrop(module,
                                      [['full_conv', 'weight1' + reg_term],
                                       ['full_conv', 'weight2' + reg_term]],
                                      dropout=dropout, temporal=temporalwdrop)

        self.aux = aux

    def forward(self, xs_pad, ilens):
        """Encoder forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor state: batch of previous encoder hidden states (?, ...)
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """

        for module in self.enc:
            state = module.init_hidden(xs_pad.size(0))
            xs_pad, ilens, state = module(xs_pad, ilens, state=state, aux=self.aux)

        # make mask to remove bias value in padded part
        mask = to_device(self, make_pad_mask(ilens).unsqueeze(-1))

        return xs_pad.masked_fill(mask, 0.0), ilens, state


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
