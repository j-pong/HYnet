import torch
from torch import nn
import torch.nn.functional as F

from fairseq.models.wav2vec import ConvAggegator

from moneynet.nets.pytorch_backend.unsup.utils import pad_for_shift, select_with_ind, one_hot


class Inference(nn.Module):
    def __init__(self, idim, odim, args):
        super().__init__()
        # configuration
        assert idim == odim
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

        self.encoder = nn.Sequential(
            nn.Linear(idim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.hdim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hdim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.odim)
        )

    @staticmethod
    def energy_pooling_mask(x, part_size, share=False):
        energy = x.pow(2)
        if share:
            indices = torch.topk(energy, k=part_size * 2, dim=-1)[1]  # (B, T, cdim*2)
        else:
            indices = torch.topk(energy, k=part_size, dim=-1)[1]  # (B, T, cdim)
        mask = one_hot(indices[:, :, :part_size], num_classes=x.size(-1)).float().sum(-2)  # (B, T, hdim)
        mask_share = one_hot(indices, num_classes=x.size(-1)).float().sum(-2)  # (B, T, hdim)
        return mask, mask_share

    def hidden_exclude_activation(self, h, mask_prev):
        # byte tensor is not good choice
        if mask_prev is None:
            mask_cur, mask_cur_share = self.energy_pooling_mask(h, self.cdim, share=True)
            mask_prev = mask_cur
        else:
            assert mask_prev is not None
            h[mask_prev.byte()] = 0.0
            mask_cur, mask_cur_share = self.energy_pooling_mask(h, self.cdim, share=True)
            mask_prev = mask_prev + mask_cur
        h = h.masked_fill(~(mask_cur_share.byte()), 0.0)
        return h, mask_prev

    def forward(self, x, mask_prev=None):
        x = self.encoder(x)
        if mask_prev is not None:
            x, mask_prev = self.hidden_exclude_activation(x, mask_prev)
        else:
            mask_prev = None
        x = self.decoder(x)

        return x


class ConvInference(nn.Module):
    def __init__(self, idim, odim, args):
        super().__init__()
        # configuration
        self.idim = idim
        self.odim = odim

        agg_layers = [(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1),
                      (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]
        self.encoder = ConvAggegator(
            conv_layers=agg_layers,
            embed=idim,
            dropout=0.0,
            skip_connections=True,
            residual_scale=0.5,
            non_affine_group_norm=False,
            conv_bias=True,
            zero_pad=False,
            activation=nn.GELU(),
        )
        self.decoder = nn.Linear(512, odim * args.tnum)

    def forward(self, x):
        x = x.transpose(-2, -1)  # B, C, T
        x = self.encoder(x)
        x = x.transpose(-2, -1)  # B, T, C
        x = self.decoder(x)
        return x


class ExcInference(Inference):
    def __init__(self, idim, odim, args):
        super().__init__()
        assert idim == odim
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

        # next frame predictor
        self.input_extra = 0
        self.output_extra = 0
        # ToDo(j-pong): fix for task
        self.encoder = nn.Linear(idim + self.input_extra, self.hdim)
        self.decoder = nn.Linear(self.hdim, odim * self.output_extra)

    @staticmethod
    def energy_pooling(x, dim=-1):
        energy = x.pow(2).sum(dim)
        x_ind = torch.max(energy, dim=-1)[1]  # (B, Tmax, *)
        x = select_with_ind(x, x_ind)  # (B, Tmax, hdim)
        return x, x_ind

    def forward(self, x, mask_prev=None):
        x, _ = pad_for_shift(key=x, pad=self.input_extra,
                             window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
        h = self.encoder(x)  # (B, Tmax, *, hdim)
        # max pooling along shift size
        h, h_ind = self.energy_pooling(h)
        if mask_prev is not None:
            # max pooling along hidden size
            h, mask_prev = self.hidden_exclude_activation(h, mask_prev)
        # feedforward decoder
        x_ext = self.decoder(h)
        # output trunk along feature side with window
        x_ext = [select_with_ind(x_ext, x_ext.size(-1) - 1 - h_ind - i) for i in torch.arange(self.idim).flip(0)]
        x = torch.stack(x_ext, dim=-1)

        return x
