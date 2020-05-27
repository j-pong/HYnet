import math

import torch
from torch import nn
import torch.nn.functional as F

# from fairseq.models.wav2vec import norm_block

from moneynet.nets.pytorch_backend.unsup.utils import pad_for_shift, select_with_ind, one_hot


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ConvInference(nn.Module):
    def __init__(self, idim, odim,
                 conv_layers=[(512, 10, 5),
                              (512, 8, 4),
                              (512, 4, 2),
                              (512, 4, 2),
                              (512, 4, 2),
                              (512, 1, 1),
                              (512, 1, 1)],
                 dropout=0.0,
                 log_compression=False,
                 skip_connections=False,
                 residual_scale=0.5,
                 non_affine_group_norm=False,
                 activation=nn.ReLU()):
        super(ConvInference, self).__init__()
        self.idim = idim
        self.odim = odim

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = idim
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        x = torch.transpose(x, -2, -1)
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        print(x.size())
        exit()

        return x


class ExcInference(nn.Module):
    def __init__(self, idim, odim,
                 hdim,
                 cdim,
                 etype):
        super(ExcInference, self).__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.cdim = cdim

        # next frame predictor
        self.encoder_type = etype
        if self.encoder_type == 'conv1d':
            self.input_extra = idim
            self.output_extra = odim
            self.encoder = nn.Linear(idim + self.input_extra, self.hdim)
            self.decoder_src = nn.Linear(self.hdim, odim + self.output_extra)
            self.decoder_self = nn.Linear(self.hdim, idim + self.input_extra)
        elif self.encoder_type == 'linear':
            self.encoder = nn.Linear(idim, self.hdim)
            self.decoder_src = nn.Linear(self.hdim, odim)
            self.decoder_self = nn.Linear(self.hdim, idim)

    @staticmethod
    def energy_pooling(x, dim=-1):
        energy = x.pow(2).sum(dim)
        x_ind = torch.max(energy, dim=-1)[1]  # (B, Tmax, *)
        x = select_with_ind(x, x_ind)  # (B, Tmax, hdim)
        return x, x_ind

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

    def forward(self, x, mask_prev, decoder_type):
        if self.encoder_type == 'conv1d':
            x, _ = pad_for_shift(key=x, pad=self.input_extra,
                                 window=self.input_extra + self.idim)  # (B, Tmax, *, idim)
            h = self.encoder(x)  # (B, Tmax, *, hdim)
            # max pooling along shift size
            h, h_ind = self.energy_pooling(h)
            # max pooling along hidden size
            h, mask_prev = self.hidden_exclude_activation(h, mask_prev)
            # feedforward decoder
            assert self.idim == self.odim
            if decoder_type == 'self':
                x_ext = self.decoder_self(h)
            elif decoder_type == 'src':
                x_ext = self.decoder_src(h)
            # output trunk along feature side with window
            x_ext = [select_with_ind(x_ext, x_ext.size(-1) - 1 - h_ind - i) for i in torch.arange(self.idim).flip(0)]
            x = torch.stack(x_ext, dim=-1)
        elif self.encoder_type == 'linear':
            h = self.encoder(x)
            # max pooling along hidden size
            h, mask_prev = self.hidden_exclude_activation(h, mask_prev)
            # feedforward decoder
            assert self.idim == self.odim
            if decoder_type == 'self':
                x = self.decoder_self(h)
            elif decoder_type == 'src':
                x = self.decoder_src(h)

        return x, h, mask_prev
