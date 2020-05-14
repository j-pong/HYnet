import torch
from torch import nn
import torch.nn.functional as F

from moneynet.nets.pytorch_backend.unsup.utils import pad_for_shift, select_with_ind


class InferenceNet(nn.Module):
    def __init__(self, idim, odim, args):
        super(InferenceNet, self).__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

        # next frame predictor
        self.encoder_type = args.etype
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
    def one_hot(y, num_classes):
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(y_tensor.device)

        return zeros.scatter(scatter_dim, y_tensor, 1)

    @staticmethod
    def energy_pooling(x, dim=-1):
        energy = x.pow(2).sum(dim)
        x_ind = torch.max(energy, dim=-1)[1]  # (B, Tmax, *)
        x = select_with_ind(x, x_ind)  # (B, Tmax, hdim)
        return x, x_ind

    def energy_pooling_mask(self, x, part_size, share=False):
        energy = x.pow(2)
        if share:
            indices = torch.topk(energy, k=part_size * 2, dim=-1)[1]  # (B, T, cdim*2)
        else:
            indices = torch.topk(energy, k=part_size, dim=-1)[1]  # (B, T, cdim)
        mask = self.one_hot(indices[:, :, :part_size], num_classes=x.size(-1)).float().sum(-2)  # (B, T, hdim)
        mask_share = self.one_hot(indices, num_classes=x.size(-1)).float().sum(-2)  # (B, T, hdim)
        return mask, mask_share

    def hidden_exclude_activation(self, h, mask_prev):
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
