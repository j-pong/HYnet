import torch
from torch import nn

from fairseq.models.wav2vec import ConvAggegator

from moneynet.nets.pytorch_backend.unsup.utils import pad_for_shift, select_with_ind, one_hot


class Inference(nn.Module):
    def __init__(self, idim, odim, args):
        super().__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.cdim = args.cdim

        self.bias = args.bias
        self.encoder = nn.ModuleList([
            nn.Linear(idim, self.hdim, bias=self.bias),
            nn.ReLU()
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.odim, bias=self.bias)
        ])

    @staticmethod
    def calculate_ratio(x, x_base):
        with torch.no_grad():
            rat = x / x_base
            rat[torch.isnan(rat)] = 0.0

        return rat

    def forward_(self, x, module_list):
        ratio = []
        for idx, module in enumerate(module_list):
            if isinstance(module, nn.Linear):
                if idx > 0:
                    ratio.append(self.calculate_ratio(x, x_base))
                x_base = module(x)
                x = x_base
            elif isinstance(module, nn.ReLU):
                x = module(x)
            else:
                raise AttributeError("Current network architecture is not supported!")

            if len(module_list) - 1 == idx:
                ratio.append(self.calculate_ratio(x, x_base))

        return x, ratio

    @staticmethod
    def brew_(module_list, ratio, w_hat=None, bias_hat=None):
        i = 0
        for module in module_list:
            if isinstance(module, nn.Linear):
                w = module.weight
                if w_hat is None:
                    w_hat = ratio[i].view(-1, ratio[i].size(-1)).unsqueeze(1) * w.transpose(-2, -1).unsqueeze(
                        0)  # (B * iter * tnum * Tmax, 1, C1) * (1, d, C1)  -> (B_new, d, C1)
                else:
                    w_hat = torch.matmul(w_hat, w.transpose(-2, -1))  # (B_new, d, C) x (C, C*)  -> (B, d, C*)
                    w_hat = ratio[i].view(-1, ratio[i].size(-1)).unsqueeze(1) * w_hat  # (B_new, 1, C*) * (B_new, d, C*)

                if module.bias is not None:
                    b = module.bias
                    if bias_hat is None:
                        bias_hat = ratio[i].view(-1, ratio[i].size(-1)) * b.unsqueeze(0)  # (B_new, C1) * (1, C1)
                    else:
                        bias_hat = torch.matmul(bias_hat, w.transpose(-2, -1))
                        bias_hat = ratio[i].view(-1, ratio[i].size(-1)) * (bias_hat + b)  # (B_new, C*) * (B_new, C*)
                else:
                    bias_hat = None
                i += 1
            elif isinstance(module, nn.ReLU):
                pass
            else:
                raise AttributeError("Current network architecture, {}, is not supported!".format(module))

        return w_hat, bias_hat

    def forward(self, x):
        x, ratio_enc = self.forward_(x, module_list=self.encoder)
        x, ratio_dec = self.forward_(x, module_list=self.decoder)

        return x, ratio_enc, ratio_dec

    def brew(self, ratios):
        p_hat = self.brew_(self.encoder, ratios[0])
        p_hat = self.brew_(self.decoder, ratios[1], p_hat[0], p_hat[1])

        return p_hat

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
