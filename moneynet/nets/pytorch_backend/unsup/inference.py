import torch
from torch import nn

from moneynet.nets.pytorch_backend.unsup.utils import pad_for_shift, select_with_ind, one_hot

from fairseq.models.wav2vec import Wav2VecModel, Wav2VecPredictionsModel


class Inference(nn.Module):
    def __init__(self, idim, odim, args):
        super().__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.tnum = args.tnum

        self.bias = args.bias

        self.embed = nn.ModuleList([
            nn.Linear(idim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias)
        ])
        self.transform = nn.ModuleList([
            nn.Linear(idim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, odim, bias=self.bias)
        ])

    @staticmethod
    def calculate_ratio(x, x_base):
        rat = x / x_base
        rat[torch.isnan(rat) | torch.isinf(rat)] = 0.0

        return rat

    @staticmethod
    def amp(ratio, w=None, bias=None):
        if w is None and bias is None:
            raise AttributeError("Both of parameter is None")
        if w is not None:
            w = ratio.view(-1, ratio.size(-1)).unsqueeze(1) * w  # (B_new, 1, C*) * (B_new, d, C*)
        if bias is not None:
            bias = ratio.view(-1, ratio.size(-1)) * bias  # (B_new, C*) * (B_new, C*)
        return w, bias

    @staticmethod
    def brew_(module_lists, ratio, split_dim=None, w_hat=None, bias_hat=None):
        with torch.no_grad():
            i = 0
            for module_list in module_lists:
                for module in module_list:
                    if isinstance(module, nn.Linear):
                        if split_dim is None:
                            w = module.weight
                        elif split_dim > 0:
                            w = module.weight[:split_dim * ratio[i].size(-1)]
                        elif split_dim < 0:
                            w = module.weight[-split_dim * ratio[i].size(-1):]
                        else:
                            raise AttributeError

                        if w_hat is None:
                            w_hat = ratio[i].view(-1, ratio[i].size(-1)).unsqueeze(1) * \
                                    w.transpose(-2, -1).unsqueeze(0)
                            # (B * iter * tnum * Tmax, 1, C1) * (1, d, C1)  -> (B_new, d, C1)
                        else:
                            w_hat = torch.matmul(w_hat, w.transpose(-2, -1))  # (B_new, d, C) x (C, C*)  -> (B, d, C*)
                            w_hat = ratio[i].view(-1, ratio[i].size(-1)).unsqueeze(
                                1) * w_hat  # (B_new, 1, C*) * (B_new, d, C*)

                        if module.bias is not None:
                            if split_dim is None:
                                b = module.bias
                            elif split_dim > 0:
                                b = module.bias[:split_dim * ratio[i].size(-1)]
                            elif split_dim < 0:
                                b = module.bias[-split_dim * ratio[i].size(-1):]
                            else:
                                raise AttributeError

                            if bias_hat is None:
                                bias_hat = ratio[i].view(-1, ratio[i].size(-1)) * b.unsqueeze(
                                    0)  # (B_new, C1) * (1, C1)
                            else:
                                bias_hat = torch.matmul(bias_hat, w.transpose(-2, -1))
                                bias_hat = ratio[i].view(-1, ratio[i].size(-1)) * (
                                        bias_hat + b)  # (B_new, C*) * (B_new, C*)
                        else:
                            bias_hat = None
                        i += 1
                    elif isinstance(module, nn.ReLU):
                        pass
                    elif isinstance(module, nn.PReLU):
                        pass
                    else:
                        raise AttributeError("Current network architecture, {}, is not supported!".format(module))

            return w_hat, bias_hat

    def forward(self, x, module_list):
        ratio = []
        for idx, module in enumerate(module_list):
            if isinstance(module, nn.Linear):
                if idx > 0:
                    ratio.append(self.calculate_ratio(x, x_base))
                x_base = module(x)
                x = x_base
            elif isinstance(module, nn.ReLU):
                x = module(x)
            elif isinstance(module, nn.PReLU):
                x = module(x)
            else:
                raise AttributeError("Current network architecture is not supported!")

            if len(module_list) - 1 == idx:
                ratio.append(self.calculate_ratio(x, x_base))

        return x, ratio


class HirInference(Inference):
    def __init__(self, idim, odim, args):
        super(Inference, self).__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.tnum = args.tnum

        self.bias = args.bias

        self.encoder_q = nn.ModuleList([
            nn.Linear(idim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
        ])
        self.encoder_k = nn.ModuleList([
            nn.Linear(idim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(self.hdim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, odim, bias=self.bias)
        ])


class FairInference(Inference):
    def __init__(self, idim, odim, args):
        super(Inference, self).__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.tnum = args.tnum

        self.bias = args.bias

        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=idim,
            out_dim=odim,
            prediction_steps=args.prediction_steps,
            n_negatives=args.num_negatives,
            cross_sample_negatives=args.cross_sample_negatives,
            sample_distance=args.sample_distance,
            dropout=args.dropout,
            offset=0,
            balanced_classes=args.balanced_classes,
            infonce=args.infonce,
        )

    def forward(self, x, x_agg):
        """
        Input
        x : B C T
        x_agg : B C T

        Output
        result : python.dictionary with key [cpc_logits, cpc_targets]
        """
        result = {}

        x, targets = self.wav2vec_predictions(x_agg, x)

        result["cpc_logits"] = x
        result["cpc_targets"] = targets

        return result


class ExcInference(Inference):
    def __init__(self, idim, odim, args):
        super(Inference, self).__init__()
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
                             window=self.input_extra + self.idim)  # B, Tmax, *, idim
        h = self.encoder(x)  # B, Tmax, *, hdim
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
