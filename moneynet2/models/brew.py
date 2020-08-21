import torch
from torch import nn

from moneynet.nets.pytorch_backend.unsup.utils import pad_for_shift, select_with_ind, one_hot

from fairseq.models.wav2vec import Wav2VecModel, Wav2VecPredictionsModel


class AbcModel(nn.Module):
    def __init__(self, idim, odim, args):
        super().__init__()
        # configuration
        self.idim = idim
        self.odim = odim
        self.hdim = args.hdim
        self.tnum = args.tnum

        self.bias = args.bias

        self.linear = nn.ModuleList([
            nn.Linear(idim, self.hdim, bias=self.bias),
            nn.ReLU(),
            nn.Linear(self.hdim, self.hdim, bias=self.bias)
        ])

    @staticmethod
    def calculate_ratio(x, x_base):
        rat = x / x_base
        rat[torch.isnan(rat) | torch.isinf(rat)] = 0.0

        return rat

    @staticmethod
    def brew_(module_lists, ratio, split_dim=None, w_hat=None, bias_hat=None):
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