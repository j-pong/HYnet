import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

class BrewLayer(nn.Module):
    def __init__(
        self, 
        sample_size,
        hidden_size, 
        target_size, 
        bias=True):
        super().__init__()

        self.fnn = nn.ModuleList([
            nn.Linear(sample_size, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, target_size, bias=bias)
        ])

    @staticmethod
    def calculate_ratio(x, x_base):
        rat = x / x_base
        rat[torch.isnan(rat) | torch.isinf(rat)] = 0.0

        return rat

    def brew(self, ratio, split_dim=None, w_hat=None, b_hat=None):
        i = 0
        
        for module in self.fnn:
            if isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                rat = ratio[i].view(-1, ratio[i].size(-1))  # (?, C)

                if w_hat is None:
                    w_hat = rat.unsqueeze(1) * w.unsqueeze(0)
                    # (?, 1, C1) * (1, d, C1)  -> (?, d, C1)
                else:
                    w_hat = rat.unsqueeze(1) * w_hat  # (?, 1, C*) * (?, d, C*)

                if b is not None:
                    if b_hat is None:
                        b_hat = rat * b.unsqueeze(0)
                        # (?, C1) * (1, C1) -> (?, C1)
                    else:
                        b_hat = rat * b_hat
                        # (?, C) + (?, C) -> (?, C*)
                i += 1
            elif isinstance(module, nn.Linear):
                w = (module.weight).transpose(-2,- 1)
                b = module.bias

                if w_hat is not None:
                    w_hat = torch.matmul(w_hat, w)
                        # (?, d, C) x (C, C*)  -> (?, d, C*)
                if b is not None:
                    if b_hat is not None:
                        b_hat = torch.matmul(b_hat, w)
                            # (?, C) x (C, C*) -> (?, C*)
                        b_hat = b.unsqueeze(0) + b_hat

            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

        return w_hat, b_hat

    def forward(self, x):
        ratio = []
        for idx, module in enumerate(self.fnn):
            if isinstance(module, nn.Linear):
                x = module(x)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                x_base = x
                x = module(x)
                ratio.append(self.calculate_ratio(x, x_base))
            else:
                raise AttributeError(
                    "Current network architecture is not supported!")

        return x, ratio

class Att(nn.Module):
    def __init__(
        self):
        super().__init__()

        self.min_value = float(
            np.finfo(torch.tensor(0, dtype=torch.float).numpy().dtype).min
        )
        self.mask_diag = False

        self.softmax = nn.Softmax(dim=-1)

        self.kernel = None

    def forward(self, x):
        _, tsz, csz = x.size()
        q, k, v = torch.split(x, dim=-1, split_size_or_sections=int(csz / 3))

        score = torch.matmul(k, q.transpose(-2, -1)) / np.sqrt(q.size(-1))
        if self.mask_diag:
            mask = torch.zeros_like(score)
            mask[:, range(tsz), range(tsz)] = 1
            mask = mask.bool()
            score.masked_fill_(mask, self.min_value)

        kernel = self.softmax(score)
        # if self.mask_diag:
        #     kernel.masked_fill_(mask, 0.0)
        #     assert torch.sum((kernel.diagonal(dim1=-2, dim2=-1) != 0.0).float()) == 0.0
        self.kernel = kernel

        x = torch.matmul(kernel, v)

        return x

class BrewAttLayer(nn.Module):
    def __init__(
        self,
        hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.atts = [Att(), Att()]

        self.cnn = nn.ModuleList([
            nn.Unfold(kernel_size=(7,7), stride=(1,1)),
            nn.Linear(in_features=7*7*1, out_features=self.hidden_size * 3, bias=False),
            self.att[0],
            nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size * 3),
            self.att[1],
            nn.Fold(kernel_size=(1,1), output_size=(22,22)),
        ])

    def calculate_ratio(self, x, x_base): 
        rat = x / x_base
        rat[torch.isnan(rat) | torch.isinf(rat)] = 0.0

        return rat
    
    def mv_nrom(self, x):
        mean = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean(torch.pow(x - mean, exponent=2), dim=2, keepdim=True)
        x = (x - mean) / var
        return x

    def brew(self, ratio, w_hat=None, b_hat=None):
        i = 0
        
        for module in self.cnn:
            if isinstance(module, nn.Unfold):
                if w_hat is not None:
                    pass
            elif isinstance(module, nn.Linear):
                w = module.weight
                # (C*, C)
                assert module.bias is None
                if w_hat is not None:
                    w_hat = torch.matmul(w_hat.transpose(-2,- 1), w)
                    # (B, w* * h*, d, C) x (C, C*)  -> (B, w * h, d, C*)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                rat = ratio[i]
                # (B, C*, w * h)
                if w_hat is None:
                    w_hat = rat.unsqueeze(2) * w.unsqueeze(0)
                    # (B, C*, w * h) * (1, C1, d)  -> (?, d, C1)
                else:
                    w_hat = rat.unsqueeze(1) * w_hat  # (?, 1, C*) * (?, d, C*)

                i += 1
            elif isinstance(module, Att):
                rat = ratio[i]
                # (B, w * h, w * h)
                if w_hat is None:
                    w_hat = rat.unsqueeze(2) * w.unsqueeze(0)
                    # (?, C1, 1, w * h) * (1, C1, d)  -> (?, d, C1)
                else:
                    w_hat = rat.unsqueeze(1) * w_hat  # (?, 1, C*) * (?, d, C*)

                i += 1            
            elif isinstance(module, nn.Fold):
                pass
            
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

    def forward(self, x):
        ratio = []

        batch_size = x.size(0)

        x = x.view(batch_size, 1, 28, 28)

        # make patch for weight        
        for module in self.cnn:
            if isinstance(module, nn.Unfold):
                x = module(x)
            elif isinstance(module, nn.Linear):
                x = module(x.transpose(1, 2))
                # (B, T, C)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                x_base = x
                x = module(x)
                ratio.append(self.calculate_ratio(x, x_base))
            elif isinstance(module, Att):
                mean = torch.mean(x, dim=2, keepdim=True)
                var = torch.mean(torch.pow(x - mean, exponent=2), dim=2, keepdim=True)
                # (B, T, C)

                x = module(x)
                x = self.mv_nrom(x)
                x = x * var + mean 

                x = x.transpose(1, 2)
                # (B, C, T)
                ratio.append(module.kernel)
            elif isinstance(module, nn.Fold):
                x = module(x)
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

        return x, ratio