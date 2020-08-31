import torch
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
            if isinstance(module, nn.Linear):
                w = (module.weight).transpose(-2,- 1)
                b = module.bias

                rat = ratio[i].view(-1, ratio[i].size(-1))  # (?, C)

                if w_hat is None:
                    w_hat = rat.unsqueeze(1) * w.unsqueeze(0)
                    # (?, 1, C1) * (1, d, C1)  -> (?, d, C1)
                else:
                    w_hat = torch.matmul(w_hat, w)
                    # (?, d, C) x (C, C*)  -> (?, d, C*)
                    w_hat = rat.unsqueeze(1) * w_hat  # (?, 1, C*) * (?, d, C*)

                if b is not None:
                    if b_hat is None:
                        b_hat = rat * b.unsqueeze(0)
                        # (?, C1) * (1, C1) -> (?, C1)
                    else:
                        b_hat = torch.matmul(b_hat, w)
                        # (?, C) x (C, C*) -> (?, C*)
                        b_hat = rat * (b.unsqueeze(0) + b_hat)
                        # (?, C) + (?, C) -> (?, C*)

                i += 1
            elif isinstance(module, nn.ReLU):
                pass
            elif isinstance(module, nn.PReLU):
                pass
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

        return w_hat, b_hat

    def forward(self, x):
        ratio = []
        for idx, module in enumerate(self.fnn):
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
                raise AttributeError(
                    "Current network architecture is not supported!")

            if len(self.fnn) - 1 == idx:
                ratio.append(self.calculate_ratio(x, x_base))

        return x, ratio
