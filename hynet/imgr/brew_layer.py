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

    @torch.no_grad()
    def brew(self, ratio, split_dim=None, w_hat=None, bias_hat=None):
        i = 0
        
        for module in self.fnn:
            if isinstance(module, nn.Linear):
                w = module.weight
                b = module.bias
                assert b is None

                if w_hat is None:
                    w_hat = ratio[i].view(-1, ratio[i].size(-1)).unsqueeze(1) * \
                        w.transpose(-2, -1).unsqueeze(0)
                    # (?, 1, C1) * (1, d, C1)  -> (?, d, C1)
                else:
                    # (?, d, C) x (C, C*)  -> (?, d, C*)
                    w_hat = torch.matmul(w_hat, w.transpose(-2, -1))
                    w_hat = ratio[i].view(-1, ratio[i].size(-1)).unsqueeze(
                        1) * w_hat  # (?, 1, C*) * (?, d, C*)

                i += 1
            elif isinstance(module, nn.ReLU):
                pass
            elif isinstance(module, nn.PReLU):
                pass
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

        return w_hat, bias_hat

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
