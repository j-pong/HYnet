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
            if isinstance(module, nn.Linear):
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
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
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

class BrewCnnLayer(nn.Module):
    def __init__(
        self):
        super().__init__()

        self.cnn = nn.ModuleList([
            nn.Unfold(kernel_size=(7,7), stride=(1,1)),
            nn.Linear(in_features=7*7*1, out_features=10, bias=False),
            nn.Fold(kernel_size=(1,1), output_size=(22,22)),
            nn.ReLU(),
            nn.Unfold(kernel_size=(5,5), stride=(1,1)),
            nn.Linear(in_features=5*5*10, out_features=10, bias=False),
            nn.Fold(kernel_size=(1,1), output_size=(18,18)),
            nn.ReLU()
        ])

    @staticmethod
    def calculate_ratio(x, x_base): 
        rat = x / x_base
        rat[torch.isnan(rat) | torch.isinf(rat)] = 0.0

        return rat

    def brew(self, ratio, split_dim=None, w_hat=None, b_hat=None):
        i = 0
        
        for module in self.cnn:
            if isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                rat = ratio[i]
                # (B, C*, w, h)
                if w_hat is None:
                    w_hat = rat.unsqueeze(1) * w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    # (?, 1, C1, w, h) * (1, d, C1, 1, 1)  -> (?, d, C1, w, h)
                else:
                    w_hat = rat.unsqueeze(1) * w_hat  
                    # (?, 1, C*, w, h) * (?, d, C*, w, h) -> (?, d, C*, w, h)

                i += 1
            elif isinstance(module, nn.Unfold):
                pass
            elif isinstance(module, nn.Fold):
                pass
            elif isinstance(module, nn.Linear):
                w = (module.weight).transpose(-2,- 1)
                # (C, C*)
                assert module.bias is None

                if w_hat is not None:
                    w_hat = torch.matmul(w_hat, w)
                    # (?, d, C, w, h) x (C, C*)  -> (?, d, C*)
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

        return w_hat, b_hat

    def forward(self, x):
        ratio = []

        batch_size = x.size(0)

        x = x.view(batch_size, 1, 28, 28)

        # make patch for weight        
        for module in self.cnn:
            if isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                x_base = x
                x = module(x)
                ratio.append(self.calculate_ratio(x, x_base))
                print(ratio[-1].size())
            elif isinstance(module, nn.Unfold):
                x = module(x)
            elif isinstance(module, nn.Fold):
                x = module(x)
            elif isinstance(module, nn.Linear):
                x = module(x.transpose(1, 2))
                x = x.transpose(1, 2)
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

            
        exit()

        return x, ratio