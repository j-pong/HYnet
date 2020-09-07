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

class BrewCnnLayer(nn.Module):
    def __init__(
        self,
        sample_size,
        hidden_size,
        target_size,
        bias=True):
        super().__init__()

        self.fnn = nn.ModuleList([
            nn.Conv2d(1, 5, kernel_size=(7,7), stride=(1,1), bias=False),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=(5,5), stride=(1,1), bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(18 * 18 * 10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        ])

    @staticmethod
    def calculate_ratio(x, x_base):
        rat = x / x_base
        rat[torch.isnan(rat) | torch.isinf(rat)] = 0.0

        return rat

    def brew(self, ratio, split_dim=None, w_hat=None, b_hat=None):
        i = 0

        i_max = len(ratio)
        
        for module in self.fnn:
            if isinstance(module, nn.Conv2d):
                w = module.weight
                if module.transposed:
                    w = w.transpose(0,1)
                # (out_ch, in_ch, *kernel_size)
                # b = module.bias

                rat = ratio[i].unsqueeze(1)  # (B, 1, out_ch, w, h)
                if i == 0:
                    batch_size = rat.size(0)

                if w_hat is None:
                    w = w.reshape(1, w.size(0), -1, 1, 1).transpose(1, 2)  # (1, in_ch * *kernel_size, out_ch)
                    w_hat = w * rat  # (B, in_ch * *kernel_size, out_ch, w, h)
                    w_hat = w_hat.reshape(w_hat.size(0) * w_hat.size(1), w_hat.size(2), w_hat.size(3), w_hat.size(4))
                else:
                    w_hat_unf = F.unfold(w_hat, kernel_size=module.kernel_size)
                    # (B * prev_in_ch * *prev_kernel_size, *kernel_size * in_ch, w_new * h_new)
                    w = w.reshape(w.size(0), -1)
                    w_hat = torch.matmul(w_hat_unf.transpose(-2, -1), w.t())  
                    # (B * prev_in_ch * *prev_kernel_size, w_new * h_new, out_ch)
                i += 1

            elif isinstance(module, nn.Linear):
                w = (module.weight).transpose(-2,- 1)
                b = module.bias
                if i_max > i:
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
                else:
                    w_hat = torch.matmul(w_hat, w)
                    # (?, d, C) x (C, C*)  -> (?, d, C*)

                    if b is not None:
                        b_hat = torch.matmul(b_hat, w)
                        # (?, C) x (C, C*) -> (?, C*)
                        b_hat = (b.unsqueeze(0) + b_hat)
                        # (?, C) + (?, C) -> (?, C*)
            elif isinstance(module, nn.ReLU):
                pass
            elif isinstance(module, nn.PReLU):
                pass
            elif isinstance(module, nn.Flatten):
                w_hat = w_hat.reshape(batch_size, int(w_hat.size(0) / batch_size), -1)
                
            else:
                raise AttributeError(
                    "Current network architecture, {}, is not supported!".format(module))

        return w_hat, b_hat

    def forward(self, x):
        ratio = []

        batch_size = x.size(0)

        x = x.view(batch_size, 1, 28, 28)

        # make patch for weight        
        for idx, module in enumerate(self.fnn):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                x = module(x)
            elif isinstance(module, nn.ReLU) or isinstance(module, nn.PReLU):
                x_base = x
                x = module(x)
                ratio.append(self.calculate_ratio(x, x_base))
            elif isinstance(module, nn.Flatten):
                x = module(x)
            else:
                raise AttributeError(
                    "Current network architecture is not supported!")

        return x, ratio