import torch
import torch.nn.functional as F
import numpy as np

resol = 128
denorm = 63.5
d = 3

t = torch.linspace(0, 1, resol)

# prepare tracer
basis_set = [torch.sin(2 * np.pi * t * bs) for bs in torch.arange(1, d + 1)]
basis_set = torch.stack(basis_set, dim=-1).unsqueeze(0)  # 1, T, d

# random feature generation
x = torch.rand([1, 1, d])  # B, 1, d

# attach the tracer
new_x = x * basis_set  # B, T, d
x_rev = torch.matmul(basis_set.transpose(-2, -1), new_x) / denorm

# normal distribution initialization dnn arch.
w1 = torch.normal(mean=0, std=1, size=(d, 64))
h = torch.matmul(new_x, w1)  # B, T, C1
h = F.relu(h)
w2 = torch.normal(mean=0, std=1, size=(64, 4))
y = torch.matmul(h, w2)  # B, T, d'

# tracing feature coefficient
w_hat = torch.matmul(basis_set.transpose(-2, -1), y) / denorm  # B, d, d'
# Note: The w_hat is transformed by the transform function. Thus we should compare x with w_hat.
print(x, w_hat)

# check loss by non-linearity
new_x_hat = y.unsqueeze(2) / w_hat.unsqueeze(1)  # B, T, d, d'
# new_x_hat = torch.masked_fill(new_x_hat, torch.isinf(new_x_hat), 0)
x_used = torch.sum(new_x_hat * basis_set.unsqueeze(-1), dim=1) / denorm  # B, d, d'

