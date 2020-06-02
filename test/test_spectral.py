import torch
import torch.nn.functional as F
import numpy as np

resol = 256
idim = 6
odim = 4
denorm = (resol + idim * 2 - 1) / 2  # (resol - (idim -2)) / 2  # why this value work for normalizing ? 63.5

t = torch.linspace(0, 1, resol)

# prepare tracer
basis_set = [torch.cos(2 * np.pi * t * bs) for bs in torch.arange(1, idim + 1)]  # why cos doesn't work properly
basis_set = torch.stack(basis_set, dim=-1).unsqueeze(0)  # 1, T, d

# random feature generation
x = torch.rand([1, 1, idim])  # B, 1, d
y = torch.rand([1, 1, odim])  # B, 1, d'

# attach the tracer
x_ = x * basis_set  # B, T, d
x_rev = torch.matmul(x_.transpose(-2, -1), basis_set).sum(-1) / denorm

# normal distribution initialization dnn arch.
w1 = torch.normal(mean=0, std=1, size=(idim, 64))
h = torch.matmul(x_, w1)  # B, T, C1
h = F.relu(h)
w2 = torch.normal(mean=0, std=1, size=(64, 64))
h = torch.matmul(h, w2)  # B, T, d'
h = F.relu(h)
w3 = torch.normal(mean=0, std=1, size=(64, odim))
y_hat = torch.matmul(h, w3)  # B, T, d'

# tracing feature coefficient
lam = torch.matmul(y_hat.transpose(-2, -1), basis_set) / denorm  # B, d', d
w_hat = lam / x
att_p = torch.abs(w_hat)/torch.sum(torch.abs(w_hat))
print(att_p)
# y_hat_hat = torch.matmul(x_, w_hat.transpose(-2, -1))
# lam_hat = torch.matmul(y_hat_hat.transpose(-2, -1), basis_set) / denorm  # B, d', d
