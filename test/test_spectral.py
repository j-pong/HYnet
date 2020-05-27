import torch
import numpy as np

resol = 128
denorm = 63.5
f_dim = 3

x = torch.linspace(0, 1, resol)
units_freq = torch.arange(1, 3)

# prepare tracer
basis_state = torch.arange(1, 4)
basis_set = [torch.sin(2 * np.pi * x * bs) for bs in basis_state]
basis_set = torch.stack(basis_set, dim=-1)  # T, f_dim

# random feature generation
x = torch.rand([1, f_dim])

# attach the tracer
new_x = x * basis_set  # T, f_dim

# normal distribution initialization dnn arch.
w1 = torch.normal(0, 1, [f_dim, 64])
new_x = torch.matmul(new_x, w1)  # T, 64
w2 = torch.normal(0, 1, [64, 1])
y = torch.matmul(new_x, w2)  # T, 1

# tracing feature coefficient
coef_hat = torch.matmul(basis_set.t(), y) / denorm

# print
print(coef_hat)
