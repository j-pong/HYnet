import torch
import numpy as np

idim = 6
odim = 4

resol = 256
denorm = (resol + idim * 2 - 1) / 2  # (resol - (idim - 2)) / 2

t = torch.linspace(0, 1, resol)

# prepare tracer
basis_set = [torch.cos(2 * np.pi * t * bs) for bs in torch.arange(1, idim + 1)]
basis_set = torch.stack(basis_set, dim=-1).unsqueeze(0)  # 1, T, d

# prepare network
w1 = torch.normal(mean=0, std=1, size=(idim, 64))
w2 = torch.normal(mean=0, std=1, size=(64, 64))
w3 = torch.normal(mean=0, std=1, size=(64, odim))


def forward(x, ratio=None):
    # normal distribution initialization dnn arch.
    if ratio is None:
        ratio = []
        h = torch.matmul(x, w1)  # B, 1, c1
        h_ = torch.relu(h)
        ratio.append(h_ / h)
        h = torch.matmul(h_, w2)  # B, 1, c2
        h_ = torch.relu(h)
        ratio.append(h_ / h)
        out = torch.matmul(h_, w3)  # B, 1, d'
    else:
        h = torch.matmul(x, w1)  # 1, T, c1
        h = h * ratio[0]  # B, T, c1
        h = torch.matmul(h, w2)  # B, T, c2
        h = h * ratio[1]
        out = torch.matmul(h, w3)  # B, T, d'

    return out, ratio


# random feature generation
x = torch.normal(mean=0, std=1, size=(1, 1, idim))  # x = torch.rand([1, 1, idim])  # B, 1, d
y_hat, ratio = forward(x)  # B, 1, d'
complex_basis, _ = forward(basis_set, ratio)

# tracing feature coefficient
w_hat = torch.matmul(complex_basis.transpose(-2, -1), basis_set) / denorm  # B, d', d
# evaluation spectral disentangling
y_hat_hat = torch.matmul(x, w_hat.transpose(-2, -1))
# print(y_hat)
print(y_hat_hat)

w_hat = w1.unsqueeze(0) * ratio[0]
w_hat = torch.matmul(w_hat, w2)
w_hat = w_hat * ratio[1]
w_hat = torch.matmul(w_hat, w3).transpose(-2,-1)

# evaluation spectral disentangling
y_hat_hat = torch.matmul(x, w_hat.transpose(-2, -1))
print(y_hat)
print(y_hat_hat)

# w_hat_x = w_hat * torch.sign(x)
# w_hat_x_p = torch.relu(w_hat)
# w_hat_x_n = torch.relu(-w_hat)
# e_loss = torch.sum(w_hat_x_p) + torch.sum(w_hat_x_n)
# print(e_loss)
