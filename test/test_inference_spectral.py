import torch
import torch.nn.functional as F

from moneynet.bin.unsup_train import get_parser
from moneynet.nets.pytorch_backend.ar_unsup_simnn import Net
from moneynet.nets.pytorch_backend.unsup.inference import Inference

import numpy as np

idim = 6
odim = 4

parser = get_parser()
Net.add_arguments(parser)

args = parser.parse_args()

net = Inference(idim, odim, args=args)

# random feature generation
x = torch.normal(mean=0, std=1, size=(3, 1, 1, 3, idim))  # x = torch.rand([1, 1, idim])  # B, 1, d
y_hat, p_hat = net(x)  # B, 1, d'
print(y_hat.view(-1, odim))

w_hat = p_hat[0]
bias_hat = p_hat[1]

# evaluation spectral disentangling
print(x.size(), w_hat.size())
y_hat_hat = (x.view(-1, idim).unsqueeze(-1) * w_hat).sum(-2) + bias_hat
print(y_hat_hat)

print(torch.abs(y_hat.view(-1, odim) - y_hat_hat).sum())

# w_hat_x = w_hat * torch.sign(x)
# w_hat_x_p = torch.relu(w_hat)
# w_hat_x_n = torch.relu(-w_hat)
# e_loss = torch.sum(w_hat_x_p) + torch.sum(w_hat_x_n)
# print(e_loss)
