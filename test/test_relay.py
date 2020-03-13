#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def relay(theta, h, idim, cdim, hdim, energy_th):
    # :param torch.Tensor h: batch of padded source sequences (B, Tmax, hdim)
    h_energy = h.pow(2)
    indices = torch.topk(h_energy, k=cdim, dim=-1)[1]  # (B, T, cdim)

    move_mask = torch.abs(theta - idim + 1) > energy_th  # (B, T)
    cum_move_indices = move_mask.float().cumsum(-1).long()
    indices = [torch.cat([ind[0:1], ind[m]], dim=0) if ind[m].size(0) > 0 else ind[0:1]
               for m, ind in zip(move_mask, indices)]  # list (B,) with (T_b, cdim)

    indices = [indices[i][ind, :] for i, ind in enumerate(cum_move_indices)]
    indices = torch.stack(indices, dim=0) # (B, T, cdim)
    mask = F.one_hot(indices, num_classes=hdim).float().sum(-2)  # (B, T, hdim)

    return mask, move_mask


def main():
    # make random data
    h = torch.rand(2 * 30 * 66).view(2, 30, 66)
    plt.subplot(2, 2, 1)
    plt.imshow(h[0].numpy())

    # make virtual theta
    theta = torch.arange(60).view(2, 30)
    mask, move_mask = relay(theta, h, 40, 4, 66, 35)

    plt.subplot(2, 2, 2)
    plt.imshow(mask[0].numpy())
    plt.subplot(2, 1, 2)
    plt.imshow(move_mask.float().numpy())
    plt.show()


if __name__ == '__main__':
    main()
