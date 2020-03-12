#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt

from moneynet.nets.utils import pad_for_shift, reverse_pad_for_shift


def select_with_ind(x, ind):
    # get size of tensors
    x_size = x.size()
    ind_size = ind.size()
    assert len(x_size) > len(ind_size)

    # ind type check and change
    if ind.type() is not torch.LongTensor:
        ind = ind.long()

    # get new size
    x_new_size = []
    x_return_size = []
    for i, ind_s in enumerate(ind_size):
        if i == 0:
            ind_s_sum = ind_s
        else:
            ind_s_sum *= ind_s
        x_return_size.append(ind_s)
    x_new_size.append(ind_s_sum)
    for i, x_s in enumerate(x_size[i + 1:]):
        x_new_size.append(x_s)
        if i != 0:
            x_return_size.append(x_s)

    # select with ind
    x = x.view(x_new_size)
    ind = ind.view(x_new_size[0])
    x = x[torch.arange(x_new_size[0]), ind.long()].view(x_return_size)

    return x


def main():
    # case 1
    # make random data
    x = torch.rand(2 * 30 * 40).view(2, 30, 40)

    # pad
    x_aug, _ = pad_for_shift(x, pad=40-1, window=40, mask=False)
    print(x_aug.size())

    # make virtual theta
    theta = torch.ones((2, 30)) * 50

    # select with theta
    x_opt = select_with_ind(x_aug, theta)
    print(x_opt.size())

    # reverse pad
    x_hat = reverse_pad_for_shift(x_opt, pad=40-1, theta=theta, window=40)
    print(x_hat.size())

    plt.subplot(1, 3, 1)
    plt.imshow(x_aug[0,0].numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(x_opt[0].numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(x_hat[0].numpy())
    plt.show()

    # # case 2
    # # make random data
    # x = torch.rand(2 * 30 * 40).view(2, 30, 40)
    #
    # # pad
    # x_aug, _ = pad_for_shift(x, pad=40, window=80, mask=False)
    # print(x_aug.size())
    #
    # # make virtual theta
    # theta = torch.ones((2, 30)) * 10
    #
    # # select with theta
    # x_opt = select_with_ind(x_aug, theta)
    # print(x_opt.size())
    #
    # # reverse pad
    # x_hat = reverse_pad_for_shift(x_opt, pad=40, theta=theta, window=80)
    # print(x_hat.size())
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(x_opt[0].numpy())
    # plt.subplot(1, 2, 2)
    # plt.imshow(x_hat[0].numpy())
    # plt.show()


if __name__ == '__main__':
    main()
