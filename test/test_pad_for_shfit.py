#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt

from moneynet.nets.utils import pad_for_shift, reverse_pad_for_shift, select_with_ind


def main():
    # case 0 select_with_ind check
    theta = torch.ones((2, 30)).long() * 10
    # theta = torch.arange(30).view(1, 30).repeat((2, 1))
    x = torch.rand(2 * 30 * 80).view(2, 30, 80)
    plt.subplot(1, 2, 1)
    plt.imshow(x[0].numpy())
    x_hat = torch.stack([select_with_ind(x, theta + i) for i in torch.arange(40)], dim=-1)
    plt.subplot(1, 2, 2)
    plt.imshow(x_hat[0].numpy())
    plt.show()

    # # case 1 pad check
    # # make random data
    # x = torch.rand(2 * 30 * 40).view(2, 30, 40)

    # # pad
    # x_aug, _ = pad_for_shift(x, pad=40 - 1, window=40, mask=False)
    # print(x_aug.size())
    #
    # # make virtual theta
    # theta = torch.ones((2, 30)) * 50
    #
    # # select with theta
    # x_opt = select_with_ind(x_aug, theta)
    # print(x_opt.size())
    #
    # # reverse pad
    # x_hat = reverse_pad_for_shift(x_opt, pad=40 - 1, theta=theta, window=40)
    # print(x_hat.size())
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(x_aug[0, 0].numpy())
    # plt.subplot(1, 3, 2)
    # plt.imshow(x_opt[0].numpy())
    # plt.subplot(1, 3, 3)
    # plt.imshow(x_hat[0].numpy())
    # plt.show()



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
