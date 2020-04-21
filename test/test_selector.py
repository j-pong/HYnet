#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt

from moneynet.utils.datasets.pikachu_dataset import Pikachu
from moneynet.nets.unsup.utils import pad_for_shift, select_with_ind
import configargparse

import numpy as np


def get_parser():
    """Get parser of training arguments."""
    parser = configargparse.ArgumentParser(
        description='Training Unsupervised Sequential model on one CPU, one or multiple GPUs',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # general configuration
    parser.add_argument('--ngpu', default=1, type=int,
                        help='Number of GPUs. If not given, use all visible devices')
    parser.add_argument('--ncpu', default=16, type=int,
                        help='Number of CPUs. If not given, use all visible devices')
    parser.add_argument('--train-dtype', default="float32",
                        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
                        help='Data type for training')
    parser.add_argument('--indir', type=str, default='dump',
                        help='Input directory')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', type=str, nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--batch-size', '--batch-seqs', '-b', default=1, type=int,
                        help='Maximum seqs in a minibatch (0 to disable)')
    parser.add_argument('--eval-batch-size', '--eval-batch-seqs', '-eb', default=7, type=int,
                        help='')
    parser.add_argument('--low-interval-epochs', default=10, type=int,
                        help="Evaluation interval epochs")
    parser.add_argument('--high-interval-epochs', default=200, type=int,
                        help="Evaluation interval epochs")
    parser.add_argument('--save-interval-epochs', default=1000, type=int,
                        help="Save interval epochs")
    parser.add_argument('--pin-memory', default=0, type=int,
                        help='')
    parser.add_argument('--datamper', default=1, type=int,
                        help='')

    # task related
    parser.add_argument('--feat-type', default='mfcc', type=str,
                        choices=['stft', 'mfcc'],
                        help='Feature type for audio')
    parser.add_argument('--feat-dim', default=40, type=int,
                        help='Feature dimension')
    parser.add_argument('--ignore_in', default=0, type=float,
                        help='')
    parser.add_argument('--ignore_out', default=float('NaN'), type=float,
                        help='Hidden layer dimension')
    parser.add_argument('--hdim', default=1024, type=int,
                        help='Hidden layer dimension')
    parser.add_argument('--cdim', default=32, type=int,
                        help='')
    parser.add_argument('--similarity', default='cos', type=str,
                        choices=['cos'], help='Similarity metric')
    parser.add_argument('--temperature', default=0.01, type=float,
                        help='')
    parser.add_argument('--self-train', default=0, type=int,
                        help='')
    parser.add_argument('--encoder-type', default='linear', type=str,
                        choices=['linear', 'conv1d'], help='')
    parser.add_argument('--energy-threshold', default=10, type=float,
                        help='')
    parser.add_argument('--relay-bypass', default=0, type=int,
                        help='')
    parser.add_argument('--num-targets', default=16, type=int,
                        help='')

    return parser


def set_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.autoscale(enable=True, axis='x', tight=True)


def prob(x, temperature, pseudo_zero=1e-6):
    score = torch.abs(x)
    score[score < pseudo_zero] = pseudo_zero
    score = score / score.sum(dim=-1, keepdim=True)
    score = torch.exp(torch.log(score) / temperature)
    p = score / score.sum(dim=-1, keepdim=True)
    return p


def similarity(x, y, norm=True):
    if norm:
        denom = (torch.norm(y, dim=-1) * torch.norm(x, dim=-1) + 1e-6)
        sim = torch.sum(y * x, dim=-1) / denom
        sim[torch.isnan(sim)] = 0.0
    else:
        sim = torch.sum(y * x, dim=-1)
    return sim


def max_variance(p, dim=-1):
    mean = torch.max(p, dim=dim, keepdim=True)[0]  # (B, T, 1)
    # get normalized confidence
    numer = (mean - p).pow(2)
    denom = p.size(dim) - 1
    return torch.sum(numer, dim=-1) / denom


def reporter(max_idxs, x, p, i, fname):
    # buffer display for evaluating
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    v = max_idxs.numpy()
    fig.suptitle(fname)
    im = ax.imshow(v.T, aspect='auto')
    set_style(ax)
    ax.grid()
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 1, 2)
    v = x[0].numpy()
    im = ax.imshow(v.T, aspect='auto')
    set_style(ax)
    ax.grid()
    fig.colorbar(im, ax=ax)

    ax = fig.add_subplot(3, 1, 3)
    v = p[0].numpy()
    im = ax.imshow(v.T, aspect='auto')
    set_style(ax)
    ax.grid()
    fig.colorbar(im, ax=ax)

    # fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig("test/test_selector_{}.png".format(i))
    plt.close()


def main():
    # training process related
    iter = 10
    # action related
    pad = 40 - 1
    window = 40
    margin = 1
    # attention related
    temper = 0.1

    parser = get_parser()
    args, _ = parser.parse_known_args()

    train_dataset = Pikachu(args=args, train=True)

    for x in train_dataset:
        # prepare data
        filename = x['fname'][0]
        ys = torch.abs(x['target'].clone())
        x = torch.abs(x['input'].clone())

        # check how much data is left.
        p_prev = None
        for i in range(iter):
            x_aug, _ = pad_for_shift(key=x, pad=pad, window=window)
            max_idxs = []
            p_attn = None
            for k in range(args.num_targets):
                # select
                y = ys[:, :, k, :]
                score = torch.sum(y.unsqueeze(-2) * x, dim=-1)
                sim_max, sim_max_idx = torch.max(score, dim=-1)  # (B, Tmax)
                y_align = select_with_ind(x_aug, sim_max_idx)

                # find high contribution element
                score = torch.sum(y * y_align, dim=-1)
                if p_attn is None:
                    p_attn = prob(score, temperature=temper)
                else:
                    p_attn += prob(score, temperature=temper)

                # metaphysics point detection
                sim_max_idx = sim_max_idx - pad
                e_mask = torch.abs(sim_max_idx) > (k + margin)
                # remove a similarity allocated with the point
                sim_max_idx = sim_max_idx.masked_fill(e_mask, 0.0)
                # result buffering
                max_idxs.append(sim_max_idx[0])
                # y_aligns.append(y_align)
            max_idxs = torch.stack(max_idxs, dim=-1)

            p = prob(x, temperature=temper)
            reporter(max_idxs, x, p, i, filename)
            if p_prev is not None:
                print(similarity(p, p_prev) > 0.5)
            p_prev = p

            x = x * (1 - p)
            if i == 5:
                break
        break


if __name__ == '__main__':
    main()
