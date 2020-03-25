#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt

from moneynet.utils.datasets.pikachu_dataset import Pikachu

import configargparse

from tqdm import tqdm


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

    return parser


def set_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.autoscale(enable=True, axis='x', tight=True)


def attention(x, y, temper, type=2, pseudo_zero=1e-6):
    energy = torch.pow(x, 2).sum(dim=-1, keepdim=True)
    mask_trivial = energy < pseudo_zero
    if type == 1:
        denom = torch.norm(x, dim=-1, keepdim=True) * torch.norm(y, dim=-1, keepdim=True)
        score = x * y / denom
        score = score / temper
        max_x = torch.max(score, dim=-1, keepdim=True)[0]
        exp_x = torch.exp(score - max_x)
        attn = exp_x / torch.sum(exp_x, dim=-1, keepdim=True)
        attn = attn.masked_fill(mask_trivial, 1 / attn.size(-1))
    if type == 2:
        score = (x * y)
        score[score < pseudo_zero] = pseudo_zero
        score = score / score.sum(dim=-1, keepdim=True)
        score = torch.exp(torch.log(score) / temper)
        attn = score / score.sum(dim=-1, keepdim=True)

    return attn


def max_variance(p, dim=-1):
    mean = torch.max(p, dim=dim, keepdim=True)[0]  # (B, T, 1)
    # get normalized confidence
    numer = (mean - p).pow(2)
    denom = p.size(dim) - 1
    return torch.sum(numer, dim=-1) / denom


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    train_dataset = Pikachu(args=args, train=True)

    for x in train_dataset:
        # prepare data
        print(x['fname'])
        y = x['target'].clone()
        x = x['input'].clone()

        # hyperparameter
        iter = 10
        consistency_sim_th = 0.80
        pseudo_zero = 1e-8
        temper = 0.08

        # buffer
        attns = []
        x_res = []

        # base data setting
        base_x = x.clone()
        # base_denom = torch.pow(base_x, 2).sum(-1)
        for i in range(iter):
            # initialization loop variable
            j = 0
            x_ = x.clone()
            while True:
                # check how much data is left.
                attn = attention(x_, y, temper=temper, type=2, pseudo_zero=pseudo_zero)  # (B, T)
                assert torch.isnan(attn).sum() == 0.0
                var = max_variance(attn, dim=-1)  # (B, T, 1)
                if j == 0:
                    att_init = attn
                    attnadd = attn
                else:
                    attnadd += attn

                # compute residual data
                x_ = (x_ - x_ * attn)

                # check similarity of each data frame
                denom = torch.norm(attn, dim=-1) * torch.norm(att_init, dim=-1)
                sim = torch.sum(attn * att_init, dim=-1) / denom  # (B, T)
                sim = sim.mean()
                end_condition = var.mean() < 1e-5
                if sim < consistency_sim_th or end_condition:
                    break
                pow_res = torch.pow(x, 2).sum(-1)  # (B, T)
                j += 1

            attns.append((attnadd * x)[0])
            x = x_
            x_res.append(x[0])
            print(i, j, float(sim), var.mean(), pow_res.mean())
            print(var)
            if end_condition:
                break

        iter = len(x_res) + 1

        # buffer display for evaluating
        fig = plt.figure()
        ax = fig.add_subplot(iter, 2, 2)
        ax.imshow(base_x[0].numpy().T, aspect='auto')
        set_style(ax)
        for i, x_re in enumerate(x_res):
            ax = fig.add_subplot(iter, 2, 2 * (i + 2))
            ax.imshow(x_re.numpy().T, aspect='auto')
            set_style(ax)
        for j, attn in enumerate(attns):
            ax = fig.add_subplot(iter, 2, 2 * j + 1)
            ax.imshow(attn.numpy().T, aspect='auto')
            set_style(ax)
        # fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        plt.show()
        break


if __name__ == '__main__':
    main()
