#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

import matplotlib.pyplot as plt

from moneynet.utils.datasets.pikachu_dataset import Pikachu

import configargparse


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


from moneynet.nets.unsup.utils import temp_softmax


def temp_softmax(x, T=10.0, dim=-1):
    x = x / T
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_x)
    x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    return x


def attention(x, y, temper, type=2):
    if type == 1:
        denom = (torch.norm(x, dim=-1, keepdim=True) * torch.norm(y, dim=-1, keepdim=True) + 1e-6)
        score = x * y / denom
        attn = temp_softmax(score, T=temper, dim=-1)
    if type == 2:
        attn = (x * y) / (x * y).sum(dim=-1, keepdim=True)
    attn[torch.isnan(attn)] = 0.0
    return attn


def cosim(x, y):
    denom = (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6)
    score = torch.sum(y * x, dim=-1) / denom
    return score


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    train_dataset = Pikachu(args=args, train=True)

    for x in train_dataset:
        # prepare data
        print(x['fname'])
        x = x['input'].to('cuda').clone()

        # hyperparameter
        iter = 10
        consistency_sim_th = 0.99
        pseudo_zero = 1e-6

        # buffer
        attns = []
        x_res = []

        # base data setting
        base_x = x.clone()
        base_denom = torch.abs(base_x).sum(-1)
        for i in range(iter):
            # initialization loop variable
            j = 0
            while True:
                # check how much data is left.
                pow_res = torch.abs(x).sum(-1) / base_denom  # (B, T)
                attn = attention(x, base_x, temper=0.1)  # (B, T)
                if j == 0:
                    att_init = attn
                    attnadd = attn
                else:
                    attnadd += attn

                # compute residual data
                x = (x - x * attn)

                # check similarity of each data frame
                mask_trivial = pow_res > pseudo_zero
                denom = torch.norm(attn, dim=-1) * torch.norm(att_init, dim=-1)
                sim = torch.sum(attn * att_init, dim=-1) / denom  # (B, T)
                sim = sim.masked_select(mask_trivial).mean()
                # print(mask_trivial.float().sum() / (mask_trivial.size(0) * mask_trivial.size(1)))
                if sim < consistency_sim_th:
                    break
                if pow_res.mean() < pseudo_zero:
                    break
                j += 1

            attns.append(attnadd[0])
            x_res.append(x[0])
            print(i, j, float(pow_res.mean()), float(sim))
            if pow_res.mean() < pseudo_zero:
                break

        iter = len(x_res)
        if iter == 10:
            break

        # buffer display for evaluating
        for i, x_re in enumerate(x_res):
            plt.subplot(iter, 2, 2 * (i + 1))
            plt.imshow(x_re.cpu().numpy().T, aspect='auto')
        for j, attn in enumerate(attns):
            plt.subplot(iter, 2, 2 * (j) + 1)
            plt.imshow(attn.cpu().numpy().T, aspect='auto')
        plt.show()


if __name__ == '__main__':
    main()