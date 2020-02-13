#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import random
import subprocess
import sys

import configargparse
import numpy as np


# NOTE: you need this func to generate our sphinx doc
def get_parser():
    """Get parser of training arguments."""
    parser = configargparse.ArgumentParser(
        description='Training Unsupervised Sequential model on one CPU, one or multiple GPUs',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    # general configuration
    parser.add_argument('--ngpu', default=1, type=int,
                        help='Number of GPUs. If not given, use all visible devices')
    parser.add_argument('--train-dtype', default="float32",
                        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
                        help='Data type for training')
    parser.add_argument('--datadir', type=str, default='./data',
                        help='Raw data directory')
    parser.add_argument('--indir', type=str, required=True,
                        help='Input directory')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', type=str, nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--batch-size', '--batch-seqs', '-b', default=8, type=int,
                        help='Maximum seqs in a minibatch (0 to disable)')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?',
                        help="Tensorboard log directory path")
    parser.add_argument('--eval-interval-epochs', default=1, type=int,
                        help="Evaluation interval epochs")
    parser.add_argument('--save-interval-epochs', default=1, type=int,
                        help="Save interval epochs")
    parser.add_argument('--report-interval-iters', default=100, type=int,
                        help="Report interval iterations")

    # optimization related
    parser.add_argument('--opt', default='sgd', type=str,
                        choices=['adam', 'sgd'],
                        help='Optimizer')
    parser.add_argument('--accum-grad', default=1, type=int,
                        help='Number of gradient accumuration')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon for optimizer')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--epochs', '-e', default=30000, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=1, type=float,
                        help='Gradient norm threshold to clip')

    # task related
    parser.add_argument('--feat-type', default='mfcc', type=str,
                        choices=['stft', 'mfcc'],
                        help='Feature type for audio')
    parser.add_argument('--feat-dim', default=40, type=int,
                        help='Feature dimension')

    return parser


def main(cmd_args):
    """Run training."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(','))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(['nvidia-smi', '-L'],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split('\n')) - 1
    else:
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    from moneynet.unsup.unsup_ar import train
    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
