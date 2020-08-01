#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Automatic speech recognition model training script."""

import logging
import os
import random
import subprocess
import sys

from distutils.version import LooseVersion

import configargparse
import numpy as np
import torch

from espnet.utils.cli_utils import strtobool
from espnet.utils.training.batchfy import BATCH_COUNT_CHOICES

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion("1.2")


# NOTE: you need this func to generate our sphinx doc
def get_parser(parser=None, required=True):
    """Get default arguments."""
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model on one CPU, "
            "one or multiple GPUs",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="second config file path that overwrites the settings in `--config`.",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="third config file path that overwrites the settings in "
        "`--config` and `--config2`.",
    )

    parser.add_argument(
        "--ngpu",
        default=None,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--train-dtype",
        default="float32",
        choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
        help="Data type for training (only pytorch backend). "
        "O0,O1,.. flags require apex. "
        "See https://nvidia.github.io/apex/amp.html#opt-levels",
    )
    parser.add_argument(
        "--backend",
        default="pytorch",
        type=str,
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument(
        "--outdir", type=str, help="Output directory"
    )
    parser.add_argument("--debugmode", default=1, type=int, help="Debugmode")
    parser.add_argument("--dict", help="Dictionary")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--debugdir", type=str, help="Output directory for debugging")
    parser.add_argument(
        "--resume",
        "-r",
        default="",
        nargs="?",
        help="Resume the training from snapshot",
    )
    parser.add_argument(
        "--minibatches",
        "-N",
        type=int,
        default="-1",
        help="Process only N minibatches (for debug)",
    )
    parser.add_argument("--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        nargs="?",
        help="Tensorboard log dir path",
    )
    parser.add_argument(
        "--report-interval-iters",
        default=100,
        type=int,
        help="Report interval iterations",
    )
    parser.add_argument(
        "--save-interval-iters",
        default=0,
        type=int,
        help="Save snapshot interval iterations",
    )
    # task related
    parser.add_argument(
        "--train-json",
        type=str,
        default=None,
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default=None,
        help="Filename of validation label data (json)",
    )
    # network architecture
    parser.add_argument(
        "--model-module",
        type=str,
        default=None,
        help="model defined module (default: espnet.nets.xxx_backend.e2e_asr:E2E)",
    )

    # loss related
    # minibatch related
    parser.add_argument(
        "--sortagrad",
        default=0,
        type=int,
        nargs="?",
        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",
    )
    parser.add_argument(
        "--batch-count",
        default="auto",
        choices=BATCH_COUNT_CHOICES,
        help="How to count batch_size. "
        "The default (auto) will find how to count by args.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch-seqs",
        "-b",
        default=0,
        type=int,
        help="Maximum seqs in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-bins",
        default=0,
        type=int,
        help="Maximum bins in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-in",
        default=0,
        type=int,
        help="Maximum input frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-out",
        default=0,
        type=int,
        help="Maximum output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--batch-frames-inout",
        default=0,
        type=int,
        help="Maximum input+output frames in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--maxlen-in",
        "--batch-seq-maxlen-in",
        default=800,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the input sequence length > ML.",
    )
    parser.add_argument(
        "--maxlen-out",
        "--batch-seq-maxlen-out",
        default=150,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the output sequence length > ML",
    )
    parser.add_argument(
        "--n-iter-processes",
        default=0,
        type=int,
        help="Number of processes of iterator",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        nargs="?",
        help="The configuration file for the pre-processing",
    )
    # optimization related
    parser.add_argument(
        "--opt",
        default="adadelta",
        type=str,
        choices=["adadelta", "adam", "noam", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--accum-grad", default=1, type=int, help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Epsilon constant for optimizer"
    )
    parser.add_argument(
        "--eps-decay", default=0.01, type=float, help="Decaying ratio of epsilon"
    )
    parser.add_argument(
        "--weight-decay", default=0.0, type=float, help="Weight decay ratio"
    )
    parser.add_argument(
        "--criterion",
        default="loss",
        type=str,
        choices=["loss"],
        help="Criterion to perform epsilon decay",
    )
    parser.add_argument(
        "--threshold", default=1e-4, type=float, help="Threshold to stop iteration"
    )
    parser.add_argument(
        "--epochs", "-e", default=30, type=int, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--early-stop-criterion",
        default="validation/main/acc",
        type=str,
        nargs="?",
        help="Value to monitor to trigger an early stopping of the training",
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs to wait without improvement "
        "before stopping the training",
    )
    parser.add_argument(
        "--grad-clip", default=5, type=float, help="Gradient norm threshold to clip"
    )
    parser.add_argument(
        "--num-save-attention",
        default=3,
        type=int,
        help="Number of samples of attention to be saved",
    )
    parser.add_argument(
        "--grad-noise",
        type=strtobool,
        default=False,
        help="The flag to switch to use noise injection to gradients during training",
    )
    # front end related
    parser.add_argument(
        "--use-frontend",
        type=strtobool,
        default=False,
        help="The flag to switch to use frontend system.",
    )

    # Feature transform: Normalization
    parser.add_argument(
        "--stats-file",
        type=str,
        default=None,
        help="The stats file for the feature normalization",
    )
    parser.add_argument(
        "--apply-uttmvn",
        type=strtobool,
        default=True,
        help="Apply utterance level mean " "variance normalization.",
    )
    parser.add_argument("--uttmvn-norm-means", type=strtobool, default=True, help="")
    parser.add_argument("--uttmvn-norm-vars", type=strtobool, default=False, help="")
    # Feature transform: Fbank
    parser.add_argument(
        "--fbank-fs",
        type=int,
        default=16000,
        help="The sample frequency used for " "the mel-fbank creation.",
    )
    parser.add_argument(
        "--n-mels", type=int, default=80, help="The number of mel-frequency bins."
    )
    parser.add_argument("--fbank-fmin", type=float, default=0.0, help="")
    parser.add_argument("--fbank-fmax", type=float, default=None, help="")
    return parser


def main(cmd_args):
    """Run the main training function."""
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    if args.backend == "chainer" and args.train_dtype != "float32":
        raise NotImplementedError(
            f"chainer backend does not support --train-dtype {args.train_dtype}."
            "Use --dtype float32."
        )
    if args.ngpu == 0 and args.train_dtype in ("O0", "O1", "O2", "O3", "float16"):
        raise ValueError(
            f"--train-dtype {args.train_dtype} does not support the CPU backend."
        )

    from espnet.utils.dynamic_import import dynamic_import

    if args.model_module is None:
        model_module = "espnet.nets." + args.backend + "_backend.e2e_asr:E2E"
    else:
        model_module = args.model_module
    model_class = dynamic_import(model_module)
    model_class.add_arguments(parser)

    args = parser.parse_args(cmd_args)
    args.model_module = model_module
    if "chainer_backend" in args.model_module:
        args.backend = "chainer"
    if "pytorch_backend" in args.model_module:
        args.backend = "pytorch"

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # If --ngpu is not given,
    #   1. if CUDA_VISIBLE_DEVICES is set, all visible devices
    #   2. if nvidia-smi exists, use all devices
    #   3. else ngpu=0
    if args.ngpu is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is not None:
            ngpu = len(cvd.split(","))
        else:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
            try:
                p = subprocess.run(
                    ["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                ngpu = 0
            else:
                ngpu = len(p.stderr.decode().split("\n")) - 1
    else:
        if is_torch_1_2_plus and args.ngpu != 1:
            logging.debug(
                "There are some bugs with multi-GPU processing in PyTorch 1.2+"
                + " (see https://github.com/pytorch/pytorch/issues/21108)"
            )
        ngpu = args.ngpu
    logging.info(f"ngpu: {ngpu}")

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # set random seed
    logging.info("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list
    else:
        args.char_list = None

    # train
    logging.info("backend = " + args.backend)

    from moneynet.defect.pytorch_pipe.defect import train
    train(args)


if __name__ == "__main__":
    main(sys.argv[1:])
