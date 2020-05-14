#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""RNN sequence-to-sequence speech recognition model (pytorch)."""

from distutils.version import LooseVersion
import argparse
from itertools import groupby
import logging
import math
import os

import chainer
from chainer import reporter
import editdistance
import numpy as np
import six
import torch
import torch.nn.functional as F

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.frontends.feature_transform import (
    feature_transform_for,  # noqa: H301
)
from espnet.nets.pytorch_backend.frontends.frontend import frontend_for
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.cli_utils import strtobool

from moneynet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from moneynet.nets.pytorch_backend.initialization import orthogonal_init_parameters
from moneynet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from moneynet.nets.pytorch_backend.rnn.encoders import encoder_for

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""
    
    def report(self, loss_ctc, loss_ce, acc, cer_ctc, cer, wer, mtl_loss):
        """Report at every step."""
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_ce": loss_ce}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2E.encoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        """Add arguments for the encoder."""
        group = parser.add_argument_group("E2E encoder setting")
        # encoder
        group.add_argument(
            "--etype",
            default="blstmp",
            type=str,
            choices=[
                "lstm",
                "blstm",
                "lstmp",
                "blstmp",
                "vgglstmp",
                "vggblstmp",
                "vgglstm",
                "vggblstm",
                "gru",
                "bgru",
                "grup",
                "bgrup",
                "vgggrup",
                "vggbgrup",
                "vgggru",
                "vggbgru",
            ],
            help="Type of encoder network architecture",
        )
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers "
            "(for shared recognition part in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--eprojs", default=320, type=int, help="Number of encoder projection units"
        )
        group.add_argument(
            "--dropout-rate", default=None, type=float, help="dropout in rnn layers. use --dropout-rate if None is set"
        )
        group.add_argument(
            "--subsample",
            default="1",
            type=str,
            help="Subsample input frames x_y_z means "
            "subsample every x frame at 1st layer, "
            "every y frame at 2nd layer etc.",
        )

        group.add_argument(
            "--sampling-probability",
            default=0.0,
            type=float,
            help="Ratio of predicted labels fed back to decoder",
        )
        group.add_argument(
            "--lsm-type",
            const="",
            default="",
            type=str,
            nargs="?",
            choices=["", "unigram"],
            help="Apply label smoothing with a specified distribution type",
        )

        # output
        group.add_argument(
            "--oversampling", default=1, type=int, help=""
        )
        group.add_argument(
            "--outer", default=0, type=int, help=""
        )
        group.add_argument(
            "--residual", default=0, type=int, help=""
        )
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super(E2E, self).__init__()
        torch.nn.Module.__init__(self)
        self.mtlalpha = args.mtlalpha
        assert 0.0 <= self.mtlalpha <= 1.0, "mtlalpha should be [0.0, 1.0]"
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        self.outdir = args.outdir

        # target matching system organization
        self.oversampling = args.oversampling
        self.residual = args.residual
        self.outer = args.outer
        self.poster = torch.nn.Linear(args.eprojs, odim * self.oversampling)
        if self.outer:
            if self.residual:
                self.matcher_res = torch.nn.Linear(idim, odim)
                self.matcher = torch.nn.Linear(odim, odim)
            else:
                self.matcher = torch.nn.Linear(odim + idim, odim)

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="rnn")
        self.reporter = Reporter()

        # label smoothing info
        if args.lsm_type and os.path.isfile(args.train_json):
            logging.info("Use label smoothing with " + args.lsm_type)
            labeldist = label_smoothing_dist(
                odim, args.lsm_type, transcript=args.train_json
            )
        else:
            labeldist = None

        if getattr(args, "use_frontend", False):  # use getattr to keep compatibility
            self.frontend = frontend_for(args, idim)
            self.feature_transform = feature_transform_for(args, (idim - 1) * 2)
            idim = args.n_mels
        else:
            self.frontend = None

        # encoder
        self.enc = encoder_for(args, idim, self.subsample)
        # ctc
        self.ctc = ctc_for(args, odim)

        # weight initialization
        self.init_like_chainer()

        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        self.rnnlm = None

        self.logzero = -10000000000.0
        self.loss = None
        self.acc = None

    def init_like_chainer(self):
        """Initialize weight like chainer.

        chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
        pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
        however, there are two exceptions as far as I know.
        - EmbedID.W ~ Normal(0, 1)
        - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
        """
        lecun_normal_init_parameters(self)
        # exceptions
        # embed weight ~ Normal(0, 1)
        # forget-bias = 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745

    def init_orthogonal(self):
        """Initialize weight orthogonal
        initiate bias to zero
        initiate linear weight orthogonal
        """
        orthogonal_init_parameters(self)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        # 0. Frontend
        if self.frontend is not None:
            hs_pad, hlens, mask = self.frontend(to_torch_tensor(xs_pad), ilens)
            hs_pad, hlens = self.feature_transform(hs_pad, hlens)
        else:
            hs_pad, hlens = xs_pad, ilens

        # 1. RNN Encoder
        hs_pad, hlens, _ = self.enc(hs_pad, hlens)

        # TODO: Not sure about oversampling & outer
        # 2. post-processing layer for target dimension
        if self.outer:
            post_pad = self.poster(hs_pad)
            post_pad = post_pad.view(post_pad.size(0), -1, self.odim)
            if post_pad.size(1) != xs_pad.size(1):
                if post_pad.size(1) < xs_pad.size(1):
                    xs_pad = xs_pad[:, :post_pad.size(1)].contiguous()
                else:
                    raise ValueError("target size {} and pred size {} is mismatch".format(xs_pad.size(1), post_pad.size(1)))
            if self.residual:
                post_pad = post_pad + self.matcher_res(xs_pad)
            else:
                post_pad = torch.cat([post_pad, xs_pad], dim=-1)
            pred_pad = self.matcher(post_pad)
        else:
            pred_pad = self.poster(hs_pad)
            pred_pad = pred_pad.view(pred_pad.size(0), -1, self.odim)
        self.pred_pad = pred_pad
        if pred_pad.size(1) != ys_pad.size(1):
            if pred_pad.size(1) < ys_pad.size(1):
                ys_pad = ys_pad[:, :pred_pad.size(1)].contiguous()
            else:
                raise ValueError("target size {} and pred size {} is mismatch".format(ys_pad.size(1), pred_pad.size(1)))

        # 3. CTC loss
        if self.mtlalpha == 0:
            self.loss_ctc = None
        else:
            self.loss_ctc = self.ctc(hs_pad, hlens, ys_pad)

        # 3. CE loss
        if LooseVersion(torch.__version__) < LooseVersion("1.0"):
            reduction_str = "elementwise_mean"
        else:
            reduction_str = "mean"
        self.loss_ce = F.cross_entropy(
            pred_pad.view(-1, self.odim),
            ys_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction=reduction_str,
        )
        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_pad, ignore_label=self.ignore_id
        )

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer, cer_ctc = None, None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = self.loss_ce
            loss_ce_data = float(self.loss_ce)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = self.loss_ctc
            loss_ce_data = None
            loss_ctc_data = float(self.loss_ctc)
        else:
            self.loss = alpha * self.loss_ctc + (1 - alpha) * self.loss_ce
            loss_ce_data = float(self.loss_ce)
            loss_ctc_data = float(self.loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_ce_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            pass
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.dec, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: input acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        ilens = [x.shape[0]]

        # subsample frame
        x = x[:: self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)
        # make a utt list (1) to use the same interface for encoder
        hs = h.contiguous().unsqueeze(0)

        # 0. Frontend
        if self.frontend is not None:
            enhanced, hlens, mask = self.frontend(hs, ilens)
            hs, hlens = self.feature_transform(enhanced, hlens)
        else:
            hs, hlens = hs, ilens

        # 1. forward RNN encoder
        hs, _, _ = self.enc(hs, hlens)
        return hs.squeeze(0)

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E beam search.

        :param ndarray x: input acoustic feature (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        hs = self.encode(x).unsqueeze(0)
        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(hs)[0]
        else:
            lpz = None

        hyps = self.poster(hs)
        hyps = hyps.view(-1, self.odim)

        logging.info("input lengths: " + str(hyps.size(0)))

        return hyps

    def subsample_frames(self, x):
        """Subsample speeh frames in the encoder."""
        # subsample frame
        x = x[:: self.subsample[0], :]
        ilen = [x.shape[0]]
        h = to_device(self, torch.from_numpy(np.array(x, dtype=np.float32)))
        h.contiguous()
        return h, ilen
