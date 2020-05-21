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

from moneynet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from moneynet.nets.pytorch_backend.initialization import orthogonal_init_parameters
from moneynet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from moneynet.nets.pytorch_backend.ICT.encoders import encoder_for
from moneynet.nets.pytorch_backend.ICT.nets_utils import mixup_data, mixup_logit
from moneynet.nets.pytorch_backend.ICT.nets_utils import get_current_consistency_weight

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss_ce, loss_mse, stu_acc, ema_acc, loss):
        """Report at every step."""
        reporter.report({"loss_ce": loss_ce}, self)
        reporter.report({"loss_mse": loss_mse}, self)
        reporter.report({"student_acc": stu_acc}, self)
        reporter.report({"teacher_acc": ema_acc}, self)
        logging.info("total loss:" + str(loss))
        reporter.report({"loss": loss}, self)


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
        self.etype = args.etype
        self.verbose = args.verbose
        # NOTE: for self.build method
        self.outdir = args.outdir

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="rnn")
        self.reporter = Reporter()

        # ICT related
        self.consistency_weight = args.consistency_weight
        self.consistency_rampup_starts = args.consistency_rampup_starts
        self.consistency_rampup_ends = args.consistency_rampup_ends
        self.mixup_alpha = args.mixup_alpha

        # if True, print out student model accuracy
        self.show_student_model_acc = args.show_student_model_acc

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
        self.enc = encoder_for(args, idim, odim, self.subsample)
        self.ema_enc = encoder_for(args, idim, odim, self.subsample)
        for param in self.ema_enc.parameters():
            param.detach_()
        # leave ctc for future works
        # self.ctc = ctc_for(args, odim)

        # weight initialization
        if args.initializer == "lecun":
            self.init_like_chainer()
        elif args.initializer == "orthogonal":
            self.init_orthogonal()
        else:
            raise NotImplementedError(
                "unknown initializer: " + args.initializer
            )

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

    def match_pad(self, pred_pad, ys_pad):
        pred_pad = pred_pad.view(pred_pad.size(0), -1, self.odim)
        if pred_pad.size(1) != ys_pad.size(1):
            if pred_pad.size(1) < ys_pad.size(1):
                ys_pad = ys_pad[:, :pred_pad.size(1)].contiguous()
            else:
                raise ValueError(
                    "target size {} and pred size {} is mismatch".format(ys_pad.size(1), pred_pad.size(1)))
        return pred_pad, ys_pad

    def forward(self, xs_pad, ilens, ys_pad, ul_xs_pad, ul_ilens, ul_ys_pad, process_info):
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
            ul_hs_pad, ul_hlens, ul_mask = self.frontend(to_torch_tensor(ul_xs_pad), ul_ilens)
            ul_hs_pad, ul_hlens = self.feature_transform(ul_hs_pad, ul_hlens)
        else:
            hs_pad, hlens, ul_hs_pad, ul_hlens = xs_pad, ilens, ul_xs_pad, ul_ilens

        # Calculating student model accuracy consumes twice the time.
        # Recommend to keep show-student-model-acc argument False
        if self.show_student_model_acc:
            ul_pred_pad, ul_hlens, _ = self.enc(ul_hs_pad, ul_hlens)
            ul_pred_pad, ul_ys_pad = self.match_pad(ul_pred_pad, ul_ys_pad)
            self.stu_acc = th_accuracy(
                ul_pred_pad.view(-1, self.odim), ul_ys_pad, ignore_label=self.ignore_id
            )
        else:
            self.stu_acc = None

        # 1. Mixup feature
        if self.mixup_alpha > 0.0:
            hs_pad, ys_pad, ys_pad_b, _, lam = mixup_data(hs_pad, ys_pad, hlens, self.mixup_alpha)
            ul_hs_pad_mixed, ul_ys_pad, _, ul_shuf_idx, ul_lam = mixup_data(ul_hs_pad, ul_ys_pad, ul_hlens, self.mixup_alpha)

        # 2. RNN Encoder
        pred_pad, hlens, _ = self.enc(hs_pad, hlens)
        if self.mixup_alpha > 0.0:
            ul_pred_pad, ul_hlens, _ = self.enc(ul_hs_pad_mixed, ul_hlens)
        else:
            ul_pred_pad, ul_hlens, _ = self.enc(ul_hs_pad, ul_hlens)
        ema_ul_pred_pad, ema_ul_hlens, _ = self.ema_enc(ul_hs_pad, ul_hlens)

        # 3. post-processing layer for target dimension
        pred_pad, ys_pad = self.match_pad(pred_pad, ys_pad)
        if self.mixup_alpha > 0.0:
            pred_pad, ys_pad_b = self.match_pad(pred_pad, ys_pad_b)
        ul_pred_pad, ul_ys_pad = self.match_pad(ul_pred_pad, ul_ys_pad)
        ema_ul_pred_pad, ul_ys_pad = self.match_pad(ema_ul_pred_pad, ul_ys_pad)

        # 4. mixup ema model output
        # Calculate EMA model accuracy before mixup
        self.ema_acc = th_accuracy(
            ema_ul_pred_pad.view(-1, self.odim), ul_ys_pad, ignore_label=self.ignore_id
        )
        if self.mixup_alpha > 0.0:
            ema_ul_pred_pad = mixup_logit(ema_ul_pred_pad, ul_hlens, ul_shuf_idx, ul_lam)

        # 5. Supervised loss
        if LooseVersion(torch.__version__) < LooseVersion("1.0"):
            reduction_str = "elementwise_mean"
        else:
            reduction_str = "mean"
        if self.mixup_alpha > 0.0:
            loss_ce_a = F.cross_entropy(
                pred_pad.view(-1, self.odim),
                ys_pad.view(-1),
                ignore_index=self.ignore_id,
                reduction=reduction_str,
            )
            loss_ce_b = F.cross_entropy(
                pred_pad.view(-1, self.odim),
                ys_pad_b.view(-1),
                ignore_index=self.ignore_id,
                reduction=reduction_str,
            )
            self.loss_ce = lam * loss_ce_a + (1 - lam) * loss_ce_b
        else:
            self.loss_ce = F.cross_entropy(
                pred_pad.view(-1, self.odim),
                ys_pad.view(-1),
                ignore_index=self.ignore_id,
                reduction=reduction_str,
            )

        # 6. Consistency loss
        self.loss_mse = F.mse_loss(
            ul_pred_pad.view(-1, self.odim),
            ema_ul_pred_pad.view(-1, self.odim),
            reduction=reduction_str
        )

        # 7. Total loss
        if process_info is not None:
            if process_info["epoch"] < self.consistency_rampup_starts:
                consistency_weight = 0
            else:
                consistency_weight = get_current_consistency_weight(
                    self.consistency_weight,
                    process_info["epoch"],
                    process_info["current_position"],
                    process_info["batch_len"],
                    self.consistency_rampup_starts,
                    self.consistency_rampup_ends
                )
        else:
            consistency_weight = 0
        self.loss = self.loss_ce + consistency_weight * self.loss_mse

        loss_ce_data = float(self.loss_ce)
        loss_mse_data = float(self.loss_mse)
        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ce_data, loss_mse_data, self.stu_acc, self.ema_acc, loss_data
            )
        else:
            pass
        return self.loss

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
        hyps = self.encode(x).unsqueeze(0)
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
