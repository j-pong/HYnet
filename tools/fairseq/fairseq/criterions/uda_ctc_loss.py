# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round


@dataclass
class UdaCtcCriterion(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )

    uda_alpha: float = field(
        default=1.0,
        metadata={
            "help": "loss weight for uda consistency loss"
        },
    )
    proj_uda: bool = field(
        default=False,
        metadata={"help": "true if not uda KL-Div loss but uda CTC loss"},
    )
    viterbi_uda: bool = field(
        default=False,
        metadata={"help": "true if not uda KL-Div loss but uda CTC loss"},
    )
    apply_mpl: bool = field(
        default=False,
        metadata={"help": "true for applying momentum pseudo-labeling"},
    )
    mpl_alpha: float = field(
        default=0.0,
        metadata={
            "help": "loss weight for uda consistency loss"
        },
    )


@register_criterion("uda_ctc", dataclass=UdaCtcCriterion)
class UdaCtcCriterion(FairseqCriterion):
    def __init__(self, cfg: UdaCtcCriterion, task: FairseqTask):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None:
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        if cfg.viterbi_uda:
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.word_score = -1
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0
            dec_args.labels = "ltr"

            self.viterbi_decoder = W2lViterbiDecoder(dec_args, task.target_dictionary)
        else:
            self.viterbi_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

        self.uda_alpha = cfg.uda_alpha
        self.proj_uda = True if self.viterbi_decoder is not None else cfg.proj_uda
        self.apply_mpl = cfg.apply_mpl
        self.off_model = None
        self.on_model_tmp = None
        self.mpl_alpha = cfg.mpl_alpha

    def forward(self, model, sample, reduce=True):
        if "mode" not in sample["net_input"]:
            sample["net_input"]["mode"] = "labeled"

        if sample["net_input"]["mode"] == "labeled":
            uda_loss = torch.tensor(0)
            uda_sample_size = 0
            uda_ntokens = 0
            uda_nsentences = 0

            net_output = model(**sample["net_input"])

            lprobs = model.get_normalized_probs(
                net_output, log_probs=True
            ).contiguous()  # (T, B, C) from the encoder

            if "src_lengths" in sample["net_input"]:
                input_lengths = sample["net_input"]["src_lengths"]
            else:
                if net_output["padding_mask"] is not None:
                    non_padding_mask = ~net_output["padding_mask"]
                    input_lengths = non_padding_mask.long().sum(-1)
                else:
                    input_lengths = lprobs.new_full(
                        (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                    )

            pad_mask = (sample["target"] != self.pad_idx) & (
                    sample["target"] != self.eos_idx
            )
            targets_flat = sample["target"].masked_select(pad_mask)
            if "target_lengths" in sample:
                target_lengths = sample["target_lengths"]
            else:
                target_lengths = pad_mask.sum(-1)

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = F.ctc_loss(
                    lprobs,
                    targets_flat,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="sum",
                    zero_infinity=self.zero_infinity,
                )

            ctc_ntokens = (
                sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
            )
            ctc_sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
            ctc_nsentences = sample["id"].numel()

            loss = ctc_loss
            sample_size = ctc_sample_size

        # uda consistency training for unlabeled dataset
        elif sample["net_input"]["mode"] == "unlabeled":
            # set logging output elements to default 0
            ctc_loss = torch.tensor(0)
            ctc_ntokens = 0
            ctc_sample_size = 0
            ctc_nsentences = 0

            # forward original input with no gradient path
            if self.apply_mpl:
                # Momentum Pseudo-labeling
                import copy
                import os
                from fairseq.models.wav2vec.wav2vec2_uda import Wav2VecUDA, Wav2VecUdaEncoder
                if self.on_model_tmp is None:
                    pt_name = "temp" + str(sample["net_input"]["source"].device) + ".pt"
                    torch.save(model.state_dict(), pt_name)
                    off_w2v_encoder = Wav2VecUdaEncoder(model.w2v_encoder.cfg, model.w2v_encoder.output_size)
                    self.on_model_tmp = Wav2VecUDA(model.cfg, off_w2v_encoder).to(torch.device("cuda"))
                    self.on_model_tmp = self.on_model_tmp.half()
                    self.on_model_tmp.load_state_dict(torch.load(pt_name))
                    os.remove(pt_name)
                    self.on_model_tmp.eval()
                if self.off_model is None:
                    self.off_model = self.on_model_tmp
                self.off_model.eval()

                updated_flag = False
                with torch.no_grad():
                    for off_p, on_p_tmp, on_p in zip(self.off_model.parameters(), self.on_model_tmp.parameters(), model.parameters()):
                        updated = on_p_tmp != on_p
                        if True in updated:
                            off_p.copy_(self.mpl_alpha * off_p.data + (1 - self.mpl_alpha) * on_p.data)
                    if updated_flag:
                        self.on_model_tmp = None
                    
                    self.off_model.w2v_encoder.apply_mask = False
                    net_output = self.off_model(proj=self.proj_uda, **sample["net_input"])
            else:
                with torch.no_grad():
                    model.w2v_encoder.apply_mask = False
                    net_output = model(proj=self.proj_uda, **sample["net_input"])

            # forward perturbed input
            model.w2v_encoder.apply_mask = True
            ptb_net_output = model(proj=self.proj_uda, **sample["net_input"])

            # get log softmax probability
            lprobs = model.get_normalized_probs(
                net_output, log_probs=True
            ).contiguous()  # (T, B, C) from the encoder
            ptb_lprobs = model.get_normalized_probs(
                ptb_net_output, log_probs=True
            ).contiguous()  # (T, B, C) from the encoder

            # if viterbi_uda argument is true, then targets for consistency training are set to
            # decoded outputs (pseudo-label) through viterbi decoding algorithm
            if self.viterbi_decoder:
                device = lprobs.device
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()
                decoded_batch = []

                for lp in lprobs_t:
                    lp = lp.unsqueeze(0)
                    decoded = self.viterbi_decoder.decode(lp)
                    decoded = decoded[0][0]
                    decoded_batch.append(decoded["tokens"])
                decoded_flat = torch.cat(decoded_batch).to(device)

            # use KL-divergence for consistency regularization in hidden space,
            # or CTC for consistency regularization in target space
            with torch.backends.cudnn.flags(enabled=False):
                if not self.viterbi_decoder:
                    uda_loss = F.kl_div(
                        ptb_lprobs,
                        lprobs,
                        reduction="sum",
                        log_target=True,
                    )
                else:
                    if "src_lengths" in sample["net_input"]:
                        input_lengths = sample["net_input"]["src_lengths"]
                    else:
                        input_lengths = ptb_lprobs.new_full(
                            (ptb_lprobs.size(1),), ptb_lprobs.size(0), dtype=torch.long
                        )
                    target_lengths = torch.tensor([len(decoded_batch[i]) for i in range(len(decoded_batch))]).to(device)
                    uda_loss = F.ctc_loss(
                        ptb_lprobs,
                        decoded_flat,
                        input_lengths,
                        target_lengths,
                        blank=self.blank_idx,
                        reduction="sum",
                        zero_infinity=self.zero_infinity,
                    )
                uda_loss = self.uda_alpha * uda_loss
            if net_output["freeze"] or net_output["freeze_uda"]:
                uda_loss = uda_loss*0

            uda_sample_size = len(net_output["encoder_out"])
            uda_ntokens = uda_sample_size
            uda_nsentences = sample["id"].numel()

            loss = uda_loss
            sample_size = uda_sample_size

        logging_output = {
            "loss": utils.item(loss.data) / sample_size,
            "ctc_loss": utils.item(ctc_loss.data),  # * sample['ntokens'],
            "uda_loss": utils.item(uda_loss.data),
            "ctc_ntokens": ctc_ntokens,
            "uda_ntokens": uda_ntokens,
            "ctc_nsentences": ctc_nsentences,
            "uda_nsentences": uda_nsentences,
            "ctc_sample_size": ctc_sample_size,
            "uda_sample_size": uda_sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        uda_loss_sum = utils.item(sum(log.get("uda_loss", 0) for log in logging_outputs))
        ctc_ntokens = utils.item(sum(log.get("ctc_ntokens", 0) for log in logging_outputs))
        uda_ntokens = utils.item(sum(log.get("uda_ntokens", 0) for log in logging_outputs))
        ctc_nsentences = utils.item(
            sum(log.get("ctc_nsentences", 0) for log in logging_outputs)
        )
        uda_nsentences = utils.item(
            sum(log.get("uda_nsentences", 0) for log in logging_outputs)
        )
        ctc_sample_size = utils.item(
            sum(log.get("ctc_sample_size", 0) for log in logging_outputs)
        )
        uda_sample_size = utils.item(
            sum(log.get("uda_sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / ctc_sample_size / math.log(2) if ctc_sample_size != 0 else 0, ctc_sample_size, round=3
        )
        metrics.log_scalar(
            "uda_loss", uda_loss_sum / uda_sample_size / math.log(2) if uda_sample_size != 0 else 0, uda_sample_size, round=3
        )
        metrics.log_scalar("ctc_ntokens", ctc_ntokens)
        metrics.log_scalar("ctc_nsentences", ctc_nsentences)
        metrics.log_scalar("uda_ntokens", uda_ntokens)
        metrics.log_scalar("uda_nsentences", uda_nsentences)
        if ctc_sample_size != ctc_ntokens:
            metrics.log_scalar(
                "ctc_nll_loss", ctc_loss_sum / ctc_ntokens / math.log(2) if ctc_ntokens != 0 else 0, ctc_ntokens, round=3
            )
        if uda_sample_size != uda_ntokens:
            metrics.log_scalar(
                "uda_nll_loss", uda_loss_sum / uda_ntokens / math.log(2) if uda_ntokens != 0 else 0, uda_ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
