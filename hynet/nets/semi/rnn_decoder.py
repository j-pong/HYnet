import random

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.utils.get_default_kwargs import get_default_kwargs

from hynet.nets.semi.attentions import initial_att
from hynet.nets.semi.beam_search import BeamSearch
from hynet.nets.semi.utils import *

def build_attention_list(
    eprojs: int,
    dunits: int,
    atype: str = "location",
    num_att: int = 1,
    num_encs: int = 1,
    aheads: int = 4,
    adim: int = 320,
    awin: int = 5,
    aconv_chans: int = 10,
    aconv_filts: int = 100,
    han_mode: bool = False,
    han_type=None,
    han_heads: int = 4,
    han_dim: int = 320,
    han_conv_chans: int = -1,
    han_conv_filts: int = 100,
    han_win: int = 5,
):

    att_list = torch.nn.ModuleList()
    if num_encs == 1:
        for i in range(num_att):
            att = initial_att(
                atype,
                eprojs,
                dunits,
                aheads,
                adim,
                awin,
                aconv_chans,
                aconv_filts,
            )
            att_list.append(att)
    elif num_encs > 1:  # no multi-speaker mode
        if han_mode:
            att = initial_att(
                han_type,
                eprojs,
                dunits,
                han_heads,
                han_dim,
                han_win,
                han_conv_chans,
                han_conv_filts,
                han_mode=True,
            )
            return att
        else:
            att_list = torch.nn.ModuleList()
            for idx in range(num_encs):
                att = initial_att(
                    atype[idx],
                    eprojs,
                    dunits,
                    aheads[idx],
                    adim[idx],
                    awin[idx],
                    aconv_chans[idx],
                    aconv_filts[idx],
                )
                att_list.append(att)
    else:
        raise ValueError(
            "Number of encoders needs to be more than one. {}".format(num_encs)
        )
    return att_list


class BeamRNNDecoder(AbsDecoder):
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 320,
        sampling_probability: float = 0.0,
        dropout: float = 0.0,
        context_residual: bool = False,
        replace_sos: bool = False,
        num_encs: int = 1,
        att_conf: dict = get_default_kwargs(build_attention_list),
    ):
        # FIXME(kamo): The parts of num_spk should be refactored more more more
        assert check_argument_types()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported: rnn_type={rnn_type}")

        super().__init__()
        eprojs = encoder_output_size
        self.dtype = rnn_type
        self.dunits = hidden_size
        self.dlayers = num_layers
        self.context_residual = context_residual
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.odim = vocab_size
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        self.num_encs = num_encs

        # for multilingual translation
        self.replace_sos = replace_sos

        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(hidden_size + eprojs, hidden_size)
            if self.dtype == "lstm"
            else torch.nn.GRUCell(hidden_size + eprojs, hidden_size)
        ]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(hidden_size, hidden_size)
                if self.dtype == "lstm"
                else torch.nn.GRUCell(hidden_size, hidden_size)
            ]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf

        if context_residual:
            self.output = torch.nn.Linear(hidden_size + eprojs, vocab_size)
        else:
            self.output = torch.nn.Linear(hidden_size, vocab_size)

        self.att_list = build_attention_list(
            eprojs=eprojs, dunits=hidden_size, **att_conf
        )

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for i in range(1, self.dlayers):
                z_list[i], c_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]),
                    (z_prev[i], c_prev[i]),
                )
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for i in range(1, self.dlayers):
                z_list[i] = self.decoder[i](
                    self.dropout_dec[i - 1](z_list[i - 1]), z_prev[i]
                )
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_in_pad, ys_in_lens, beam_width=0, beam_length=5, th_beta=0.0, strm_idx=0):
        if beam_width == 0:
            # to support mutiple encoder asr mode, in single encoder mode,
            # convert torch.Tensor to List of torch.Tensor
            if self.num_encs == 1:
                hs_pad = [hs_pad]
                hlens = [hlens]

            # attention index for the attention module
            # in SPA (speaker parallel attention),
            # att_idx is used to select attention module. In other cases, it is 0.
            att_idx = min(strm_idx, len(self.att_list) - 1)

            # hlens should be list of list of integer
            hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

            # get dim, length info
            olength = ys_in_pad.size(1)

            # initialization
            c_list = [self.zero_state(hs_pad[0])]
            z_list = [self.zero_state(hs_pad[0])]
            for _ in range(1, self.dlayers):
                c_list.append(self.zero_state(hs_pad[0]))
                z_list.append(self.zero_state(hs_pad[0]))
            z_all = []
            if self.num_encs == 1:
                att_w = None
                self.att_list[att_idx].reset()  # reset pre-computation of h
            else:
                att_w_list = [None] * (self.num_encs + 1)  # atts + han
                att_c_list = [None] * self.num_encs  # atts
                for idx in range(self.num_encs + 1):
                    # reset pre-computation of h in atts and han
                    self.att_list[idx].reset()

            # pre-computation of embedding
            eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

            # loop for an output sequence
            for i in range(olength):
                if self.num_encs == 1:
                    att_c, att_w = self.att_list[att_idx](
                        hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w
                    )
                else:
                    for idx in range(self.num_encs):
                        att_c_list[idx], att_w_list[idx] = self.att_list[idx](
                            hs_pad[idx],
                            hlens[idx],
                            self.dropout_dec[0](z_list[0]),
                            att_w_list[idx],
                        )
                    hs_pad_han = torch.stack(att_c_list, dim=1)
                    hlens_han = [self.num_encs] * len(ys_in_pad)
                    att_c, att_w_list[self.num_encs] = self.att_list[self.num_encs](
                        hs_pad_han,
                        hlens_han,
                        self.dropout_dec[0](z_list[0]),
                        att_w_list[self.num_encs],
                    )
                if i > 0 and random.random() < self.sampling_probability:
                    z_out = self.output(z_all[-1])
                    z_out = np.argmax(z_out.detach().cpu(), axis=1)
                    z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                    ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
                else:
                    # utt x (zdim + hdim)
                    ey = torch.cat((eys[:, i, :], att_c), dim=1)
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
                if self.context_residual:
                    z_all.append(
                        torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
                    )  # utt x (zdim + hdim)
                else:
                    z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        else:
            # attention index for the attention module
            # in SPA (speaker parallel attention),
            # att_idx is used to select attention module. In other cases, it is 0.
            hs_pad = [hs_pad]
            hlens = [hlens]

            att_idx = min(strm_idx, len(self.att_list) - 1)

            # hlens should be list of list of integer
            hlens = [list(map(int, hlens[idx])) for idx in range(self.num_encs)]

            # get dim, length info
            olength = ys_in_pad.size(1)

            # initialization
            c_list = [self.zero_state(hs_pad[0])]
            z_list = [self.zero_state(hs_pad[0])]
            for _ in range(1, self.dlayers):
                c_list.append(self.zero_state(hs_pad[0]))
                z_list.append(self.zero_state(hs_pad[0]))
            z_all = []

            att_w = None
            self.att_list[att_idx].reset()  # reset pre-computation of h

            # pre-computation of embedding
            eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

            # loop for an output sequence
            current_beam_length = np.zeros(hs_pad[0].size(0), int)  # batch-wise beam length status
            beam_processed_idx = [[None]]*hs_pad[0].size(0)     # (B, ?, 2(tuple)) Index ranges where beam search have been processed
            # batch_range dict(B, ?, 2(tuple)) Beam-batchfied beam batch range
            # best_batch_idx dict(B, ?, 1(int)) Batch index of the best beam path
            one_bests = {"batch_range": [[None]]*hs_pad[0].size(0), "best_batch_idx": [[None]]*hs_pad[0].size(0)}
            score = torch.zeros(hs_pad[0].size(0))  # beam search score
            ended_hyps = torch.zeros(hs_pad[0].size(0))  # hyps met <eos>
            skip = []
            for i in range(olength):

                att_c, att_w = self.att_list[att_idx](
                    hs_pad[0], hlens[0], self.dropout_dec[0](z_list[0]), att_w, beam=True
                )
                ####### Beam Search #########
                # if i > 0:
                #     z_out = self.output(z_all[-1])
                #
                #     z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, skip, beam_processed_idx = make_beamset(
                #         z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, th_beta, skip, i, beam_processed_idx, beam_width)
                #     z_out = to_device(self, z_out)
                #     th_beta = to_device(self, th_beta)
                #     for z in range(len(z_list)):
                #         z_list[z] = to_device(self, z_list[z])
                #
                #     z_out = self.dropout_emb(self.embed(z_out))
                #     ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
                # else:
                #     # utt x (zdim + hdim)
                #     ey = torch.cat((eys[:, i, :], att_c), dim=1)
                #
                # z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
                #
                # z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)
                
                ########### Partial Beam Search ###################
                if i > 0:
                    z_out = self.output(z_all[-1])
                    z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, skip, beam_processed_idx, current_beam_length, one_bests = make_partial_beamset(
                        z_out, score, z_list, c_list, ys_in_pad, hs_pad, hlens, att_w, att_c, olength, th_beta, skip, i,
                        beam_processed_idx, current_beam_length, beam_width, beam_length, one_bests)
                    z_out = to_device(self, z_out)
                    th_beta = to_device(self, th_beta)
                    for z in range(len(z_list)):
                        z_list[z] = to_device(self, z_list[z])

                    z_out = self.dropout_emb(self.embed(z_out))
                    ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
                else:
                    # utt x (zdim + hdim)
                    ey = torch.cat((eys[:, i, :], att_c), dim=1)

                z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)

                if i == olength - 1 and True in current_beam_length.astype(bool):
                    beam_processing_idx = current_beam_length.nonzero()[0]
                    for bpidx, f_l in enumerate(skip):
                        f, l = f_l
                        f = f - bpidx * beam_width
                        l = l - bpidx * beam_width
                        z_out, score, one_best = pick_one_best(z_out, (f, l), score)
                        one_bests = update_one_bests(one_best, (f, l), beam_processing_idx[bpidx], one_bests)

                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        if beam_width == 0:
            z_all = torch.stack(z_all, dim=1)
            z_all = self.output(z_all)
            z_all.masked_fill_(
                make_pad_mask(ys_in_lens, z_all, 1),
                0,
            )
            return z_all, None

        else:
            ### Beam Search ###
            # score normalization concerned with eos
            # for batch_idx, beam_processed in enumerate(beam_processed_idx):
            #     if beam_processed[0] is None:
            #         continue
            #     # score normalization
            #     score[batch_idx:batch_idx + beam_width] = score[batch_idx:batch_idx + beam_width] / (olength - beam_processed[0])
            #     # filter z_all
            #     z_all[beam_processed[0]:], score, one_best = pick_one_best(z_all[beam_processed[0]:], (batch_idx, batch_idx + beam_width), score)
            # z_all = torch.stack(z_all, dim=1)
            # z_all = self.output(z_all)
            # z_all.masked_fill_(
            #     make_pad_mask(ys_in_lens, z_all, 1),
            #     0,
            # )

            ### Partial beam search ###
            # filter z_all to one best path
            # z_all: (T, Dynamic_beam_batch, dunits)
            for batch_idx, beam_processed_f_l in enumerate(beam_processed_idx):
                if beam_processed_f_l[0] is None:
                    continue
                one_best = one_bests["best_batch_idx"][batch_idx]
                for nth_processed_idx, f_l in enumerate(beam_processed_f_l):
                    best_batch_idx = one_best[nth_processed_idx]
                    f, l = f_l
                    temp = z_all[f:l].copy()    # (beam_length, dynamic_beam_batch, dunits)
                    for temp_idx, z in enumerate(temp):
                        z = torch.cat((z[:batch_idx], z[batch_idx + best_batch_idx].unsqueeze(0), z[batch_idx + beam_width:]), 0)
                        temp[temp_idx] = z
                    z_all[f:l] = temp

            z_all = torch.stack(z_all, dim=1)
            z_all = self.output(z_all)
            z_all.masked_fill_(
                make_pad_mask(ys_in_lens, z_all, 1),
                0,
            )
            return z_all, beam_processed_idx

    def init_state(self, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        c_list = [self.zero_state(x[0].unsqueeze(0))]
        z_list = [self.zero_state(x[0].unsqueeze(0))]
        for _ in range(1, self.dlayers):
            c_list.append(self.zero_state(x[0].unsqueeze(0)))
            z_list.append(self.zero_state(x[0].unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att_list) - 1)
        if self.num_encs == 1:
            a = None
            self.att_list[att_idx].reset()  # reset pre-computation of h
        else:
            a = [None] * (self.num_encs + 1)  # atts + han
            for idx in range(self.num_encs + 1):
                # reset pre-computation of h in atts and han
                self.att_list[idx].reset()
        return dict(
            c_prev=c_list[:],
            z_prev=z_list[:],
            a_prev=a,
            workspace=(att_idx, z_list, c_list),
        )

    def score(self, yseq, state, x):
        # to support mutiple encoder asr mode, in single encoder mode,
        # convert torch.Tensor to List of torch.Tensor
        if self.num_encs == 1:
            x = [x]

        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        if self.num_encs == 1:
            att_c, att_w = self.att_list[att_idx](
                x[0].unsqueeze(0),
                [x[0].size(0)],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"],
            )
        else:
            att_w = [None] * (self.num_encs + 1)  # atts + han
            att_c_list = [None] * self.num_encs  # atts
            for idx in range(self.num_encs):
                att_c_list[idx], att_w[idx] = self.att_list[idx](
                    x[idx].unsqueeze(0),
                    [x[idx].size(0)],
                    self.dropout_dec[0](state["z_prev"][0]),
                    state["a_prev"][idx],
                )
            h_han = torch.stack(att_c_list, dim=1)
            att_c, att_w[self.num_encs] = self.att_list[self.num_encs](
                h_han,
                [self.num_encs],
                self.dropout_dec[0](state["z_prev"][0]),
                state["a_prev"][self.num_encs],
            )
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(
            ey, z_list, c_list, state["z_prev"], state["c_prev"]
        )
        if self.context_residual:
            logits = self.output(
                torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1)
            )
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return (
            logp,
            dict(
                c_prev=c_list[:],
                z_prev=z_list[:],
                a_prev=att_w,
                workspace=(att_idx, z_list, c_list),
            ),
        )
