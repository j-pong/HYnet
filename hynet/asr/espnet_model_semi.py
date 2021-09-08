from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
            self,
            vocab_size: int,
            token_list: Union[Tuple[str, ...], List[str]],
            frontend: Optional[AbsFrontend],
            specaug: Optional[AbsSpecAug],
            normalize: Optional[AbsNormalize],
            encoder: AbsEncoder,
            decoder: AbsDecoder,
            ctc: CTC,
            rnnt_decoder: None,
            ctc_weight: float = 0.5,
            ignore_id: int = -1,
            lsm_weight: float = 0.0,
            length_normalized_loss: bool = False,
            report_cer: bool = True,
            report_wer: bool = True,
            sym_space: str = "<space>",
            sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.adddiontal_utt_mvn = None
        self.encoder = encoder
        self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.register_buffer('th_beta', torch.tensor(0.0, dtype=torch.float))

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

    def forward(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            mode=None,
            iepoch=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
                speech.shape[0]
                == speech_lengths.shape[0]
                == text.shape[0]
                == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # for bootstrapping beta
        self.iepoch = iepoch

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            loss_att, acc_att, cer_att, wer_att, err_pred = None, None, None, None, None
        else:
            loss_att, acc_att, cer_att, wer_att, err_pred = self._calc_att_loss(
                encoder_out, encoder_out_lens, text, text_lengths, mode
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            loss_ctc, cer_ctc = None, None
        else:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
            err_pred=err_pred,
            th_beta=float(self.th_beta),
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation for spectrogram
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
                if self.adddiontal_utt_mvn is not None:
                    feats, feats_lengths = self.adddiontal_utt_mvn(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
            mode=None,
    ):
        if mode is None:
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_in_lens = ys_pad_lens + 1
        else:
            from espnet.nets.pytorch_backend.nets_utils import pad_list

            _sos = ys_pad.new([self.sos])
            _ignore = ys_pad.new([self.ignore_id])

            ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

            ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
            ys_out = [torch.cat([y, _ignore], dim=0) for y in ys]

            ys_in_pad = pad_list(ys_in, self.eos)
            ys_out_pad = pad_list(ys_out, self.ignore_id)

            ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        err_pred = 0.0
        if mode is None:
            decoder_out, _, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )
            
        elif mode == 'pseudo':
            ### Pseudo label baseline ###
            decoder_out, _, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, mode=mode
            )

            # Calculate pred dist
            pred_dist = torch.softmax(decoder_out, dim=-1)
            bsz, tsz = ys_out_pad.size()

            # Select target label probability from pred_dist
            pred_prob = pred_dist.view(bsz * tsz, -1)[torch.arange(bsz * tsz),
                                                      ys_out_pad.view(bsz * tsz)]
            pred_prob = pred_prob.view(bsz, tsz)
            
            # check errors
            repl_mask = pred_prob < self.th_beta
            repl_mask = [rm[:l] for rm, l in zip(repl_mask, ys_pad_lens)]
            err_pred = float(torch.tensor([rm.float().mean() for rm in repl_mask]).mean())

        elif mode == 'gradient_masking':
            ### Gradient Masking ###
            with torch.no_grad():
                decoder_out, _, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, mode=mode
                )
                # Calculate pred dist
                pred_dist = torch.softmax(decoder_out, dim=-1)

                bsz, tsz = ys_out_pad.size()

                # Select target label probability from pred_dist
                pred_prob = pred_dist.view(bsz * tsz, -1)[torch.arange(bsz * tsz),
                                                          ys_out_pad.view(bsz * tsz)]
                pred_prob = pred_prob.view(bsz, tsz)

                # Wrong labels position with confidence filtering
                repl_mask = pred_prob < self.th_beta
                repl_mask = [rm[:l] for rm, l in zip(repl_mask, ys_pad_lens)]
                err_pred = float(torch.tensor([rm.float().mean() for rm in repl_mask]).mean())

            _sos = ys_pad.new([self.sos])
            _ignore = ys_pad.new([self.ignore_id])

            ys_in = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]
            ys_out = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]

            for rm, y in zip(repl_mask, ys_in):
                y[rm] = 1
            for rm, y in zip(repl_mask, ys_out):
                y[rm] = self.ignore_id

            ys_in = [torch.cat([_sos, y], dim=0) for y in ys_in]
            ys_out = [torch.cat([y, _ignore], dim=0) for y in ys_out]

            ys_in_pad = pad_list(ys_in, self.eos)
            ys_out_pad = pad_list(ys_out, self.ignore_id)

            decoder_out, _, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )

        elif mode == 'recursive_gradient_masking':
            ### Recursive Gradient Masking ###
            with torch.no_grad():
                decoder_out, _, repl_masks = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, ys_out_pad=ys_out_pad, mode=mode,
                    th_beta=self.th_beta, ignore_id=self.ignore_id
                )

                # Wrong labels position with confidence filtering
                repl_mask = repl_masks
                repl_mask = [rm[:l] for rm, l in zip(repl_mask, ys_pad_lens)]
                err_pred = float(torch.tensor([rm.float().mean() for rm in repl_mask]).mean())

            _sos = ys_pad.new([self.sos])
            _ignore = ys_pad.new([self.ignore_id])

            ys_in = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]
            ys_out = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]

            for rm, y in zip(repl_mask, ys_in):
                y[rm] = 1
            for rm, y in zip(repl_mask, ys_out):
                y[rm] = self.ignore_id

            ys_in = [torch.cat([_sos, y], dim=0) for y in ys_in]
            ys_out = [torch.cat([y, _ignore], dim=0) for y in ys_out]

            ys_in_pad = pad_list(ys_in, self.eos)
            ys_out_pad = pad_list(ys_out, self.ignore_id)

            decoder_out, _, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
            )

        elif mode == 'bootstrap':
            ### Bootstrapping ###
            with torch.no_grad():
                decoder_out, _, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, ys_out_pad=ys_out_pad, mode=None,
                    th_beta=self.th_beta, ignore_id=self.ignore_id
                )

                decoder_out_noisy, _, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens, ys_out_pad=ys_out_pad, mode=mode,
                    th_beta=self.th_beta, ignore_id=self.ignore_id
                )

                # self-prediction label for entropy minimization
                ys_out_refurbished_pad = torch.argmax(decoder_out_noisy.clone(), -1)
                ys_pad_refurbished_lens = []
                # TODO: maybe find eos?
                for batch_idx, y in enumerate(ys_out_refurbished_pad):
                    if self.eos in y:
                        ys_out_refurbished_pad[batch_idx, torch.where(y == self.eos)[0][0]:] = self.ignore_id
                        ys_pad_refurbished_lens.append(torch.where(y == self.ignore_id)[0][0])
                    elif 0 in y:
                        ys_out_refurbished_pad[batch_idx, torch.where(y == 0)[0][0]:] = self.ignore_id
                        ys_pad_refurbished_lens.append(torch.where(y == self.ignore_id)[0][0])
                    else:
                        ys_pad_refurbished_lens.append(len(y))
                ys_out_refurbished_pad = ys_out_refurbished_pad.to(ys_pad.device)
                ys_pad_refurbished_lens = torch.tensor(ys_pad_refurbished_lens).detach()
                ys_in_refurbished_lengths = ys_pad_refurbished_lens.clone().to(ys_pad.device) + 1
                ys_refurbished_pad = ys_out_refurbished_pad.clone().detach()

                # Calculate pred dist
                pred_dist = torch.softmax(decoder_out, dim=-1)
                pred_dist_noisy = torch.softmax(decoder_out_noisy, dim=-1)

                bsz, tsz = ys_out_pad.size()

                # Select target label probability from pred_dist
                pred_prob = pred_dist.view(bsz * tsz, -1)[torch.arange(bsz * tsz),
                                                          ys_out_pad.view(bsz * tsz)]
                pred_prob = pred_prob.view(bsz, tsz)
                pred_prob_noisy = pred_dist_noisy.view(bsz * tsz, -1)[torch.arange(bsz * tsz),
                                                          ys_out_refurbished_pad.view(bsz * tsz)]
                pred_prob_noisy = pred_prob_noisy.view(bsz, tsz)

                # Wrong labels position with confidence filtering
                repl_mask = pred_prob < self.th_beta
                repl_mask_noisy = pred_prob_noisy < self.th_beta
                repl_mask = [rm[:l] for rm, l in zip(repl_mask, ys_pad_lens)]
                repl_mask_noisy = [rm[:l] for rm, l in zip(repl_mask_noisy, ys_pad_refurbished_lens)]

                # pseudo ref vs. hyp TER
                import editdistance
                err_pred_list = list()
                for ys_out_pseudo, ys_out_noisy, pred_len, pred_noisy_len in zip(ys_out_pad, ys_out_refurbished_pad, ys_pad_lens, ys_pad_refurbished_lens):
                    err_pred_list.append(editdistance.eval(ys_out_pseudo[:pred_len].tolist(), ys_out_noisy[:pred_noisy_len].tolist()))
                err_pred = torch.tensor(float(sum(err_pred_list) / sum(ys_pad_lens))).to(ys_out_refurbished_pad.device)

            _sos = ys_pad.new([self.sos])
            _ignore = ys_pad.new([self.ignore_id])
            _sos_refurbished = ys_refurbished_pad.new([self.sos])
            _ignore_refurbished = ys_refurbished_pad.new([self.ignore_id])

            ys_in = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]
            ys_out = [y[y != self.ignore_id] for y in ys_pad.clone().detach()]
            ys_refurbished_in = [y[y != self.ignore_id] for y in ys_refurbished_pad.clone().detach()]
            ys_refurbished_out = [y[y != self.ignore_id] for y in ys_refurbished_pad.clone().detach()]

            for rm, y in zip(repl_mask, ys_in):
                y[rm] = 1
            for rm, y in zip(repl_mask, ys_out):
                y[rm] = self.ignore_id
            for rm, y in zip(repl_mask_noisy, ys_refurbished_in):
                y[rm] = 1
            for rm, y in zip(repl_mask_noisy, ys_refurbished_out):
                y[rm] = self.ignore_id

            ys_in = [torch.cat([_sos, y], dim=0) for y in ys_in]
            ys_out = [torch.cat([y, _ignore], dim=0) for y in ys_out]
            ys_refurbished_in = [torch.cat([_sos_refurbished, y], dim=0) for y in ys_refurbished_in]
            ys_refurbished_out = [torch.cat([y, _ignore_refurbished], dim=0) for y in ys_refurbished_out]

            ys_in_pad = pad_list(ys_in, self.eos)
            ys_out_pad = pad_list(ys_out, self.ignore_id)
            ys_in_refurbished_pad = pad_list(ys_refurbished_in, self.eos).to(ys_out_refurbished_pad.device)
            ys_out_refurbished_pad = pad_list(ys_refurbished_out, self.ignore_id).to(ys_in_refurbished_pad.device)

            decoder_out, _, _ = self.decoder(
                    encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens)
            decoder_out_noisy, _, _ = self.decoder(
                encoder_out, encoder_out_lens, ys_in_refurbished_pad, ys_in_refurbished_lengths)

        else:
            raise AttributeError("{} mode is not supported!".format(mode))

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        if mode is not "bootstrap":
            acc_att = th_accuracy(
                decoder_out.view(-1, self.vocab_size),
                ys_out_pad,
                ignore_label=self.ignore_id,
            )
            loss_att_refurbished = torch.tensor(0.0).to(decoder_out.device)
        else:
            acc_att = torch.tensor(0.0).to(decoder_out.device)
            loss_att_refurbished = self.criterion_att(decoder_out_noisy, ys_out_refurbished_pad)

        # mixup ratio
        # according to bootstrap paper, beta=0.8 worked well for hard bootstrapping
        if self.iepoch is not None:
            beta = 0.8
        else:
            beta = 1.0
        loss_att = beta * loss_att + (1 - beta) * loss_att_refurbished

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None or mode == "pseudo":
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, err_pred

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def _calc_confidence(
            self,
            ys_in_pad,
            ys_out_pad,
            ys_in_lens,
            encoder_out,
            encoder_out_lens,
            decoder_out,
    ):
        bsz, tsz = ys_in_pad.size()
        ys_in_pad = ys_in_pad.view(bsz * tsz)

        # Simulating
        corrupt_pos = torch.rand_like(ys_in_pad.float()) > 0.9
        corrupt_ys_in_pad = torch.randint_like(ys_in_pad.float(), low=1, high=5000)
        ys_in_pad[corrupt_pos] = corrupt_ys_in_pad[corrupt_pos].long()
        ys_in_pad = ys_in_pad.view(bsz, tsz)

        decoder_corrupt_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # Caculating predictive distribution with corrupted labels
        pred_dist = torch.softmax(decoder_out, dim=-1)
        pred_corrupt_dist = torch.softmax(decoder_corrupt_out, dim=-1)

        # Confidence
        def select_mean_var(x, indces):
            bsz, tsz = indces.size()
            select_mean = x.view(bsz * tsz, -1)[torch.arange(bsz * tsz),
                                                indces.view(bsz * tsz)]
            # select_var = torch.square(x.view(bsz * tsz, -1) - select_mean.unsqueeze(-1)).sum(-1)
            select_var = x.var(-1)
            return select_mean.view(bsz, tsz), select_var.view(bsz, tsz)

        select_mean, select_var = select_mean_var(pred_dist, ys_out_pad)
        insertion = torch.ones_like(ys_out_pad) * 4999
        corrupt_ys_out_pad = torch.cat([corrupt_ys_in_pad.view(bsz, tsz)[:, 1:], insertion[:, 0:1]], dim=-1).view(
            bsz * tsz)
        ys_out_pad = ys_out_pad.view(bsz * tsz)
        ys_out_pad[corrupt_pos] = corrupt_ys_in_pad[corrupt_pos].long()
        ys_out_pad = ys_out_pad.view(bsz, tsz)
        select_mean_corrupt, select_var_corrupt = select_mean_var(pred_corrupt_dist, ys_out_pad)

        # Plot the result
        import matplotlib.pyplot as plt

        plt.clf()

        plt.subplot(3, 1, 1)
        plt.title('Predictive probability')
        plt.xlim(0, ys_out_pad.size(1))
        plt.plot(select_mean[0].detach().cpu().numpy(), 'D-')
        plt.plot(select_mean_corrupt[0].detach().cpu().numpy(), 'o-')
        plt.grid()

        plt.subplot(3, 1, 2)
        plt.title('Corrupted label position')
        plt.xlim(0, ys_out_pad.size(1))
        plt.plot(corrupt_pos.view(bsz, tsz)[0].detach().cpu().numpy(), 'rD-')
        plt.grid()

        plt.subplot(3, 1, 3)
        plt.title('Corrupted label position')
        plt.xlim(0, ys_out_pad.size(1))
        plt.plot(corrupt_pos.view(bsz, tsz)[0].detach().cpu().numpy(), 'rD-')
        plt.grid()

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        p = f"test_{pred_dist.device}.png"
        plt.savefig(p)

        raise ValueError("This process should be stoped at this line")
