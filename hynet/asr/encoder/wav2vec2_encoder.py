# Copyright 2021 Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
from filelock import FileLock
import logging
import os
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder

import fairseq
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model


class FairSeqWav2VecCtc(AbsEncoder):
    """FairSeq Wav2Vec2 encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        w2v_url: url to Wav2Vec2.0 pretrained model
        w2v_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int,
        w2v_url: str,
        w2v_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        load_proj: bool = True,
        layerdrop: float = 0.0,
        activation_dropout: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()

        if w2v_url != "":
            try:
                import fairseq
                from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
            except Exception as e:
                print("Error: FairSeq is not properly installed.")
                print(
                    "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
                )
                raise e

        self.w2v_model_path = download_w2v(w2v_url, w2v_dir_path)

        self._output_size = output_size

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.w2v_model_path],
            arg_overrides={"data": w2v_dir_path},
        )
        model = models[0]

        if isinstance(model, Wav2VecCtc):
            self.encoders = model

            self.pretrained_params = copy.deepcopy(model.state_dict())

            if not load_proj or model.w2v_encoder.proj.out_features != output_size:
                # TODO(xkc09): try LSTM
                self.output_layer = torch.nn.Sequential(
                    torch.nn.Linear(model.w2v_encoder.proj.in_features, output_size),
                )
            else:
                self.output_layer = None

            if not load_proj:
                self.encoders.w2v_encoder.proj = None

        elif isinstance(model, Wav2Vec2Model):
            self.encoders = model
            in_features = model.final_proj.in_features
            self.encoders.final_proj = None
            self.encoders.encoder.layer_drop = layerdrop
            for layer in self.encoders.encoder.layers:
                layer.dropout2 = torch.nn.Dropout(layerdrop, inplace=False)

            self.pretrained_params = copy.deepcopy(model.state_dict())

            # TODO(xkc09): try LSTM
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(in_features, output_size),
            )

        else:
            raise Exception(
                "Error: pretrained models should be: "
                "'Wav2VecCTC' or 'Wav2Vec2Model' class"
            )
        
        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))
        self.load_proj = load_proj

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairSeqWav2Vec2 Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates
        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning wav2vec parameters!")

        with torch.no_grad() if not ft else contextlib.nullcontext():
            if isinstance(self.encoders, Wav2VecCtc):
                features_only = False if self.load_proj else True
                self.encoders.w2v_encoder.apply_mask = self.training
                enc_outputs = self.encoders.w2v_encoder(
                    xs_pad,
                    masks,
                    tbc=False,
                    features_only=features_only,
                )
                xs_pad = enc_outputs["encoder_out"]  # (B,T,C),
                masks = enc_outputs["padding_mask"]  # (B, T)

            elif isinstance(self.encoders, Wav2Vec2Model):
                enc_outputs = self.encoders(
                    xs_pad,
                    masks,
                    mask=self.training,
                    features_only=True,
                )
                xs_pad = enc_outputs["x"]  # (B,T,C),
                masks = enc_outputs["padding_mask"]  # (B, T)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)


        olens = (~masks).sum(dim=1)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Wav2Vec model parameters reloaded!")


def download_w2v(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"Wav2Vec model downloaded {model_path}")
        else:
            logging.info(f"Wav2Vec model {model_path} already exists.")

    return model_path


# def FairSeqPreprocess(feats, curr_sample_rate):
#     if feats.dim() == 2:
#         feats = feats.mean(-1)
#
#     if curr_sample_rate != self.sample_rate:
#         raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")
#
#     assert feats.dim() == 1, feats.dim()
#
#     if self.normalize:
#         with torch.no_grad():
#             feats = F.layer_norm(feats, feats.shape)
#     return feats
