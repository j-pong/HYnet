# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import (
    AddTargetDataset,
    BinarizedAudioDataset,
    ConcatDataset,
    Dictionary,
    FileAudioDataset,
    FairseqDataset,
    encoders,
)
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)
from fairseq.data import Dictionary, data_utils, iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@dataclass
class InferredW2vConfig:
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")


@dataclass
class UDAFinetuningConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to labeled data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default=None,
        metadata={
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)",
        },
    )

    tpu: bool = II("common.tpu")


@register_task("uda_finetuning", dataclass=UDAFinetuningConfig)
class UDAFinetuningTask(FairseqTask):
    """ """

    cfg: UDAFinetuningConfig

    def __init__(
        self,
        cfg: UDAFinetuningConfig,
    ):
        super().__init__(cfg)
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"

        self.cfg = cfg
        self.blank_symbol = "<s>"

        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    @classmethod
    def setup_task(cls, cfg: UDAFinetuningConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (UDAFinetuningConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            return Dictionary.load(dict_path)
        return None

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            assert (
                cfg.inferred_w2v_config is not None
            ), "inferred_w2v_config must be set"
            return OmegaConf.to_container(
                cfg.inferred_w2v_config, resolve=True, enum_to_str=True
            )
        else:
            return {}

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """

        # if len(split.split(",")) > 1:
        #     l_split = split.split(",")[0]
        #     ul_split = split.split(",")[1]
        #
        #     if l_split not in self.datasets:
        #         raise KeyError("Dataset not loaded: " + l_split)
        #     elif ul_split not in self.datasets:
        #         raise KeyError("Dataset not loaded: " + ul_split)
        #     if not isinstance(self.l_split, FairseqDataset) or not isinstance(self.ul_split, FairseqDataset):
        #         raise TypeError("Datasets are expected to be of type FairseqDataset")
        #
        #     return [self.l_split, self.ul_split]

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError("Datasets are expected to be of type FairseqDataset")
        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        if isinstance(dataset, ConcatDataset):
            semi = True
        else:
            semi = False
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        if not semi:
            dataset.set_epoch(epoch)

            # get indices ordered by example size
            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
    
            # filter examples that are too large
            if max_positions is not None:
                indices = self.filter_indices_by_size(
                    indices, dataset, max_positions, ignore_invalid_inputs
                )
    
            # create mini-batches with given size constraints
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
    
            # return a reusable, sharded iterator
            epoch_iter = iterators.EpochBatchIterator(
                dataset=dataset,
                collate_fn=dataset.collater,
                batch_sampler=batch_sampler,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                buffer_size=data_buffer_size,
            )
    
            if can_reuse_epoch_itr:
                self.dataset_to_epoch_iter[dataset] = epoch_iter

        elif semi:
            dataset.set_epoch(epoch)

            mult_ratio = [1, 0.1]
            assert len(dataset.datasets) == 2

            # initialize the dataset with the correct starting epoch
            dataset.set_epoch(epoch)

            batch_samplers = dataset.get_batch_samplers(
                mult_ratio, required_batch_size_multiple, seed, max_tokens=[max_tokens]*2,
                max_sentences=[max_sentences]*2, max_positions=[max_positions]*2
            )

            # return a reusable, sharded iterator
            epoch_iter = iterators.GroupedEpochBatchIterator(
                dataset=dataset,
                collate_fn=dataset.collater,
                batch_samplers=batch_samplers,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                buffer_size=data_buffer_size,
            )

            if can_reuse_epoch_itr:
                self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        split_temp = split.split(",")
        if len(split_temp) > 1:
            semi = True
            l_split = split_temp[0]
            ul_split = split_temp[1]
        else:
            semi = False
            split = split_temp[0]
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        if not semi:
            manifest_path = os.path.join(data_path, "{}.tsv".format(split))

            self.datasets[split] = FileAudioDataset(
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )

            if self.cfg.tpu and task_cfg["mask_channel_prob"] == 0.0:
                logger.info(
                    "Pretraining on TPUs may suffer convergence "
                    "issues when training with `mask_channel_prob` value of "
                    "0. You may want to set this to a low value close to 0."
                )

            if task_cfg.labels:
                label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
                skipped_indices = getattr(self.datasets[split], "skipped_indices", set())
                with open(label_path, "r") as f:
                    labels = [line for i, line in enumerate(f) if i not in skipped_indices]

                assert len(labels) == len(self.datasets[split]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match"
                )

                process_label = LabelEncoder(self.target_dictionary)

                self.datasets[split] = AddTargetDataset(
                    self.datasets[split],
                    labels,
                    pad=self.target_dictionary.pad(),
                    eos=self.target_dictionary.eos(),
                    batch_targets=True,
                    process_label=process_label,
                    add_to_input=task_cfg.get("autoregressive", False),
                )

        elif semi:
            l_manifest_path = os.path.join(data_path, "{}.tsv".format(l_split))
            ul_manifest_path = os.path.join(data_path, "{}.tsv".format(ul_split))

            self.l_split = FileAudioDataset(
                manifest_path=l_manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )
            self.ul_split = FileAudioDataset(
                manifest_path=ul_manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )

            if self.cfg.tpu and task_cfg["mask_channel_prob"] == 0.0:
                logger.info(
                    "Pretraining on TPUs may suffer convergence "
                    "issues when training with `mask_channel_prob` value of "
                    "0. You may want to set this to a low value close to 0."
                )

            if task_cfg.labels:
                label_path = os.path.join(data_path, f"{l_split}.{task_cfg.labels}")
                skipped_indices = getattr(self.l_split, "skipped_indices", set())
                with open(label_path, "r") as f:
                    labels = [line for i, line in enumerate(f) if i not in skipped_indices]

                assert len(labels) == len(self.l_split), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.l_split)}) do not match"
                )

                process_label = LabelEncoder(self.target_dictionary)

                self.l_split = AddTargetDataset(
                    self.l_split,
                    labels,
                    pad=self.target_dictionary.pad(),
                    eos=self.target_dictionary.eos(),
                    batch_targets=True,
                    process_label=process_label,
                    add_to_input=task_cfg.get("autoregressive", False),
                )

            mdsets = [
                ModalityDatasetItem(
                    "labeled",
                    self.l_split,
                    None,
                    None,
                    None,
                ),
                ModalityDatasetItem(
                    "unlabeled",
                    self.ul_split,
                    None,
                    None,
                    None,
                ),
            ]
            self.datasets[split] = MultiModalityDataset(mdsets)

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            if "w2v_args" in actualized_cfg:
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.cfg.eval_wer_post_process,
                escape_unk=True,
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_chars > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
        if num_words > 0:
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )
