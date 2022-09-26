from dataclasses import  is_dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from fairseq import distributed_utils, utils
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

from examples.speech_recognition.new.infer import config_path, InferConfig, main, InferenceProcessor
from examples.speech_recognition.new.infer import reset_logging, logger

# from force_aligner import Segmentor as ForcedSegmentor
from aligner import Segmentor

class InferenceProcessorSeg(InferenceProcessor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Task inferene_step warpper
        self.manifest = {}

        self.segmentor = Segmentor(self.models[0], self.tgt_dict)
        # self.segmentor = ForcedSegmentor(self.models[0], self.tgt_dict)
    
    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.gen_timer.start()
        sample_id = sample["id"][0].item()
        self.manifest[sample_id] = {}

        results = self.segmentor.segment(sample)

        # ltr = results["ltr"].strip()
        # self.manifest[sample_id]["ltr"] = ltr

        dur = results['dur']
        dur = " ".join([f"{d[0]} {d[1]} |" for d in dur])
        self.manifest[sample_id]["dur"] = dur


def main(cfg: InferConfig) -> float:
    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 4000000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")

    logger.info(cfg.common_eval.path)

    with InferenceProcessorSeg(cfg) as processor:
        for sample in processor:
            processor.process_sample(sample)

        # processor.log_generation_time()

        if cfg.decoding.results_path is not None:
            processor.merge_shards()

        if distributed_utils.is_master(cfg.distributed_training):
            # ltr_out = open(f"train_pl.ltr", 'w') 
            # wrd_out = open(f"train_pl.wrd", 'w')
            dur_out = open(f"train.dur", 'w')
            for sample_id in sorted(processor.manifest.keys()):
                # print(processor.manifest[sample_id]["ltr"], file=ltr_out)
                # print(
                    # processor.manifest[sample_id]["ltr"].replace(" ", "").replace("|", " "),
                    # file=wrd_out,
                # )
                print(processor.manifest[sample_id]["dur"], file=dur_out)

        return 0.0

@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    utils.import_user_module(cfg.common)

    # logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    return wer


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
