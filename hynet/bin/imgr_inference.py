#!/usr/bin/env python3
import argparse
import dataclasses
from dataclasses import is_dataclass
import logging
import time
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Sequence

import numpy as np

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.schedulers.abs_scheduler import AbsScheduler
from espnet2.schedulers.abs_scheduler import AbsValEpochStepScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochStepScheduler
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.reporter import Reporter
from espnet2.train.reporter import SubReporter
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption

from espnet2.train.trainer import Trainer, ReduceOp, TrainerOptions, GradScaler, SummaryWriter, Path

from hynet.tasks.imgr import ImgrTask

from tqdm import tqdm

class ImgrInference(Trainer):
    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        reporter: Reporter,
        scaler: Optional[GradScaler],
        output_dir: Path,
        max_epoch: int,
        seed: int,
        patience: Optional[int],
        keep_nbest_models: int,
        early_stopping_criterion: Sequence[str],
        best_model_criterion: Sequence[Sequence[str]],
        val_scheduler_criterion: Sequence[str],
        trainer_options,
        distributed_option: DistributedOption,
        find_unused_parameters: bool = False,
    ) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)

        start_epoch = reporter.get_epoch() + 1

        dp_model = model

        reporter.set_epoch(start_epoch)

        if not distributed_option.distributed or distributed_option.dist_rank == 0:
            with reporter.observe("attns") as sub_reporter:
                cls.plot_attention(
                    model=dp_model,
                    output_dir=output_dir / "brew_ws",
                    iterator=valid_iter_factory.build_iter(start_epoch),
                    reporter=sub_reporter,
                    options=trainer_options
                )
    
            logging.info(reporter.log_message())

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: torch.nn.Module,
        output_dir: Path,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
        flag: bool = False,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        model.eval()
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (ids, batch) in tqdm(iterator):
            assert isinstance(batch, dict), type(batch)

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            _, stats, weight = model(**batch)

            
            img_list = stats['aux']
            if flag:
                cls.plot_(img_list, output_dir, ids, reporter.get_epoch())
                flag = False
            del stats['aux']

            reporter.register(stats, weight)
            reporter.next()

    @classmethod
    def plot_(
        cls, 
        img_list: List, 
        output_dir: Path, 
        ids: int, 
        epoch: int,
        max_plot=3
    ) -> None:
        import matplotlib.pyplot as plt

        col_max = len(img_list)
        row_max = len(img_list[0])
        
        for k in range(max_plot):
            plt.clf()

            for i in range(col_max):
                for j, img in enumerate(img_list[i]):
                    img_ = img[k].detach().cpu().numpy()

                    plt.subplot(col_max, row_max, i * row_max + (j + 1))

                    plt.imshow(img_, cmap='seismic')
                    plt.colorbar()

                    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

            if output_dir is not None:
                p = output_dir / f"valid_{ids[k]}" / f"{ids[k]}_{epoch}ep.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(p)


class ImgrTaskInference(ImgrTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = ImgrInference

def get_parser():
    parser = ImgrTaskInference.get_parser()
    return parser


def main(cmd=None):
    ImgrTaskInference.main(cmd=cmd)


if __name__ == "__main__":
    main()
