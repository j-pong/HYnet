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

class ImgrTrainer(Trainer):
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
        if start_epoch == max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        if distributed_option.distributed:
            dp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=(
                    # Perform multi-Process with multi-GPUs
                    [torch.cuda.current_device()]
                    if distributed_option.ngpu == 1
                    # Perform single-Process with multi-GPUs
                    else None
                ),
                output_device=(
                    torch.cuda.current_device()
                    if distributed_option.ngpu == 1
                    else None
                ),
                find_unused_parameters=find_unused_parameters,
            )
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
                find_unused_parameters=find_unused_parameters,
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            summary_writer = SummaryWriter(str(output_dir / "tensorboard"))
        else:
            summary_writer = None

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{max_epoch}epoch started")
            set_all_random_seed(seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=summary_writer,
                    options=trainer_options,
                )

            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options
                )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(reporter.get_value(*val_scheduler_criterion))
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                reporter.matplotlib_plot(output_dir / "images")
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer)
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # 4. Save/Update the checkpoint
                torch.save(
                    {
                        "model": model.state_dict(),
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / "checkpoint.pth",
                )

                # 5. Save the model and update the link to the best model
                torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

                # Creates a sym link latest.pth -> {iepoch}epoch.pth
                p = output_dir / "latest.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")

                _improved = []
                for _phase, k, _mode in best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            p = output_dir / f"{_phase}.{k}.best.pth"
                            if p.is_symlink() or p.exists():
                                p.unlink()
                            p.symlink_to(f"{iepoch}epoch.pth")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.info("There are no improvements in this epoch")
                else:
                    logging.info(
                        "The best model has been updated: " + ", ".join(_improved)
                    )

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[:keep_nbest_models])
                        for ph, k, m in best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )
                for e in range(1, iepoch):
                    p = output_dir / f"{e}epoch.pth"
                    if p.exists() and e not in nbests:
                        p.unlink()
                        _removed.append(str(p))
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if patience is not None:
                if reporter.check_early_stopping(patience, *early_stopping_criterion):
                    break

        else:
            logging.info(f"The training was finished at {max_epoch} epochs ")


    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: TrainerOptions,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (_, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            _, stats, weight = model(**batch)
            del stats['aux']
            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

class PseudoTrainer(Trainer):
    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        train_pseudo_iter_factory: AbsIterFactory,
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
        if start_epoch == max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        if distributed_option.distributed:
            dp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=(
                    # Perform multi-Process with multi-GPUs
                    [torch.cuda.current_device()]
                    if distributed_option.ngpu == 1
                    # Perform single-Process with multi-GPUs
                    else None
                ),
                output_device=(
                    torch.cuda.current_device()
                    if distributed_option.ngpu == 1
                    else None
                ),
                find_unused_parameters=find_unused_parameters,
            )
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
                find_unused_parameters=find_unused_parameters,
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            summary_writer = SummaryWriter(str(output_dir / "tensorboard"))
        else:
            summary_writer = None

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{max_epoch}epoch started")
            set_all_random_seed(seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train_one_epoch(
                    model=dp_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    pseudo_iterator=train_pseudo_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=summary_writer,
                    options=trainer_options,
                )

            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # att_plot doesn't support distributed
                if plot_attention_iter_factory is not None:
                    with reporter.observe("att_plot") as sub_reporter:
                        cls.plot_attention(
                            model=model,
                            output_dir=output_dir / "att_ws",
                            summary_writer=summary_writer,
                            iterator=plot_attention_iter_factory.build_iter(iepoch),
                            reporter=sub_reporter,
                            options=trainer_options,
                        )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(reporter.get_value(*val_scheduler_criterion))
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                reporter.matplotlib_plot(output_dir / "images")
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer)
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # 4. Save/Update the checkpoint
                torch.save(
                    {
                        "model": model.state_dict(),
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / "checkpoint.pth",
                )

                # 5. Save the model and update the link to the best model
                torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

                # Creates a sym link latest.pth -> {iepoch}epoch.pth
                p = output_dir / "latest.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")

                _improved = []
                for _phase, k, _mode in best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            p = output_dir / f"{_phase}.{k}.best.pth"
                            if p.is_symlink() or p.exists():
                                p.unlink()
                            p.symlink_to(f"{iepoch}epoch.pth")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.info("There are no improvements in this epoch")
                else:
                    logging.info(
                        "The best model has been updated: " + ", ".join(_improved)
                    )

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[:keep_nbest_models])
                        for ph, k, m in best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )
                for e in range(1, iepoch):
                    p = output_dir / f"{e}epoch.pth"
                    if p.exists() and e not in nbests:
                        p.unlink()
                        _removed.append(str(p))
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if patience is not None:
                if reporter.check_early_stopping(patience, *early_stopping_criterion):
                    break

        else:
            logging.info(f"The training was finished at {max_epoch} epochs ")

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        pseudo_iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer: Optional[SummaryWriter],
        options: TrainerOptions,
    ) -> bool:
        assert check_argument_types()

        # Note(kamo): assumes one optimizer
        assert cls.num_optimizers == 1, cls.num_optimizers
        assert len(optimizers) == 1, len(optimizers)
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        pseudo_iterator = iter(pseudo_iterator)

        start_time = time.perf_counter()
        for iiter, (_, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):  
            # support iterator for aux dataset
            _, pseudo_batch = pseudo_iterator.next()

            assert isinstance(batch, dict), type(batch)
            
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            with autocast(scaler is not None):
                with reporter.measure_time("forward_time"):
                    loss, stats, weight = model(**batch)
                    loss_pseudo, _, weight_pseudo = model(**pseudo_batch)
                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()
                    loss += (loss_pseudo * weight_pseudo.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    # Scales loss.  Calls backward() on scaled loss
                    # to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose
                    # for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    norm_type=grad_clip_type,
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )

                    # Must invoke scaler.update() if unscale_() is used in the iteration
                    # to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()

                else:
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        if scaler is not None:
                            # scaler.step() first unscales the gradients of
                            # the optimizer's assigned params.
                            scaler.step(optimizer)
                            # Updates the scale for next iteration.
                            scaler.update()
                        else:
                            optimizer.step()
                    if isinstance(scheduler, AbsBatchStepScheduler):
                        scheduler.step()
                optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"lr_{i}": pg["lr"]
                            for i, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        return all_steps_are_invalid

