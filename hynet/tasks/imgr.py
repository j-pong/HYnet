import argparse
import logging
import os
from pathlib import Path
from distutils.version import LooseVersion
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from espnet2.tasks.abs_task import *

import torchvision.transforms as transforms

from hynet.train.dataset import MNISTDataset, CIFAR10Dataset, CIFAR100Dataset
from hynet.train.trainer import ImgrTrainer
from hynet.iterators.img_iter_factory import ImgrIterFactory


class ImgrTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = ImgrTrainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")
        parser.set_defaults(num_att_plot=0, iterator_type="task")
        group.add_argument("--xai_excute", type=int, default=0)
        group.add_argument("--xai_mode", type=str, default="gxi")
        group.add_argument("--xai_iter", type=int, default=3)

        group = parser.add_argument_group(description="Input related")
        group.add_argument(
            "--dataset",
            type=str,
            default="cifar10",
            choices=[
                "mnist",
                "cifar10",
                "cifar100",
                "imagenet"
            ]   
        )
        
        group = parser.add_argument_group(description="Model related")
        group.add_argument("--cfg_type", type=str, default="D")
        group.add_argument("--batch_norm", type=int, default=0)
        group.add_argument("--bias", type=int, default=0)
        group.add_argument("--in_ch", type=int, default=3)
        group.add_argument("--out_ch", type=int, default=10)

        group = parser.add_argument_group(description="Teacher Model related")
        group.add_argument("--st_excute", type=int, default=0)
        group.add_argument("--teacher_model_path", type=str, default="")
        group.add_argument("--teacher_cfg_type", type=str, default="nib_B2")

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        return None

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        return None

    @classmethod
    def required_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
        if not inference:
            retval = ("image", "label")
        else:
            # Recognition mode
            retval = ("image",)
        return retval

    @classmethod
    def build_optimizers(
        cls,
        args: argparse.Namespace,
        model: torch.nn.Module,
    ) -> List[torch.optim.Optimizer]:
        if cls.num_optimizers != 1:
            raise RuntimeError(
                "build_optimizers() must be overridden if num_optimizers != 1"
            )

        optim_class = optim_classes.get(args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        if args.st_excute:
            # Teacher model parameter unused mode for st framework
            optim = optim_class(model.model.parameters(), **args.optim_conf)
        else:
            optim = optim_class(model.parameters(), **args.optim_conf)
        optimizers = [optim]
        return optimizers

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()

        if args.st_excute:
            from hynet.imgr.imgr_st_model import HynetImgrModel

            model = HynetImgrModel(
                args.xai_excute,
                args.xai_mode,
                args.xai_iter,
                args.st_excute,
                args.cfg_type,
                args.teacher_cfg_type,
                args.teacher_model_path,
                args.batch_norm,
                args.bias,
                args.in_ch,
                args.out_ch
            )
        else:
            from hynet.imgr.imgr_model import HynetImgrModel

            model = HynetImgrModel(
                args.xai_excute,
                args.xai_mode,
                args.xai_iter,
                args.st_excute,
                args.cfg_type,
                args.batch_norm,
                args.bias,
                args.in_ch,
                args.out_ch
            )

        assert check_return_type(model)
        return model

    @classmethod
    def build_task_iter_factory(
        cls, args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ) -> AbsIterFactory:
        assert check_argument_types()

        if mode == 'train':
            train = True
        elif mode == 'valid':
            train = False
        else:
            ValueError("{} is not implemented!".format(mode))

        if args.dataset == 'mnist':
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507], std=[0.267])
                ])
            dataset = MNISTDataset(
                root='data',
                train=train,
                download=False,
                transform=transform)
        elif args.dataset == 'cifar10':
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
            dataset = CIFAR10Dataset(root='data',
                                     train=train,
                                     download=False,
                                     transform=transform)
        elif args.dataset == 'cifar100':
            if train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
                ])
            dataset = CIFAR100Dataset(root='data',
                                      train=train,
                                      download=False,
                                      transform=transform)

        return ImgrIterFactory(
            dataset=dataset,
            batch_size=args.batch_size,
            seed= args.seed if args.dist_rank is None else args.seed + args.dist_rank,  # pseudo splited the data for distributed mode
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )
