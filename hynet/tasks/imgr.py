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

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.tasks.abs_task import AbsTask, IteratorOptions, AbsIterFactory
from espnet2.torch_utils.initialize import initialize
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str_or_none

from hynet.train.dataset import MNISTDataset, CIFAR10Dataset, CIFAR100Dataset
from hynet.train.trainer import ImgrTrainer
from hynet.iterators.img_iter_factory import ImgrIterFactory
from hynet.imgr.imgr_model import HynetImgrModel


class ImgrTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = ImgrTrainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")
        parser.set_defaults(
            num_att_plot=0,
            iterator_type="task")
        
        # Custom task
        group.add_argument(
            "--xai_excute",
            type=int,
            default=0 
        )
        group.add_argument(
            "--xai_mode",
            type=str,
            default="bg"
        )
        group.add_argument(
            "--xai_iter",
            type=int,
            default=3 
        )
        group.add_argument(
            "--st_excute",
            type=int,
            default=0 
        )
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
        # Model configuration
        group.add_argument(
            "--cfg_type",
            type=str,
            default="D" 
        )
        group.add_argument(
            "--batch_norm",
            type=int,
            default=0 
        )
        group.add_argument(
            "--bias",
            type=int,
            default=0 
        )
        group.add_argument(
            "--in_ch",
            type=int,
            default=3
        )
        group.add_argument(
            "--out_ch",
            type=int,
            default=10
        )

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
    def build_model(cls, args: argparse.Namespace) -> HynetImgrModel:
        assert check_argument_types()

        # 1. Build model
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
