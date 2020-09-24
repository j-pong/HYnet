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

from hynet.train.dataset import MNISTDataset, CIFAR10Dataset
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

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(HynetImgrModel),
            help="The keyword arguments for model class.",
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
            **args.model_conf,
        )

        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

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

        dataset = CIFAR10Dataset(
            root='data',
            train=train,
            download=True,
            transform=transform)
        # dataset = MNISTDataset(
        #     root='data',
        #     train=train,
        #     download=True,
        #     transform=transform)

        return ImgrIterFactory(
            dataset=dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )
