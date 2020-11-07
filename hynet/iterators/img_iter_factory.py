from typing import Any
from typing import Sequence
from typing import Union

import numpy as np
from torch.utils.data import DataLoader
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


class ImgrIterFactory(AbsIterFactory):
    def __init__(
        self,
        dataset,
        batch_size,
        num_iters_per_epoch: int = None,
        seed: int = 0,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iters_per_epoch = num_iters_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702
        self.pin_memory = pin_memory

    def build_iter(self, epoch: int, shuffle: bool = None) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle

        if self.collate_fn is not None:
            kwargs = dict(collate_fn=self.collate_fn)
        else:
            kwargs = {}

        set_all_random_seed(self.seed)

        return DataLoader(
            dataset=self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=shuffle,
        )
