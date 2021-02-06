#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

from typing import Dict
from typing import List
from typing import Union
from typeguard import check_argument_types

import numpy as np
import torch
import torchvision.transforms as transforms

from espnet.utils.cli_utils import get_commandline_args
from espnet2.utils import config_argparse
from espnet2.torch_utils.device_funcs import to_device

from hynet.iterators.img_iter_factory import ImgrIterFactory
from hynet.train.dataset import MNISTDataset, CIFAR10Dataset, CIFAR100Dataset
import torchvision.transforms as transforms

from hynet.tasks.imgr import ImgrTask

from tqdm import tqdm

# def plot_(
#     img_list: List, 
#     output_dir: Path, 
#     ids: int, 
#     epoch: int,
#     max_plot=3
# ) -> None:
#     import matplotlib.pyplot as plt

#     col_max = len(img_list)
#     row_max = len(img_list[0])
    
#     for k in range(max_plot):
#         plt.clf()

#         for i in range(col_max):
#             for j, img in enumerate(img_list[i]):
#                 img_ = img[k].detach().cpu().numpy()

#                 plt.subplot(col_max, row_max, i * row_max + (j + 1))

#                 plt.imshow(img_, cmap='seismic')
#                 plt.colorbar()

#                 plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#         if output_dir is not None:
#             p = output_dir / f"valid_{ids[k]}" / f"{ids[k]}_{epoch}ep.png"
#             p.parent.mkdir(parents=True, exist_ok=True)
#             plt.savefig(p)

def inference(
    args: argparse.Namespace,
):
    assert check_argument_types()
    train = False

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
    else: 
        raise AttributeError("This dataset is not supported!")

    iterator = ImgrIterFactory(
                dataset=dataset,
                batch_size=args.batch_size,
                seed=0,
                num_iters_per_epoch=1,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=None,
                pin_memory=args.ngpu > 0,
                )
    iterator = iterator.build_iter(1)

    if args.st_excute:
        from hynet.imgr.imgr_st_model import HynetImgrModel
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
    device = f"cuda:{torch.cuda.current_device()}"
    model.to(device)
    # model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.load_state_dict(torch.load("/home/Workspace/HYnet/egs/xai/cifar10/exp/imgr_train_vgg13_bnwob/62epoch.pth", map_location=device))
    
    ngpu = args.ngpu
    model.eval()
    count = 0
    acc_accum0 = 0.0
    acc_accum1 = 0.0
    acc_accum2 = 0.0
    for (ids, batch) in tqdm(iterator):
        assert isinstance(batch, dict), type(batch)

        batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

        _, stats, weight = model(**batch)

        img_list = stats['aux']
        # # if flag:
        # #     cls.plot_(img_list, output_dir, ids, reporter.get_epoch())
        # #     flag = False
        del stats['aux']
        acc_accum0 += stats['acc_iter0']
        acc_accum1 += stats['acc_iter1']
        acc_accum2 += stats['acc_iter2']

        count += 1
    print(acc_accum0 / count)
    print(acc_accum1 / count)
    print(acc_accum2 / count)

def get_parser():
    parser = config_argparse.ArgumentParser(
        description="IMGR Decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=88)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Task related")
    group.add_argument("--xai_excute", type=int, default=0)
    group.add_argument("--xai_mode", type=str, default="gxi")
    group.add_argument("--xai_iter", type=int, default=3)
    group.add_argument("--st_excute", type=int, default=0)

    group = parser.add_argument_group("Input related")
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
    group.add_argument("--model_file", type=str, default='')

    group = parser.add_argument_group("Model related")
    group.add_argument("--cfg_type", type=str, default="D")
    group.add_argument("--batch_norm", type=int, default=0)
    group.add_argument("--bias", type=int, default=0)
    group.add_argument("--in_ch", type=int, default=3)
    group.add_argument("--out_ch", type=int, default=10)

    return parser

def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(args)

if __name__ == "__main__":
    main()
