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

def plot_and_save(
    img_list: List, 
    output_dir: str,
    xai_mode: str, 
    ids: int, 
    max_plot=3
) -> None:
    # import matplotlib.pyplot as plt
    import matplotlib
    
    for k in range(max_plot):
        img_ = img_list[k].detach().cpu().numpy()
        for j in range(50):
            # plt.clf()
            # plt.imshow(img_[j], cmap='seismic')
            # plt.colorbar()
            # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

            if output_dir is not None:
                p = Path(output_dir) / Path("masked_input") / Path(xai_mode) /  Path(f"valid_{ids[j]}") /Path(f"iter{k}.png")
                p.parent.mkdir(parents=True, exist_ok=True)
                matplotlib.image.imsave(p, img_[j], cmap='seismic')

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
    device = f"cuda:{torch.cuda.current_device()}"
    model.to(device)
    model.load_state_dict(torch.load(Path(args.output_dir) / Path(args.model_file), map_location=device))
    
    ngpu = args.ngpu
    
    model.eval()
    
    flag = args.plot_flag
    count = 0
    reporter = {}
    for (ids, batch) in tqdm(iterator):
        assert isinstance(batch, dict), type(batch)

        batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

        _, stats, weight = model(**batch, return_plot=flag)
        
        if flag:
            img_list = stats['imgs']
            plot_and_save(img_list, args.output_dir, args.xai_mode, ids, max_plot=args.xai_iter)
            flag = False
            del stats['imgs']
        
        if count == 0:
            for k, v in stats.items():
                reporter[k] = v
        else:
            for k, v in stats.items():
                reporter[k] += v                
        count += 1

    print("=========Results=========")
    for k, v in reporter.items():
        print(k+':', float(v / count))
    

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
    parser.add_argument("--plot_flag", type=int, default=0, help="Return Maksed Input Feature for saving at output dir")

    group = parser.add_argument_group("Task related")
    group.add_argument("--xai_excute", type=int, default=0)
    group.add_argument("--xai_mode", type=str, default="gxi")
    group.add_argument("--xai_iter", type=int, default=3)

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

    group = parser.add_argument_group(description="Teacher Model related")
    group.add_argument("--st_excute", type=int, default=0)
    group.add_argument("--teacher_model_path", type=str, default="")
    group.add_argument("--teacher_cfg_type", type=str, default="nib_B2")

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
