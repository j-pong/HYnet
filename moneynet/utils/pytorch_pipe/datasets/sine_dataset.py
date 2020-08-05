#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
import numpy as np

from skimage import io, transform

from moneynet.utils.pytorch_pipe.io import load_file_list


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


# ToDo: Large eval batchsize case is not cover yet!
class SineDataset(torch.utils.data.Dataset):
    def __init__(self, args, train, transform=None, ram_memory=True):
        filelist = load_file_list(args.indir)
        train_filelist = filelist[:]
        eval_filelist = filelist[-7:]

        if train:
            self.filelist = train_filelist
            self.batch_size = args.batch_size
        else:
            self.filelist = eval_filelist
            self.batch_size = args.eval_batch_size
        self.train = train
        self.num_samples = len(self.filelist)
        self.ngpu = args.ngpu
        self.transform = transform
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out
        self.datamper = args.datamper
        self.ram_memory = ram_memory

        # ram_memory is waring to small ram case
        if ram_memory:
            print("Start buffering for ram_memory mode")
            self.buffer = {}
            for idx in tqdm(range(0, self.num_samples)):
                feat = torch.from_numpy(io.imread(self.filelist[idx]).T)
                self.buffer[idx] = feat  # numpy array attach to key that sample number

    def __len__(self):
        return self.num_samples

    def __dims__(self):
        sample = self.__getitem__(0)
        return np.shape(sample['input'])[-1], np.shape(sample['target'])[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.ram_memory:
            feat = torch.from_numpy(self.buffer[idx].T)
        else:
            feat = torch.from_numpy(io.imread(self.filelist[idx]).T)

        sample = {'input': feat, 'target': feat, 'fname': self.filelist[idx]}
        return sample
