#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
import numpy as np

from moneynet.utils.io import load_file_list


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


class Pikachu(torch.utils.data.Dataset):
    def __init__(self, args, transform=None, ram_memory=True):
        # load and set the ram mode
        self.filelist = load_file_list(args.indir)
        self.ram_memory = ram_memory

        # task related
        self.num_samples = len(self.filelist)
        self.ngpu = args.ngpu
        self.transform = transform
        self.batch_size = args.batch_size
        self.ignore_in = args.ignore_in
        self.ignore_out = args.ignore_out
        self.datamper = args.datamper

        # ram_memory is waring to small ram case
        if ram_memory:
            print("Start buffering for ram_memory mode")
            self.buffer = {}
            for idx in tqdm(range(0, self.num_samples)):
                feat = np.load(self.filelist[idx], allow_pickle=True)
                self.buffer[idx] = feat  # numpy array attach to key that sample number

    def _batch_with_padding(self, idx):
        # sampling indexs that independent to dataloader (pytorch) idx
        index_queue = np.random.randint(0, self.num_samples, size=self.batch_size)
        # batch sampling
        batch_in_feat = []
        batch_out_feat = []
        batch_fname = []
        for idx in index_queue:
            if self.ram_memory:
                feat = torch.from_numpy(self.buffer[idx].T)
            else:
                feat = torch.from_numpy(np.load(self.filelist[idx], allow_pickle=True).T)
            batch_in_feat.append(feat[:-1])
            batch_out_feat.append(feat[1:])
            batch_fname.append(self.filelist[idx])

        return pad_list(batch_in_feat, self.ignore_in), pad_list(batch_out_feat, self.ignore_out), batch_fname

    def __len__(self):
        return int(self.num_samples / self.batch_size) * self.ngpu * self.datamper

    def __dims__(self):
        sample = self.__getitem__(0)
        return np.shape(sample['input'])[-1], np.shape(sample['target'])[-1]

    def __getitem__(self, idx):
        in_feats, out_feats, fnames = self._batch_with_padding(idx)  # [B, T, C], [B, T, C], [B]
        sample = {'input': in_feats, 'target': out_feats, 'fname': fnames}
        return sample
