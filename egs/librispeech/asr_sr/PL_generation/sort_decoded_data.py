#!/usr/bin/env python3

import numpy as np
import argparse

def add_args(parser):
    parser.add_argument("--wrd_file", default=None)
    parser.add_argument("--ltr_file", default=None)
    parser.add_argument("--wrd_save", default=None)
    parser.add_argument("--ltr_save", default=None)
    return parser

parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

assert isinstance(args.wrd_file, str)
assert isinstance(args.ltr_file, str)

with open(args.wrd_file, 'r') as f:
    wrd_lists = f.readlines()
with open(args.ltr_file, 'r') as f:
    ltr_lists = f.readlines()

for lists, save_file in zip([wrd_lists, ltr_lists], [args.wrd_save, args.ltr_save]):
    texts=[]
    ids=[]
    for listt in lists:
        texts.append(listt.split("(None-")[0] + '\n')
        ids.append(int(listt.split("(None-")[1].rstrip(")\n")))
    texts = np.array(texts)
    ids = np.array(ids)
    ids_sorted = np.argsort(ids)
    texts_sorted = texts[ids_sorted]

    with open(save_file, 'w') as f:
        for text in texts_sorted:
            f.write(text)
