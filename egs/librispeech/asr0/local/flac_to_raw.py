#!/usr/bin/env python3

import os
import argparse
import sys

def get_parser():
    parser = argparse.ArgumentParser(
        description='data output path',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_dir', type=str,
                        help='data directory where recipes are saved')
    parser.add_argument('--dev1_dir', type=str,
                        help='data directory where recipes are saved')
    parser.add_argument('--dev2_dir', type=str, default=None,
                        help='data directory where recipes are saved')
    parser.add_argument('--test1_dir', type=str,
                        help='data directory where recipes are saved')
    parser.add_argument('--test2_dir', type=str, default=None,
                        help='data directory where recipes are saved')
    parser.add_argument('--out_prefix', type=str,
                        help='data output directory prefix where recipes will be saved')
    return parser

args = get_parser().parse_args()
train_base = args.train_dir + "_" + args.out_prefix
dev1_base = args.dev1_dir + "_" + args.out_prefix
if args.dev2_dir is not None:
    dev2_base = args.dev2_dir + "_" + args.out_prefix
test1_base = args.test1_dir + "_" + args.out_prefix
if args.test2_dir is not None:
    test2_base = args.test2_dir + "_" + args.out_prefix

if not os.path.exists(train_base):
    os.makedirs(train_base)

if not os.path.exists(dev1_base):
    os.makedirs(dev1_base)

if args.dev2_dir is not None and not os.path.exists(dev2_base):
    os.makedirs(dev2_base)

if not os.path.exists(test1_base):
    os.makedirs(test1_base)

if args.test2_dir is not None and not os.path.exists(test2_base):
    os.makedirs(test2_base)

train_lst = open(args.train_dir + '/wav.scp', "r").readlines()
dev1_lst = open(args.dev1_dir + '/wav.scp', "r").readlines()
test1_lst = open(args.test1_dir + '/wav.scp', "r").readlines()
if args.dev2_dir is not None:
    dev2_lst = open(args.dev2_dir + '/wav.scp', "r").readlines()
if args.test2_dir is not None:
    test2_lst = open(args.test2_dir + '/wav.scp', "r").readlines()


out_train_lst = open(train_base+"/wav.scp", "w")
out_dev1_lst = open(dev1_base+"/wav.scp", "w")
out_test1_lst= open(test1_base+"/wav.scp", "w")
if args.dev2_dir is not None:
    out_dev2_lst = open(dev2_base + "/wav.scp", "w")
if args.test2_dir is not None:
    out_test2_lst = open(test2_base + "/wav.scp", "w")

for i, line in enumerate(train_lst):
    uttid = train_lst[i].split(' ')[0]
    file_path = line.split(' ')[-2]
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_train_lst.write(uttid.rstrip("\n")+" "+file_path.split(".")[0]+".wav\n")

for i, line in enumerate(dev1_lst):
    uttid = dev1_lst[i].split(' ')[0]
    file_path = line.split(' ')[-2]
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_dev1_lst.write(uttid.rstrip("\n")+" "+file_path.split(".")[0]+".wav\n")

for i, line in enumerate(test1_lst):
    uttid = test1_lst[i].split(' ')[0]
    file_path = line.split(' ')[-2]
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_test1_lst.write(uttid.rstrip("\n")+" "+file_path.split(".")[0]+".wav\n")

if args.dev2_dir is not None:
    for i, line in enumerate(dev2_lst):
        uttid = dev2_lst[i].split(' ')[0]
        file_path = line.split(' ')[-2]
        os.system("sox " + file_path + " " + file_path.split(".")[0] + ".wav")
        out_dev2_lst.write(uttid.rstrip("\n") + " " + file_path.split(".")[0] + ".wav\n")

if args.test2_dir is not None:
    for i, line in enumerate(test2_lst):
        uttid = test2_lst[i].split(' ')[0]
        file_path = line.split(' ')[-2]
        os.system("sox " + file_path + " " + file_path.split(".")[0] + ".wav")
        out_test2_lst.write(uttid.rstrip("\n") + " " + file_path.split(".")[0] + ".wav\n")

exit()
