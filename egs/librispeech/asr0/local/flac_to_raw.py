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
train_base = args.train_dir + args.out_prefix
dev1_base = args.dev1_dir + args.out_prefix
if args.dev2_dir is not None:
    dev2_base = args.dev2_dir + args.out_prefix
test1_base = args.test1_dir + args.out_prefix
if args.test2_dir is not None:
    test2_base = args.test2_dir + args.out_prefix

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
train_uttid = train_lst.split(' ')[0]
train_lst = train_lst.split(' ')[-2]
dev1_lst = open(args.dev1_dir + '/wav.scp', "r").readlines()
dev1_uttid = dev1_lst.split(' ')[0]
dev1_lst = dev1_lst.split(' ')[-2]
test1_lst = open(args.test1_dir + '/wav.scp', "r").readlines()
test1_uttid = test1_lst.split(' ')[0]
test1_lst = test1_lst.split(' ')[-2]
if args.dev2_dir is not None:
    dev2_lst = open(args.dev2_dir + '/wav.scp', "r").readlines()
    dev2_uttid = dev2_lst.split(' ')[0]
    dev2_lst = dev2_lst.split(' ')[-2]
if args.test2_dir is not None:
    test2_lst = open(args.test2_dir + '/wav.scp', "r").readlines()
    test2_uttid = test2_lst.split(' ')[0]
    test2_lst = test2_lst.split(' ')[-2]


out_train_lst = open(train_base+"/wav.scp", "w")
out_dev1_lst = open(dev1_base+"/wav.scp", "w")
out_test1_lst= open(test1_base+"/wav.scp", "w")
if args.dev2_dir is not None:
    out_dev2_lst = open(dev2_base + "/wav.scp", "w")
if args.test2_dir is not None:
    out_test1_lst = open(test2_base + "/wav.scp", "w")

for i, line in enumerate(train_lst):
    uttid = train_uttid[i]
    file_path = line
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_train_lst.write(uttid.rstrip("\n")+" "+file_path.split(".")[0]+".wav\n")

for i, line in enumerate(dev1_lst):
    uttid = dev1_uttid[i]
    file_path = line
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_train_lst.write(uttid.rstrip("\n")+" "+file_path.split(".")[0]+".wav\n")

for i, line in enumerate(test1_lst):
    uttid = test1_uttid[i]
    file_path = line
    os.system("sox "+file_path+" "+file_path.split(".")[0]+".wav")
    out_train_lst.write(uttid.rstrip("\n")+" "+file_path.split(".")[0]+".wav\n")

if args.dev2_dir is not None:
    for i, line in enumerate(dev2_lst):
        uttid = dev2_uttid[i]
        file_path = line
        os.system("sox " + file_path + " " + file_path.split(".")[0] + ".wav")
        out_train_lst.write(uttid.rstrip("\n") + " " + file_path.split(".")[0] + ".wav\n")

if args.test2_dir is not None:
    for i, line in enumerate(test2_lst):
        uttid = test2_uttid[i]
        file_path = line
        os.system("sox " + file_path + " " + file_path.split(".")[0] + ".wav")
        out_train_lst.write(uttid.rstrip("\n") + " " + file_path.split(".")[0] + ".wav\n")

exit()
