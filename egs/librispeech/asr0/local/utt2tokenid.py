#!/usr/bin/env python3
# encoding: utf-8

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
import os

from kaldi_io import read_vec_int_ark

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description='get gmm-alignment pdf-id',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str,
                        help='data directory where recipes are saved')
    parser.add_argument('--ali_dir', type=str,
                        help='gmm alignment saved directory (where ali.*.gz exists)')
    parser.add_argument('--ali_mdl', type=str,
                        help='gmm trained model (final.mdl path)')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # get gmm-alignment and save to lab_dict
    # lab_dict = {utt_keys: alignment pdf-id}
    utts = []
    tokenid_scp = os.path.join(args.data_dir, 'tokenid.scp')
    with open(tokenid_scp, 'w') as tokenid:
        cmd = "gunzip -c " + os.path.join(args.ali_dir, 'ali.*.gz') + " | ali-to-pdf " + args.ali_mdl + " ark:- ark:-|"
        for key, array in read_vec_int_ark(cmd):
            tokenid.write(str(key) + ' ' + str(' '.join(list(map(str, array)))) + '\n')
            utts.append(key)