#!/usr/bin/env python3
# encoding: utf-8

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import logging
import sys

from kaldiio import ReadHelper, load_mat
from multiprocessing import Process, Manager

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description='get gmm-alignment pdf-id',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str,
                        help='data directory where recipes are saved')
    parser.add_argument('--ali_dir', type=str, nargs='+',
                        help='gmm alignment saved directory (where ali.*.gz exists)')
    parser.add_argument('--fea_scp', type=str,
                        help='feats.scp file path')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()

    # TODO(j-pong): multi-processing
    ali_dirs = list()
    if isinstance(args.ali_dir, list):
        for ali_dir in args.ali_dir:
            ali_dir = ali_dir + 'ali.*.gz' if ali_dir.endswith('/') else ali_dir + '/ali.*.gz'
            ali_dirs.append(ali_dir)
    else:
        ali_dirs.append(args.ali_dir + 'ali.*.gz' if args.ali_dir.endswith('/') else args.ali_dir + '/ali.*.gz')

    # get gmm-alignment and save to lab_dict
    # lab_dict = {utt_keys: alignment pdf-id}
    lab_dict = dict()
    tokenid_scp = args.data_dir + 'tokenid.scp' if args.data_dir.endswith('/') else args.data_dir + '/tokenid.scp'
    with open(tokenid_scp, 'w') as tokenid:
        for ali_dir in ali_dirs:
            with ReadHelper('ark: gunzip -c {} |'.format(ali_dir)) as labs:
                for key, array in labs:
                    lab_dict[key] = array
                    tokenid.write(str(key) + ' ' + str(' '.join(list(map(str, array)))) + '\n')
    utt_keys = list(lab_dict.keys())

    # remove utt-ids with no alignments from feat.scp, text, utt2spk
    text_path = args.data_dir + 'text_org' if args.data_dir.endswith('/') else args.data_dir + '/text_org'
    scp_path = args.fea_scp
    utt2spk_path = args.data_dir + 'utt2spk_org' if args.data_dir.endswith('/') else args.data_dir + '/utt2spk_org'

    needs_to_be_filtered = [text_path, scp_path, utt2spk_path]
    lines = []
    fids = []

    for file_path in needs_to_be_filtered:
        f = open(file_path, 'r')
        lines.append(f.readlines())
        f.close()

    for i, file_path in enumerate(needs_to_be_filtered):
        file_path = file_path.split('_org.scp')[0] if i == 0 else file_path.split('_org')[0]
        fids.append(open(file_path, 'w'))

    for i, line in enumerate(lines[0]):
        if line.split(' ')[0] not in utt_keys:
            continue
        for j, fid in enumerate(fids):
            fid.write(str(line[j][i]))
