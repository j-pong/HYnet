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

    # remove utt-ids with no alignments from feat.scp, text, utt2spk
    # text_path = os.path.join(args.data_dir, 'text')
    # utt2spk_path = os.path.join(args.data_dir + 'utt2spk')
    # scp_path = os.path.join(args.feat_dir + 'feats.scp')

    # needs_to_be_filtered = [text_path, scp_path, utt2spk_path]
    # lines = []
    # fids = []
    #
    # for i, file_path in enumerate(needs_to_be_filtered):
    #     f = open(file_path, 'r')
    #     lines.append(f.readlines())
    #     file_path = file_path.split('_org.scp')[0] + '.scp' if 'scp' in file_path else file_path.split('_org')[0]
    #     fids.append(open(file_path, 'w'))
    #     f.close()
    #
    # for i, line in enumerate(lines[0]):
    #     if line.split(' ')[0] not in utt_keys:
    #         continue
    #     for j, fid in enumerate(fids):
    #         fid.write(str(lines[j][i]))
    # for fid in fids:
    #     fid.close()
