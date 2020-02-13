#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import librosa

import soundfile as sf
import numpy as np
from tqdm import tqdm

from moneynet.utils.io import load_file_list
from moneynet.bin.unsup_train import get_parser


def make_audio_feat(filename, feat_type, raw_recon=False, feat_dim=40):
    raw, samplerate = sf.read(filename)

    if feat_type == 'stft':
        feats = librosa.stft(raw, feat_dim=64)
        raw_ = librosa.istft(feats)
        feats = librosa.amplitude_to_db(abs(feats))
    elif feat_type == 'mfcc':
        feats = librosa.feature.mfcc(y=raw, sr=samplerate, n_mfcc=feat_dim)
        raw_ = librosa.feature.inverse.mfcc_to_audio(mfcc=feats)

    if raw_recon:
        sf.write(filename + 'temp.wav', raw, samplerate)
        sf.write(filename + 'temp_recon.wav', raw_, samplerate)

    return feats, raw


if __name__ == '__main__':
    parser = get_parser()
    args, _ = parser.parse_known_args(sys.argv[1:])

    filelist = load_file_list(args.datadir)
    num_samples = len(filelist)

    if not os.path.exists(args.indir):
        os.makedirs(args.indir)

    for idx in tqdm(range(0, num_samples)):
        feat, _ = make_audio_feat(filelist[idx],
                                  feat_type=args.feat_type,
                                  feat_dim=args.feat_dim)

        file_name = os.path.basename(filelist[idx]).split('.')[0]
        np.save('{}/{}.npy'.format(args.indir, file_name), feat.astype(np.float32))
