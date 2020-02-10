import os
import librosa

import soundfile as sf
import numpy as np

import torch.utils.data as data


def load_file_list(datadir):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(datadir):
        for filename in filenames:
            f.append(os.path.join(dirpath, filename))

    return f


def make_mfcc(filename, feat_type='mfcc', raw_recon=False, n_mfcc=40):
    raw, samplerate = sf.read(filename)

    if feat_type == 'stft':
        feats = librosa.stft(raw, n_fft=64)
        raw_ = librosa.istft(feats)
        feats = librosa.amplitude_to_db(abs(feats))
    elif feat_type == 'mfcc':
        feats = librosa.feature.mfcc(y=raw, sr=samplerate, n_mfcc=n_mfcc)
        raw_ = librosa.feature.inverse.mfcc_to_audio(mfcc=feats)

    if raw_recon:
        sf.write(filename + 'temp.wav', raw, samplerate)
        sf.write(filename + 'temp_recon.wav', raw_, samplerate)

    return feats, raw


class Pikachu(data.Dataset):
    def __init__(self, root, transform=None, feat_type='mfcc'):
        self.filelist = load_file_list(root)
        self.num_samples = len(self.filelist)
        self.feat_type = feat_type
        self.transform = transform

        self.n_mfcc = 40

    def __getitem__(self, index):
        feature, raw = make_mfcc(self.filelist[index], feat_type=self.feat_type, n_mfcc=self.n_mfcc)

        if self.transform is not None:
            feature = self.transform(feature)

        sample = {'input': feature[:, 1:], 'target': feature[:, :-1]}

        return sample

    def __len__(self):
        return self.num_samples

    def __dims__(self):
        samples = self.__getitem__(0)
        return np.shape(samples['input'])[0], np.shape(samples['target'])[0]
