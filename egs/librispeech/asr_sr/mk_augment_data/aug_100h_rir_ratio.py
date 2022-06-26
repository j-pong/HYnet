# --------------------------------------------------------
# RIR-Augmentation using np.convolve 
# Modified by Juseok Seong
# --------------------------------------------------------

import soundfile as sf
import random
import librosa
import numpy as np
import os
import pickle
import math 
import time 
import gzip

import argparse

start = time.time() 
math.factorial(100000) 

def add_args(parser):
    parser.add_argument("--rir_dir", default="/DB/LibriSpeech/LibriSpeech/RIR")
    parser.add_argument("--train_clean_100", default="/home/juseok/workspace/HYnet/egs/librispeech/asr_sr/data/fairseq/train_clean_100")
    parser.add_argument("--train_aug_100", default="/home/juseok/workspace/HYnet/egs/librispeech/asr_sr/data/fairseq/train_aug100_half")
    parser.add_argument("--train_aug_DB", default="/DB/LibriSpeech/LibriSpeech/train_aug100_half")
    parser.add_argument("--aug_ratio", type=int, default=50, help="5, 10, 25, 50 100")
    return parser

parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

max_range = 100 // args.aug_ratio 

if not os.path.isdir(args.rir_dir):
    print("rir_dir have to contain rir wav file and rir list(RIRlist.txt)")
    print("Download or make rir dir")
    print("exit")
    exit()
    
if not os.path.isdir(args.train_aug_100):
    print("mkdir {}".format(args.train_aug_100))
    os.makedirs(args.train_aug_100)


with open(args.train_clean_100 + "/train.tsv" , "r") as clean_f:
    root_dir = clean_f.readline().strip()

    with open(args.train_aug_100 + "/train.tsv", 'w') as aug_f: 
        aug_f.write(args.train_aug_DB+'\n')

        with open(args.rir_dir + "/RIRlist.txt", 'r') as aug:      
            lines=aug.read().splitlines()
            aug_wavs={} 

        for i, line in enumerate(clean_f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
      
            index = items[0].find('flac') -1

            path_or_fp = os.path.join(root_dir, str(items[0]))
            path_new = os.path.join(args.train_aug_DB, str(items[0]))
            
            if not os.path.isdir(('/').join(path_new.split('/')[:-1])):
                os.makedirs(('/').join(path_new.split('/')[:-1]))
            
            # read wav-file
            wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")             
            
            aug = (random.randint(1, max_range) == 1) or max_range == 1 
            if aug:
                # augment using np.convolve
                rir_idx = random.randrange(0,60000)    
                rir_path=lines[rir_idx]
                y_rir, _ = librosa.load(args.rir_dir + rir_path.lstrip('.'), sr=16000)

                aug_wav = np.convolve(wav, y_rir)

                # save wav file
                sf.write(args.train_aug_DB + "/" +  items[0][:index] + items[0][index:], aug_wav, curr_sample_rate)
                aug_f.write(items[0][:index] + items[0][index:] +"\t"+items[1] + "\n")
            else:
                # save clean wav file
                sf.write(args.train_aug_DB + "/" + items[0][:index] + items[0][index:], wav, curr_sample_rate)
                aug_f.write(items[0][:index] + items[0][index:] + "\t" + items[1] + "\n")

end = time.time() 

print(f"{end - start:.5f} sec")		
print("done")