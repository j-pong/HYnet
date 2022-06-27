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
    parser.add_argument("--train_860", default="./data/fairseq/train_l100_ul860/ultrain.tsv")
    parser.add_argument("--train_aug_860", default="./data/fairseq/train_aug_ul860")
    parser.add_argument("--train_aug_DB", default="/DB/LibriSpeech/LibriSpeech/train-aug860")
    parser.add_argument("--aug_ratio", type=int, default=100, help="5, 10, 25, 50 100")
    return parser

parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

if not os.path.isdir(args.rir_dir):
    print("rir_dir have to contain rir wav file and rir list(RIRlist.txt)")
    print("Download or make rir dir")
    print("exit")
    exit()
    
# make "aug_ultrain.tsv" in train_aug_ul860 folder
if not os.path.isdir(args.train_aug_860):
    os.makedirs(args.train_aug_860)

with open(args.train_860 , "r") as f:
    root_dir = f.readline().strip()
    save_root_dir = args.train_aug_DB

    with open(args.train_aug_860 + "/aug_ultrain.tsv", 'w') as aug_f: 
        aug_f.write(save_root_dir+'\n')

        with open(args.rir_dir + "/RIRlist.txt", 'r') as aug:      
            lines=aug.read().splitlines()
            aug_wavs={} 

        for i, line in enumerate(f):
            rand_number = random.randrange(0,60000)
            
            items = line.strip().split("\t")
                
            assert len(items) == 2, line
            sz = int(items[1])
                
            path_or_fp = os.path.join(root_dir, str(items[0]))
            path_new = os.path.join(save_root_dir, str(items[0]))

            wav, curr_sample_rate = sf.read(path_or_fp, dtype="float32")             
        
            rir_path=lines[rand_number]
            y_rir, _ = librosa.load(args.rir_dir + rir_path.lstrip('.'), sr=16000)

            aug_wav = np.convolve(wav, y_rir)
            
            if not os.path.isdir(('/').join(path_new.split('/')[:-1])):
                os.makedirs(('/').join(path_new.split('/')[:-1]))

            index = items[0].find('flac') -1

            sf.write(args.train_aug_DB + "/" + items[0][:index] + "aug" +items[0][index:], aug_wav, curr_sample_rate)
            aug_f.write(items[0][:index] + "aug" +items[0][index:]+"\t"+items[1] +"\n")


end = time.time() 
print(f"{end - start:.5f} sec")		