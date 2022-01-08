import soundfile as sf
import random
import librosa
import numpy as np
import os
import pickle
import math 
import time 
import gzip

start = time.time() 
math.factorial(100000) 

# make "aug_ultrain.tsv" in train_aug_ul860 folder
if not os.path.isdir("./data/fairseq/train_aug_ul860"):
    os.makedirs("./data/fairseq/train_aug_ul860")

if not os.path.isdir("./data/RIR"):
    print("Download RIR file")
    exit()

with open("./data/fairseq/train_l100_ul860/ultrain.tsv" , "r") as f:
    root_dir = f.readline().strip()
    save_root_dir = "/DB/LibriSpeech/LibriSpeech/train-aug860"

    with open("./data/fairseq/train_aug_ul860/aug_ultrain.tsv", 'w') as aug_f: 
        aug_f.write(save_root_dir+'\n')

        with open("./data/RIR/RIRlist.txt", 'r') as aug:      
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
            y_rir, _ = librosa.load("./data/RIR" + rir_path.lstrip('.'), sr=16000)

            aug_wav = np.convolve(wav, y_rir)
            
            if not os.path.isdir(('/').join(path_new.split('/')[:-1])):
                os.makedirs(('/').join(path_new.split('/')[:-1]))

            index = items[0].find('flac') -1

            sf.write("/DB/LibriSpeech/LibriSpeech/train-aug860/" + items[0][:index] + "aug" +items[0][index:], aug_wav, curr_sample_rate)
            aug_f.write(items[0][:index] + "aug" +items[0][index:]+"\t"+items[1] +"\n")


end = time.time() 
print(f"{end - start:.5f} sec")		