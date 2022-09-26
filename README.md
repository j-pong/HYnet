# HYnet
Conventional / End-to-end / Representation Learning Combined Toolkit

## Installation Guide for SR

### Essentials
- CUDA VERSION == 11.0
- docker image = nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

### 0. Install pre-requisites
```bash
sudo apt-get install zlib1g-dev automake autoconf sox gfortran libtool subversion python2.7 unzip wget python3 git
```

### 1. Installaion of kaldi for speech
```bash
cd /path/to/HYnet/tools
git clone https://github.com/kaldi-asr/kaldi kaldi

cd kaldi/tools
extras/install_mkl.sh -s
extras/check_dependencies.sh
make -j 28
extras/install_irstlm.sh

cd ../src/
./configure
make depend -j 28
make -j 28
```

### 2. Installaion of espnet for input pipelines
```bash
cd /path/to/HYnet/tools
./meta_installers/install_espnet.sh
```

### 3. Installaion of hynet for customizing egs
```bash
cd /path/to/HYnet/tools
./meta_installers/install_hynet.sh
```

### 4. Installaion of fairseq
```bash
cd tools
./meta_installers/install_fairseq.sh
```

### 5. Install pre-requisite modules
```bash
cd /path/to/HYnet/egs/librispeech/asr_sr
. ./path.sh
pip install editdistance
```

## Bugfix

ctc_segmentation/ctc_segmentation_dyn.pyx error
- Remove ctc_segmentation at tools/espnet/setup.py that is included in requirements

ctc install error with pip version
```bash
pip install pip==19; pip install warpctc-pytorch==0.2.1+torch16.cuda102
```

matplotlib version error
- Remove matplotlib at tools/espnet/setup.py that is included in requirements
- OR sudo apt-get install libfreetype-dev

No CMAKE_CUDA_COMPILER could be found
```bash
export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Could NOT find NCCL
- https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html

## Run examples
for argument usage, check defaults in each files

- Librispeech Data Preparation for fairseq
```bash
cd HYnet/egs/librispeech/asr_sr
. prepare_fairseq_data.sh --ul_data_path /path/to/unlabeled_data --l_data_path /path/to/labeled_data --save_ul_data_dir /path/to/save/unlabeled_data --save_l_data_dir /path/to/save/labeled_data
```

- Wav2vec 2.0 Pretrain
```bash
cd HYnet/egs/librispeech/asr_sr
. w2v_pretrain.sh --data_dir /path/to/unlabeled_data --ngpu number_of_gpus --config_dir /path/to/config/directory --config_name yaml_file_in_config_dir
```

- Wav2vec 2.0 - CTC Finetuning
```bash
# use get_public_models.sh for pretrained representation models
cd HYnet/egs/librispeech/asr_sr
. w2v_ctc_finetune.sh --data_dir /path/to/unlabeled_data --ngpu number_of_gpus --config_dir /path/to/config/directory --config_name yaml_file_in_config_dir
```

- Wav2vec 2.0 - CTC Inference
```bash
# use get_public_lm.sh for pretrained language models
cd HYnet/egs/librispeech/asr_sr
. w2v_ctc_infer.sh --finetuned_model /path/to/finetuned_model.pt --inference_result /path/to/save/results
```

- Wav2vec 2.0 - S2S Finetuning & Inference
```bash
# Copy pretrained w2v model as /path/to/asr_sh/downloads/wav2vec_pretrained_models/libri960_big.pt
# use get_public_models.sh for pretrained representation models
cd HYnet/egs/librispeech/asr_sr
. w2v_s2s_finetune.sh
```

- Wav2vec 2.0 - S2S Semi-supervised Learning \
```bash
# Change "semi_mode" in /path/to/asr_sr/conf/tuning/train_asr_wav2vec_s2s_semi.yaml
cd HYnet/egs/librispeech/asr_sr
. w2v_s2s_finetune_semi.sh
```

## Generate pseudo-labels
for argument usage, check defaults in each files
make sure dict.ltr.txt is in /path/to/asr_sr/datas

- Audio data need to be in /path/to/audios
```bash
cd HYnet/egs/librispeech/asr_sr
. generate_pseudo_data.sh --audio_data_dir /path/to/audios --audio_extension ext(e.g., flac) \ 
--data_dir /audio_data/will/be/generated/here --pl_data_dir /pl_data/will/be/generated/here --finetuned_model /path/to/model.pt
```

## Generate duration files
for argument usage, check defaults in each files
make sure dict.ltr.txt is in /path/to/asr_sr/datas

- Audio & label data need to be in /path/to/audios in LibriSpeech data format
```bash
cd HYnet/egs/librispeech/asr_sr
. generation_duration.sh --audio_data_dir /path/to/audios --audio_extension ext(e.g., flac) \ 
--data_dir /manifest_data/will/be/generated/here --finetuned_model /path/to/model.pt
```
