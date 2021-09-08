#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$PWD/data/fairseq/train_clean_100/
pretrained_model=$PWD/models/pretrained_w2v/checkpoint_best.pt
ngpu=4
update_freq=$((24/$ngpu))

# make sure to set model save_dir path in vox_100h.yaml
# checkpoint:
#   save_dir: /path/to/model/save/dir
fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=$data_dir \
    model.w2v_path=$pretrained_model \
    distributed_training.distributed_world_size=$ngpu +optimization.update_freq='['$update_freq']' \
    --config-dir $PWD/../../../tools/fairseq/examples/wav2vec/config/finetuning \
    --config-name vox_100h
