#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$PWD/data/fairseq/train_l100_ul860/
pretrained_model=$PWD/models/pretrained_w2v/checkpoint_best.pt
ngpu=4
config_dir=$PWD/../../../tools/fairseq/examples/wav2vec/config/finetuning
config_name=vox_100h

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

# make sure to set model save_dir path in vox_100h.yaml
# checkpoint:
#   save_dir: /path/to/model/save/dir
fairseq-hydra-train_semi \
    task.data=$data_dir \
    model.w2v_path=$pretrained_model \
    distributed_training.distributed_world_size=$ngpu \
    --config-dir $config_dir \
    --config-name $config_name
