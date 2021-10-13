#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$PWD/data/fairseq/train_960/
ngpu=4
update_freq=$((128/$ngpu))

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

# make sure to set model save_dir path in wav2vec2_large_librivox.yaml
# checkpoint:
#   save_dir: /path/to/model/save/dir
fairseq-hydra-train \
    task.data=$data_dir \
    distributed_training.distributed_world_size=$ngpu +optimization.update_freq='['$update_freq']' \
    --config-dir $PWD/../../../tools/fairseq/examples/wav2vec/config/pretraining \
    --config-name wav2vec2_large_librivox
