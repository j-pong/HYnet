#!/bin/bash

# Copyright 2020 j-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
resume=

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

expdir=exp/train90
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Feature Generation"
    python moneynet/utils/compliance/librosa/make_feats.py --indir dump --outdir ${expdir} --datadir data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Training"
    python moneynet/bin/unsup_train.py --ngpu 4 --batch-size 90 --accum-grad 1 --lr 0.001 --grad-clip 5 \
                                       --ncpu 28 --datamper 4 --pin-memory 0 \
                                       --self-train 1 --encoder-type conv1d --temperature 0.08 \
                                       --indir dump --outdir ${expdir} \
                                       --resume ${resume}
fi
