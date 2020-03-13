#!/bin/bash

# Copyright 2020 Hanyang University (j-pong)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

stage=0
stop_stage=100

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

expdir=exp/train97_linear_retest
mkdir -p ${expdir}
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Feature Generation"
    python moneynet/utils/compliance/librosa/make_feats.py --indir dump --outdir ${expdir} --datadir data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Network Training"
    python moneynet/bin/unsup_train.py --ngpu 4 --batch-size 90 --accum-grad 1 \
                                       --ncpu 28 --datamper 1 --self-train 1 --encoder-type linear --pin-memory 0 \
                                       --indir dump --outdir ${expdir}
fi