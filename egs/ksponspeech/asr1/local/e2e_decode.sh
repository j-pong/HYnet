#!/bin/bash

cd /home/Workspace/HYnet/egs/ksponspeech/asr1

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=0
stop_stage=100
backend=pytorch
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=1
debugmode=1
dumpdir=dump   # directory to dump full features

# feature configuration
do_delta=false

preprocess_config=  #conf/randspec.yaml
train_config=conf/train_KT.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/tuning/lm_KT.yaml
decode_config=conf/decode_KT.yaml

lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
use_lm=false

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="spec_rmix" # tag for managing experiments.
train_set=KsponSpeech_tr
test=e2e_test

feat_recog_dir=${dumpdir}/${test}/delta${do_delta}

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ $use_lm == "true" ]; then
    if [ -z ${lmtag} ]; then
        lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
    lmexpdir=exp/${lmexpname}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if [ ${lm_n_average} -eq 0 ]; then
            lang_model=rnnlm.model.best
        else
            if ${use_lm_valbest_average}; then
                lang_model=rnnlm.val${lm_n_average}.avg.best
                opt="--log ${lmexpdir}/log"
            else
                lang_model=rnnlm.last${lm_n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${lmexpdir}/snapshot.ep.* \
                --out ${lmexpdir}/${lang_model} \
                --num ${lm_n_average}
        fi
    fi

    echo "Decoding Starts"
    pids=() # initialize pids
    for rtask in ${test}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        if [ $use_lm == "true" ]; then
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}  \
                --rnnlm ${lmexpdir}/${lang_model} \
                --api v2
        else
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}  \
                --api v2
        fi

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"

    decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
    . local/CER_WER.sh --dir ${expdir}/${decode_dir}

    ref=`cat ref.txt`
    hyp=`cat hyp.txt`
    CER_WER=`cat CER_WER.txt`
    echo ${hyp}/${CER_WER} | sed 's/\n//g'
    echo ${hyp}/${CER_WER} | sed 's/\n//g' > E2E_RESULT

fi
