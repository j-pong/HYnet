#!/bin/bash

# Copyright 2020 j-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
unsup_resume=

# feature configuration
do_delta=false

train_unsup_config=conf/train_unsup.yaml

preprocess_config=conf/specaug.yaml
train_config=conf/tuning/train_rnn.yaml

decode_config=conf/tuning/decode_rnn.yaml

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.
unsup_tag=

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_large_set=train_960
train_small_set=train_100
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done
fi

feat_tr_large_dir=${dumpdir}/${train_large_set}/delta${do_delta}; mkdir -p ${feat_tr_large_dir}
feat_tr_small_dir=${dumpdir}/${train_small_set}/delta${do_delta}; mkdir -p ${feat_tr_small_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_large_set}_org data/train_clean_100 data/train_clean_360 data/train_other_500
    utils/combine_data.sh --extra_files utt2num_frames data/${train_small_set}_org data/train_clean_100
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/dev_clean data/dev_other

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_large_set}_org data/${train_large_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_small_set}_org data/${train_small_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_large_set}/feats.scp data/${train_large_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_large_set}/feats.scp data/${train_large_set}/cmvn.ark exp/dump_feats/train ${feat_tr_large_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_small_set}/feats.scp data/${train_large_set}/cmvn.ark exp/dump_feats/train ${feat_tr_small_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_large_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_large_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/${train_large_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_large_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_large_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_large_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_large_set} ${dict} > ${feat_tr_large_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_tr_small_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_small_set} ${dict} > ${feat_tr_small_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${unsup_tag} ]; then
    expname=${train_large_set}_$(basename ${train_unsup_config%.*})
else
    expname=${train_large_set}_${unsup_tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

. ./path_fair.sh || exit 1;
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Unsupervised Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        KALDI_ROOT=${KALDI_ROOT} unsup_train.py \
        --config ${train_unsup_config} \
        --ngpu ${ngpu} \
        --backend pytorch \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${unsup_resume} \
        --train-json ${feat_tr_large_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
. ./path.sh || exit 1;

recog_model=model.loss.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Unsupervised Feature Generation"
    nj=4
    pids=() # initialize pids
    for rtask in ${train_small_set} ${train_dev} dev_clean test_clean dev_other test_other ; do
    (
        decode_dir=decode_${rtask}_${recog_model}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        feat_dump_dir=${dumpdir}/${rtask}/unsup; mkdir -p ${feat_dump_dir}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/feature.JOB.log \
            KALDI_ROOT=${KALDI_ROOT} unsup_recog.py \
            --ngpu 0 \
            --backend pytorch \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-ark ${expdir}/${decode_dir}/data.JOB.ark \
            --model ${expdir}/results/${recog_model}

        local/dump.sh --cmd "$train_cmd" --nj ${nj} \
        ${expdir}/${decode_dir} exp/dump_unsup_feats/${rtask} ${feat_dump_dir}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi

nj=32
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Unsupervised Representation Json Data Preparation"
    # make json labels
    for rtask in ${train_small_set} ${train_dev} dev_clean test_clean dev_other test_other; do
        feat_recog_dir=${dumpdir}/${rtask}/unsup
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${tag} ]; then
    expname=${train_small_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_small_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: E2E ASR training"
    # make json labels
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${dumpdir}/${train_small_set}/unsup/data_${bpemode}${nbpe}.json \
        --valid-json ${dumpdir}/${train_dev}/unsup/data_${bpemode}${nbpe}.json
fi