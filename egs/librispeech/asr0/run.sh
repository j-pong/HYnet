#!/bin/bash

# Copyright 2020 j-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
dumpdir=dump
resume=
unsup_resume=

# feature configuration
do_delta=false
preprocess_config=
train_config=conf/train.yaml
train_unsup_config=conf/train_unsup.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

datadir=/export/a15/vpanayotov/data

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_set=train_100
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

    # download the LM resources
    local/download_lm.sh $lm_url data/local/lm
    # when the "--stage 3" option is used below we skip the G2P steps, and use the
    # lexicon we have already downloaded from openslr.org/11/
    local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp
    utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
    local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
    utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
    utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_fglarge
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
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

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set}_org data/train_clean_100 # data/train_clean_360 data/train_other_500
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/dev_clean data/dev_other

    # prepare gmm training
    utils/subset_data_dir.sh --shortest data/${train_set}_org 2000 data/train_2kshort
    utils/subset_data_dir.sh data/${train_set}_org 5000 data/train_5k
    utils/subset_data_dir.sh data/${train_set}_org 10000 data/train_10k

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage \
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    utils/create_split_dir.pl \
        /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
        ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    steps/train_mono.sh --boost-silence 1.25 --nj 32 --cmd "$train_cmd" \
                      data/train_2kshort data/lang_nosp exp/mono

    # decode using the monophone model
    (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/mono exp/mono/graph_nosp_tgsmall
    for test in test_clean test_other dev_clean dev_other; do
        steps/decode.sh --nj 32 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
                      data/$test exp/mono/decode_nosp_tgsmall_$test
    done
    )&
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

    # train a first delta + delta-delta triphone system on a subset of 5000 utterances
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1

    # decode using the tri1 model
    (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/tri1 exp/tri1/graph_nosp_tgsmall
    for test in test_clean test_other dev_clean dev_other; do
        steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
                      data/$test exp/tri1/decode_nosp_tgsmall_$test
        steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
                         data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
        steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
    done
    )&
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

if [ -z ${tag} ]; then
    expname=${train_set}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend pytorch \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode 1 \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches 0 \
        --verbose 0 \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
