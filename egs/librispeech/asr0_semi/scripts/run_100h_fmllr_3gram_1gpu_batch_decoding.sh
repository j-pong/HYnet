#!/bin/bash

# Copyright 2020 j-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=-1       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
ncore=8       # number of cores for calculating number of threads
nj=8
dumpdir=dump
resume=
unsup_resume=

# feature configuration
do_delta=false
preprocess_config=
train_config=conf/train.yaml
semi_config=conf/ICT.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

datadir=/DB/librispeech

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

# bpemode (unigram or bpe)
nbpe=0
bpemode=triphone

# exp tag
tag="" # tag for managing experiments.

# feature directory
mfccdir=mfcc
fmllrdir=fmllr

# gmm alignment directory
gmmdir=exp/tri4b

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_set="train_100"
train_dev=dev
recog_set="test_clean test_other dev_clean dev_other"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done

    # download the LM resources
    local/download_lm.sh $lm_url data/local/lm
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100; do
        # use underscore-separated names in data directories.
        local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
    done

    # when the "--stage 3" option is used below we skip the G2P steps, and use the
    # lexicon we have already downloaded from openslr.org/11/
    local/prepare_dict.sh --stage 3 --nj ${nj} --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp
    utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
    local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Language Model"
    # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
    utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
    utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_fglarge
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: MFCC Feature Generation"
    # MFCC feature extraction
    for part in dev_clean test_clean dev_other test_other train_clean_100; do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj ${nj} data/$part exp/make_mfcc/$part $mfccdir
        steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Monophone Training"
    # Make some small data subsets for early system-build stages.  Note, there are 29k
    # utterances in the train_clean_100 directory which has 100 hours of data.
    # For the monophone stages we select the shortest utterances, which should make it
    # easier to align the data from a flat start.

    # prepare gmm curriculum training
    utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
    utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
    utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k

    # train a monophone system
    steps/train_mono.sh --boost-silence 1.25 --nj ${nj} --cmd "$train_cmd" \
                    data/train_2kshort data/lang_nosp exp/mono
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: tri1 Triphone Training"
    steps/align_si.sh --boost-silence 1.25 --nj ${nj} --cmd "$train_cmd" \
                    data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

    # train a first delta + delta-delta triphone system on a subset of 5000 utterances
    steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: tri2b Triphone Training"
    steps/align_si.sh --nj ${nj} --cmd "$train_cmd" \
                    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k


    # train an LDA+MLLT system.
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
                            --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                            data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: tri3b Triphone Training"
    # Align a 10k utts subset using the tri2b model
    steps/align_si.sh  --nj ${nj} --cmd "$train_cmd" --use-graphs true \
                    data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k

    # Train tri3b, which is LDA+MLLT+SAT on 10k utts
    steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                    data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: tri4b Triphone Training"
    # align the entire train_clean_100 subset using the tri3b model
    steps/align_fmllr.sh --nj ${nj} --cmd "$train_cmd" \
    data/train_clean_100 data/lang_nosp \
    exp/tri3b exp/tri3b_ali_clean_100

    # train another LDA+MLLT+SAT system on the entire 100 hour subset
    steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                        data/train_clean_100 data/lang_nosp \
                        exp/tri3b_ali_clean_100 exp/tri4b
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: tri4b Triphone Decoding"
    # Now we compute the pronunciation and silence probabilities from training data,
    # and re-create the lang directory.
    steps/get_prons.sh --cmd "$train_cmd" \
                    data/train_clean_100 data/lang_nosp exp/tri4b
    utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                    data/local/dict_nosp \
                                    exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
                                    exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

    utils/prepare_lang.sh data/local/dict \
                        "<UNK>" data/local/lang_tmp data/lang
    local/format_lms.sh --src-dir data/lang data/local/lm

    utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
    utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

    # decode using the tri4b model with pronunciation and silence probabilities
    {
        utils/mkgraph.sh \
            data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall
        mkdir exp/tri4b/decode_tgsmall_train_clean_100 && cp exp/tri4b/trans.* exp/tri4b/decode_tgsmall_train_clean_100/
        for test in dev_clean dev_other test_clean test_other; do
            steps/decode_fmllr.sh --nj ${nj} --cmd "$decode_cmd" \
                                exp/tri4b/graph_tgsmall data/$test \
                                exp/tri4b/decode_tgsmall_$test
        done
    }
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Get Alignment"

    # align the train, test, dev set using the tri4b model
    for part in dev_clean test_clean dev_other test_other train_clean_100; do
        steps/align_fmllr.sh --nj ${nj} data/${part} data/lang exp/tri4b exp/tri4b_ali_${part}

        KALDI_ROOT=${KALDI_ROOT} python local/utt2tokenid.py \
            --data_dir data/${part} \
            --ali_dir exp/tri4b_ali_${part} \
            --ali_mdl exp/tri4b/final.mdl
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 10: Fmllr Feature Generation For Network Training"
    # Generate the fmllr features; by default 40-dimensional fmllr
    for x in train_clean_100 dev_clean dev_other test_clean test_other; do
        mkdir -p data/${x}_fmllr
        cp -r data/${x}/* data/${x}_fmllr
        steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
        --transform-dir exp/tri4b/decode_tgsmall_$x \
            data/${x}_fmllr data/${x} exp/tri4b exp/make_fmllr/${x} ${fmllrdir} || exit 1
        utils/fix_data_dir.sh data/${x}_fmllr
    done

    utils/combine_data.sh --extra_files 'utt2num_frames tokenid.scp' data/train_100_org data/train_clean_100_fmllr
    utils/combine_data.sh --extra_files 'utt2num_frames tokenid.scp' data/${train_dev}_org data/dev_clean_fmllr \
        data/dev_other_fmllr
    for train_part in ${train_set}; do
        mkdir -p data/${train_part}
        cp -r data/${train_part}_org/* data/${train_part}

        # compute global CMVN
        compute-cmvn-stats scp:data/${train_part}/feats.scp data/${train_part}/cmvn.ark
    done
    mkdir -p data/${train_dev}
    cp -r data/${train_dev}_org/* data/${train_dev}

    # dump
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}_fmllr/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

# get alignment sequence index
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Json"
    # make tokenid.scp, json file and filter recipes
    for part in ${train_set} ${train_dev}; do
        local/data2json.sh --feat ${dumpdir}/${part}/delta${do_delta}/feats.scp \
            data/${part} > ${dumpdir}/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json
    done

    for part in ${recog_set}; do
        local/data2json.sh --feat ${dumpdir}/${part}/delta${do_delta}/feats.scp \
            data/${part} > ${dumpdir}/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json
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

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "stage 12: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_ICT_train.py \
        --config ${train_config} \
        --config2 ${semi_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend pytorch \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode 1 \
        --debugdir ${expdir} \
        --minibatches 0 \
        --verbose 0 \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "stage 13: Decoding"
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
            --backend pytorch \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}
    fi

    nthreads=$[ncore*2]
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # TODO: multi-gpu needs to be applicable
        ngpu=1

        # set batchsize 0 to disable batch decoding
        run.pl --gpu ${ngpu} ${expdir}/${decode_dir}/log/decode.log \
            KALDI_ROOT=${KALDI_ROOT} asr_ICT_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend pytorch \
            --recog-json ${feat_recog_dir}/data_${bpemode}${nbpe}.json \
            --result-ark ${expdir}/${decode_dir}/data.ark \
            --model ${expdir}/results/${recog_model}  \
            --api v1

        # get decoded results
        local/decode_dnn.sh --num-threads $nthreads exp/tri4b/graph_tgsmall exp/tri4b_ali_${rtask} ${feat_recog_dir} ${expdir}/${decode_dir} || exit 1;
        local/score.sh --min-lmwt 4 --max-lmwt 23 data/${rtask} exp/tri4b/graph_tgsmall ${expdir}/${decode_dir} || exit 1;
        for x in ${expdir}/${decode_dir}; do
            [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh;
        done
    )
    done
    echo "Finished"
fi
