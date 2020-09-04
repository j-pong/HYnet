#!/bin/bash

# Copyright 2020 seas2nada
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Copyright 2013 Bagher BabaAli,
#           2014-2017 Brno University of Technology (Author: Karel Vesely)
#
# TIMIT, description of the database:
# http://perso.limsi.fr/lamel/TIMIT_NISTIR4930.pdf
#
# Hon and Lee paper on TIMIT, 1988, introduces mapping to 48 training phonemes,
# then re-mapping to 39 phonemes for scoring:
# http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci
#

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
stage=0       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
ncore=8
dumpdir=dump
resume=
unsup_resume=

# feature configuration
do_delta=false
preprocess_config=
train_config=conf/tuning/train_rnn.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# model average realted (only for transformer)
n_average=1                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.

datadir=/DB/TIMIT/TIMIT

# bpemode (unigram or bpe)
nbpe=0
bpemode=triphone

# exp tag
tag="" # tag for managing experiments.

# feature directory
mfccdir=mfcc
fmllrdir=fmllr

# gmm alignment directory
gmmdir=exp/tri3

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
recog_set=test

numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000

feats_nj=10
train_nj=30
decode_nj=5

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

#timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU
timit=/DB/TIMIT/TIMIT

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    local/timit_data_prep.sh $timit || exit 1

    local/timit_prepare_dict.sh

    # Caution below: we remove optional silence by setting "--sil-prob 0.0",
    # in TIMIT the silence appears also as a word in the dictionary and is scored.
    utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
     data/local/dict "sil" data/local/lang_tmp data/lang

    local/timit_format_data.sh
fi

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for x in train dev test; do
      steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
    done

    echo ============================================================================
    echo "                     MonoPhone Training & Decoding                        "
    echo ============================================================================

    steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono

    utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/mono/graph data/dev exp/mono/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/mono/graph data/test exp/mono/decode_test

    echo ============================================================================
    echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
    echo ============================================================================

    steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
     data/train data/lang exp/mono exp/mono_ali

    # Train tri1, which is deltas + delta-deltas, on train data.
    steps/train_deltas.sh --cmd "$train_cmd" \
     $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1

    utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri1/graph data/dev exp/tri1/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri1/graph data/test exp/tri1/decode_test

    echo ============================================================================
    echo "                 tri2 : LDA + MLLT Training & Decoding                    "
    echo ============================================================================

    steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
      data/train data/lang exp/tri1 exp/tri1_ali

    steps/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2

    utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri2/graph data/dev exp/tri2/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri2/graph data/test exp/tri2/decode_test

    echo ============================================================================
    echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
    echo ============================================================================

    # Align tri2 system with train data.
    steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
     --use-graphs true data/train data/lang exp/tri2 exp/tri2_ali

    # From tri2 system, train tri3 which is LDA + MLLT + SAT.
    steps/train_sat.sh --cmd "$train_cmd" \
     $numLeavesSAT $numGaussSAT data/train data/lang exp/tri2_ali exp/tri3

    utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri3/graph data/train exp/tri3/decode_train

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri3/graph data/test exp/tri3/decode_test

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri3/graph data/dev exp/tri3/decode_dev

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri3/graph data/test exp/tri3/decode_test

    echo ====================================================================
    echo "                        SGMM2 Training                           "
    echo ====================================================================

    steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
     data/train data/lang exp/tri3 exp/tri3_ali

    echo ====================================================================
    echo "                        DBN-DNN Training                         "
    echo ====================================================================

    local/nnet/run_dnn.sh
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "stage 9: Get Alignment"

    dnn_pref=dnn4_pretrain-dbn_dnn
    # align the train, test, dev set using dbn-dnn model
    for part in dev test; do
        steps/nnet/align.sh --nj ${decode_nj} data-fmllr-tri3/train data/lang \
        exp/${dnn_pref} exp/${dnn_pref}_ali_${part}

        KALDI_ROOT=${KALDI_ROOT} python local/utt2tokenid.py \
            --data_dir data/${part} \
            --ali_dir exp/${dnn_pref}_ali_${part} \
            --ali_mdl exp/${dnn_pref}/final.mdl
    done
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 10: Data Dumping for Network Training"

    utils/combine_data.sh --extra_files 'utt2num_frames tokenid.scp' data-fmllr-tri3/${train_set}_org data-fmllr-tri3/${train_set}
    utils/combine_data.sh --extra_files 'utt2num_frames tokenid.scp' data-fmllr-tri3/${train_dev}_org data-fmllr-tri3/${train_dev}
    cp -r data-fmllr-tri3/${train_set}_org/* data-fmllr-tri3/${train_set}
    cp -r data-fmllr-tri3/${train_dev}_org/* data-fmllr-tri3/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data-fmllr-tri3/${train_set}/feats.scp data-fmllr-tri3/${train_set}/cmvn.ark

    # dump
    dump.sh --cmd "$train_cmd" --nj ${feats_nj} --do_delta ${do_delta} \
        data-fmllr-tri3/${train_set}/feats.scp data-fmllr-tri3/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${feats_nj} --do_delta ${do_delta} \
        data-fmllr-tri3/${train_dev}/feats.scp data-fmllr-tri3/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${feats_nj} --do_delta ${do_delta} \
            data-fmllr-tri3/${rtask}/feats.scp data-fmllr-tri3/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

# get alignment sequence index
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "stage 11: Json"
    # make tokenid.scp, json file and filter recipes
    for part in ${train_set} ${train_dev}; do
        local/data2json.sh --feat ${dumpdir}/${part}/delta${do_delta}/feats.scp \
            data-fmllr-tri3/${part} > ${dumpdir}/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json
    done

    for part in ${recog_set}; do
        local/data2json.sh --feat ${dumpdir}/${part}/delta${do_delta}/feats.scp \
            data-fmllr-tri3/${part} > ${dumpdir}/${part}/delta${do_delta}/data_${bpemode}${nbpe}.json
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
        asr_hyb_train.py \
        --config ${train_config} \
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

    nthreads=$[ncore*2]
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        n_model=`ls ${expdir}/results | grep snap | sed 's/snapshot.ep.//g' | sort -n | tail -1`

        ngpu=1

        # set batchsize 0 to disable batch decoding
        ${cuda_cmd} --gpu ${ngpu} ${expdir}/${decode_dir}/log/decode.log \
            KALDI_ROOT=${KALDI_ROOT} asr_hyb_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend pytorch \
            --batchsize 15 \
            --recog-json ${feat_recog_dir}/data_${bpemode}${nbpe}.json \
            --result-ark ${expdir}/${decode_dir}/data.ark \
            --model ${expdir}/results/snapshot.ep.${n_model}  \
            --api v1

        # get decoded results
        local/decode_dnn.sh --num-threads $nthreads exp/tri3/graph exp/tri3_ali_${rtask} ${feat_recog_dir} ${expdir}/${decode_dir} || exit 1;
        local/score.sh --min-lmwt 4 --max-lmwt 23 data/${rtask} exp/tri3/graph ${expdir}/${decode_dir} || exit 1;
        for x in ${expdir}/${decode_dir}; do
            [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh;
        done

    )
    done
    echo "Finished"
fi
