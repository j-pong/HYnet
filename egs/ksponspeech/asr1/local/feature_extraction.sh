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

preprocess_config=conf/randspec.yaml
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
test=test
is_kspon=true

feat_recog_dir=${dumpdir}/${test}/delta${do_delta}

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ]; then
    echo "stage 0: data preparation"
    sudo rm -rf dump/e2e_test data/e2e_test data/kaldi_test fbank mfcc

    find data/${test}/wave -name '*.wav' | awk -F '/' '{print $NF}' | sed 's/.wav//g' > data/${test}/uttid.scp
    find data/${test}/wave -name '*.wav' > data/${test}/wpath.scp
    cat data/${test}/uttid.scp | awk -F '_' '{print $1}' > data/${test}/spkid.scp
    paste data/${test}/uttid.scp data/${test}/spkid.scp | sed 's/\t/ /g' | sort -u > data/${test}/utt2spk
    echo "created utt2spk"
    utils/utt2spk_to_spk2utt.pl <data/${test}/utt2spk >data/${test}/spk2utt || exit 1
    echo "created spk2utt"
    paste data/${test}/uttid.scp data/${test}/wpath.scp | sed 's/\t/ /g' | sort -u > data/${test}/wav.scp
    echo "created wav.scp"

    rm -rf data/${test}/text data/${test}/text.txt data/${test}/text_tmp.txt
    if [[ $is_kspon == true ]]; then
        while read line; do
            cat data/KsponSpeech_tt/text | grep $line >> data/${test}/text_tmp.txt
        done < data/${test}/uttid.scp
        cat data/${test}/text_tmp.txt | awk -F '_' '{print $3"_"$4}' | sort -u > data/${test}/text
        cat data/${test}/text | awk -F ' ' '{$1=""; print $0}' | sed 's/^ //g' >> data/${test}/text.txt
        rm -rf data/${test}/texts; mkdir -p data/${test}/texts; sudo chmod -R +777 data/${test}/texts
        while read line; do
            tname=`echo $line | awk -F ' ' '{print $1}'  | sed 's/^ //g'`
            tcontent=`echo $line | awk -F ' ' '{$1=""; print $0}'  | sed 's/^ //g'`
            echo ${tcontent} > data/${test}/texts/${tname}.txt
        done < data/${test}/text
        echo "created text"
    else
        find data/${test}/texts -name '*.txt' | awk -F '/' '{print $NF}' | sed 's/.txt//g' > data/${test}/uttid.scp
        find data/${test}/texts -name '*.txt' > data/${test}/tpath.scp
        python3 local/inference.py ${test}
        paste data/${test}/uttid.scp data/${test}/text.txt | sort -u > data/${test}/text_tmp
        cat data/${test}/text_tmp | sed 's/\t/ /g' > data/${test}/text
        cat data/${test}/text_tmp | awk -F '\t' '{print $2}' > data/${test}/text.txt
        echo "created text"
    fi
    rm -rf data/${test}/uttid.scp data/${test}/tpath.scp data/${test}/text_tmp data/${test}/spkid.scp data/${test}/uttid.scp data/${test}/wpath.scp

    mkdir -p data/e2e_test data/kaldi_test
    cp -r data/${test}/* data/e2e_test/
    cp -r data/${test}/* data/kaldi_test/
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    mfccdir=mfcc
    for part in kaldi_test; do
        steps/make_mfcc.sh --cmd "$train_cmd" --nj ${nj} data/$part exp/make_mfcc/$part $mfccdir
        steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
    done

     fbankdir=fbank
     # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
     for x in e2e_test; do
         steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
             data/${x} exp/make_fbank/${x} ${fbankdir}
         utils/fix_data_dir.sh data/${x}
     done

    # compute global CMVN
    for rtask in e2e_test; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Json Data Preparation"

    for rtask in e2e_test; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done

    local/splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
fi
