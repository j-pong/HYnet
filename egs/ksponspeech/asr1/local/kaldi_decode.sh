#!/usr/bin/env bash

cd /home/Workspace/HYnet/egs/ksponspeech/asr1

stage=-1
mfccdir=mfcc
dir=exp/chain_KT/tdnn_1a_sp
test=kaldi_test
graph_dir=${dir}/graph
nj=1

. ./cmd.sh
. ./path.sh
. parse_options.sh

set -e

if [ $stage -le 2 ]; then
    echo 'stage 2: start decoding'
    echo `date`
    rm -rf exp/chain_KT/tdnn_1a_sp/decode11_$test*

    for test_set in $test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj ${nj} --cmd "$decode_cmd" --skip-scoring true \
      $graph_dir data/${test_set} $dir/decode11_${test_set} || exit 1;
    done
fi

if [ $stage -le 3 ]; then
    echo 'stage 3: start getting results'
    scoring_opts=
    prefix=_cer
    mkdir -p exp/chain_KT/tdnn_1a_sp/decode11_$test$prefix
    cp -r exp/chain_KT/tdnn_1a_sp/decode11_$test/* exp/chain_KT/tdnn_1a_sp/decode11_$test$prefix

    # getting results (see RESULTS file)
    local/score.sh --word-ins-penalty 0.0 \
        data/$test exp/chain_KT/tdnn_1a_sp/graph exp/chain_KT/tdnn_1a_sp/decode11_$test || exit 1;

    local/score_kaldi_cer_without_white_space.sh --word-ins-penalty 0.0 \
        $scoring_opts data/$test exp/chain_KT/tdnn_1a_sp/graph exp/chain_KT/tdnn_1a_sp/decode11_$test$prefix || exit 1;

    hyp=`cat exp/chain_KT/tdnn_1a_sp/decode11_$test/scoring_kaldi/wer_details/per_utt | grep hyp | sed 's/ \*\*\*//g' | awk -F ' ' '{$1=""; print $0}' | sed 's/hyp //g' | sed 's/^ //g'`
    wer=`for x in exp/chain_KT/tdnn_1a_sp/decode11_$test; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh; done | awk -F 'WER ' '{print $2}' | awk -F ' ' '{print $1}'`
    cer=`cat exp/chain_KT/tdnn_1a_sp/decode11_${test}_cer/scoring_kaldi/best_cer | sed 's/WER/CER/g' | awk -F 'CER ' '{print $2}' | awk -F ' ' '{print $1}'`

    echo $hyp/$cer/$wer | sed 's/\n//g'
    echo $hyp/$cer/$wer | sed 's/\n//g' > KALDI_RESULT
    echo `date`
fi
