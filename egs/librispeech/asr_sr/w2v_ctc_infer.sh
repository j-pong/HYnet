#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=$PWD/data/fairseq/dev_other/
finetuned_model=$PWD/models/finetuned_w2v/checkpoint_best.pt
inference_result=$PWD/fairseq_results/finetuned_w2v
word_score=-1

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

subset=dev_other
python $PWD/../../../tools/fairseq/examples/speech_recognition/infer.py $data_dir --task audio_pretraining \
--nbest 1 --path $finetuned_model --gen-subset $subset --results-path inference_result --w2l-decoder viterbi \
 --word-score $word_score --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter
