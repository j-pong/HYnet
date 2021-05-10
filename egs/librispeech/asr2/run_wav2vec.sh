#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets="test_clean test_other"

asr_config=conf/tuning/train_asr_wav2vec_ctc.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

#utils/combine_data.sh data/train_860_pseudo \
#data/train_clean_360_pseudo data/train_other_500_pseudo

# --lm_config "${lm_config}" \
# --pretrain_step 0 \
# --speed_perturb_factors "0.9 1.0 1.1" \

./asr_wav2vec.sh \
    --stage 5 \
    --lang en \
    --ngpu 1 \
    --inference_nj 8 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false\
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "data/train_960/text" "$@"
