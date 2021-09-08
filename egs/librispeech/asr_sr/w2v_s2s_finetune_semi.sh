#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
pseudo_set="train_860_pseudo"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_wav2vec_s2s_semi.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

./asr_semi.sh \
    --lang en \
    --ngpu 4 \
    --inference_nj 16 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --pseudo_set "${pseudo_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/train_960/text data/local/other_text/text" \
    --bpe_train_text "data/train_960/text" "$@"
