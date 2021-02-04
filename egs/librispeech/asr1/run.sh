#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_rnn.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

utils/combine_data.sh data/${train_set} data/train_clean_100 \
    data/train_clean_360 data/train_other_500

### ahead
./asr.sh \
    --asr_tag train_960_asr_conformer \
    --use_lm true \
    --stage 10 \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text data/local/other_text/text" "$@"
