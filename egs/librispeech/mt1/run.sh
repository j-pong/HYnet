#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
pseudo_set="train_860_pseudo"
answer_set="train_860"
valid_set="dev"
test_sets="test_860"
#test_sets="test_clean test_other dev_clean dev_other"
bpe_train_text="data/bpe_train/text"
bpe_train_text_small="data/train_clean_100/text"

#mt_config=conf/tuning/train_mt_rnn_beam_dec.yaml
mt_config=conf/tuning/train_mt_rnn_3090.yaml
#mt_config=conf/tuning/train_mt_rnn_2080conf_org.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_mt.yaml

#utils/combine_data.sh data/train_860_pseudo \
#data/train_clean_360_pseudo data/train_other_500_pseudo

# --lm_config "${lm_config}" \
# --pretrain_step 0 \
# --speed_perturb_factors "0.9 1.0 1.1" \

./mt.sh \
    --stage 1 \
    --lang en \
    --ngpu 3 \
    --inference_nj 72 \
    --nbpe 5000 \
    --max_text_length 100 \
    --mt_config "${mt_config}" \
    --use_lm false\
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --pseudo_set "${pseudo_set}" \
    --answer_set "${answer_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text data/local/other_text/text" \
    --bpe_train_text "${bpe_train_text}" "$@" \
    --bpe_train_text_small "${bpe_train_text_small}"
