#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev_clean"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_transformer3.yaml
inference_config=conf/decode_asr.yaml

# ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.
ext=flac

# valid should be set to some reasonable percentage (like 0.01) of training data to use for validation.
# To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv
# with a separately pre-processed manifest file.
valid=0

./asr_fair.sh \
    --lang en \
    --ngpu 1 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train_960/text" "$@" \
    --valid "${valid}" \
    --ext "${ext}" \
    --w2v_model_url "https://dl.fbaipublicfiles.com/fairseq/wav2vec/" \
    --w2v_model_name "wav2vec_vox_100h_new.pt" \
    --w2v_model_path "downloads/wav2vec_pretrained_models" \
    --config_name vox_100h
