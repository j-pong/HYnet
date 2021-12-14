#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ul_data_path=/DB/LibriSpeech/LibriSpeech/train-860/
l_data_dir=$PWD/data/fairseq/train_clean_100/
save_ul_data_dir=$PWD/data/fairseq/train_860/
save_l_ul_data_dir=$PWD/data/fairseq/train_l100_ul860/

data_360h_dir=/DB/LibriSpeech/LibriSpeech/train-clean-360/
data_500h_dir=/DB/LibriSpeech/LibriSpeech/train-other-500/

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

# make train-860 DB set
sudo mkdir -p $ul_data_path
for dset in $data_360h_dir $data_500h_dir; do
    sudo cp -r $dset/* $ul_data_path/
done

# generate train files
mkdir -p $save_ul_data_dir
python $PWD/../../../tools/fairseq/examples/wav2vec/wav2vec_manifest.py $ul_data_path --dest $save_ul_data_dir --ext flac --valid 0

cp -r $l_data_dir/* $save_l_ul_data_dir
cp -r $save_ul_data_dir/train.tsv $save_l_ul_data_dir/ultrain.tsv
