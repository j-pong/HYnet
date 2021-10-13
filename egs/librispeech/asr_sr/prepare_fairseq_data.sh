#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

ul_data_path=/DB/LibriSpeech/LibriSpeech/train-960/
l_data_path=/DB/LibriSpeech/LibriSpeech/train-clean-100/

save_ul_data_dir=$PWD/data/fairseq/train_960/
save_l_data_dir=$PWD/data/fairseq/train_clean_100/

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

python $PWD/../../../tools/fairseq/examples/wav2vec/wav2vec_manifest.py $ul_data_path --dest $save_ul_data_dir --ext flac --valid 0.01
python $PWD/../../../tools/fairseq/examples/wav2vec/wav2vec_manifest.py $l_data_path --dest $save_l_data_dir --ext flac --valid 0

split=train
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt -P $save_l_data_dir

# Prepare all dev/test dataset
for dset in test_clean test_other dev_clean dev_other; do
    test_data_dir=/DB/LibriSpeech/LibriSpeech/${dset//_/-}
    save_test_data_dir=$PWD/data/fairseq/$dset
    python $PWD/../../../tools/fairseq/examples/wav2vec/wav2vec_manifest.py $test_data_dir --dest $save_test_data_dir --ext flac --valid 0

    cp -r $save_l_data_dir/dict.ltr.txt $save_test_data_dir/
    python $PWD/../../../tools/fairseq/examples/wav2vec/libri_labels.py $save_l_data_dir/train.tsv --output-dir $save_l_data_dir --output-name $split
    python $PWD/../../../tools/fairseq/examples/wav2vec/libri_labels.py $save_test_data_dir/train.tsv --output-dir $save_test_data_dir --output-name $dset
    mv $save_test_data_dir/train.tsv $save_test_data_dir/$dset.tsv
    cp -r $save_test_data_dir/* $save_l_data_dir
done
