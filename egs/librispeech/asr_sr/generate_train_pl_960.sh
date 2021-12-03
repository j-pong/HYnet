#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

l100_dir=data/fairseq/train_clean_100
pl860_dir=data/fairseq/train_860_pl_viterbi
save_data_dir=data/fairseq/train_960_pl_viterbi

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

mkdir -p $save_data_dir
cp -r $l100_dir/* $save_data_dir
mv $save_data_dir/train.tsv $save_data_dir/train_org.tsv
cat $save_data_dir/train_org.tsv | sed 's#"/DB/LibriSpeech/LibriSpeech/train-clean-100"/"#DB/LibriSpeech/LibriSpeech/train-960"#g' > $save_data_dir/train.tsv
tail -n +2 $pl860_dir/train.tsv >> $save_data_dir/train.tsv
cat $pl860_dir/train.wrd >> $save_data_dir/train.wrd
cat $pl860_dir/train.ltr >> $save_data_dir/train.ltr

rm -rf $save_data_dir/train_org.tsv
