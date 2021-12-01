#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=data/fairseq/train_860
pl_data_dir=decoding_results/100h_finetuned_jpong3_conf_860pl/viterbi
save_data_dir=data/fairseq/train_860_pl_viterbi

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

wrd_prefix=hypo.word
ltr_prefix=hypo.units

temp_dir=`date +%s`temp
mkdir -p $temp_dir
cp -r $pl_data_dir/$wrd_prefix* $temp_dir/train_pl_temp.wrd
cp -r $pl_data_dir/$ltr_prefix* $temp_dir/train_pl_temp.ltr

python3 sort_decoded_data.py --wrd_file $temp_dir/train_pl_temp.wrd --ltr_file $temp_dir/train_pl_temp.ltr \
--wrd_save $temp_dir/train_pl.wrd --ltr_save $temp_dir/train_pl.ltr

mkdir -p $save_data_dir
cp -r $data_dir/* $save_data_dir/
rm -rf $save_data_dir/train.wrd; cp -r $temp_dir/train_pl.wrd $save_data_dir/train.wrd
rm -rf $save_data_dir/train.ltr; cp -r $temp_dir/train_pl.ltr $save_data_dir/train.ltr

rm -rf $temp_dir
