#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

data_dir=datas/dev_clean
subset=`echo $data_dir | awk -F '/' '{print $NF}'`
pl_data_dir=temp_pl
save_data_dir=datas/generated_pl
finetuned_model=models/SR_checkpoint_save.pt

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

export PATH=$PWD/../../../tools/fairseq:$PATH
export PYTHONPATH=$PWD/../../../tools/fairseq:$PYTHONPATH
python $PWD/../../../tools/fairseq/examples/speech_recognition/infer.py $data_dir --task audio_pretraining \
--nbest 1 --path $finetuned_model --gen-subset $subset --results-path $pl_data_dir --w2l-decoder viterbi \
 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter

#wrd_prefix=hypo.word
#ltr_prefix=hypo.units
#
#temp_dir=`date +%s`temp
#mkdir -p $temp_dir
#cp -r $pl_data_dir/$wrd_prefix* $temp_dir/train_pl_temp.wrd
#cp -r $pl_data_dir/$ltr_prefix* $temp_dir/train_pl_temp.ltr
#
#python3 sort_decoded_data.py --wrd_file $temp_dir/train_pl_temp.wrd --ltr_file $temp_dir/train_pl_temp.ltr \
#--wrd_save $temp_dir/train_pl.wrd --ltr_save $temp_dir/train_pl.ltr
#
#mkdir -p $save_data_dir
#cp -r $data_dir/* $save_data_dir/
#rm -rf $save_data_dir/train.wrd; cp -r $temp_dir/train_pl.wrd $save_data_dir/train.wrd
#rm -rf $save_data_dir/train.ltr; cp -r $temp_dir/train_pl.ltr $save_data_dir/train.ltr
#
#rm -rf $temp_dir
#rm -rf $pl_data_dir
