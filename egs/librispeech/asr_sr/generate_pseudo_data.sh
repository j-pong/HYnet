#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

audio_data_dir=/DB/LibriSpeech/dev-clean
audio_extension=flac

data_dir=datas/dev_clean
pl_data_dir=datas/generated_pl
finetuned_model=models/SR_checkpoint_save.pt

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

subset=`echo $data_dir | awk -F '/' '{print $NF}'`

export PATH=$PWD/../../../tools/fairseq:$PATH
export PYTHONPATH=$PWD/../../../tools/fairseq:$PYTHONPATH

# manifest tsv file
python $PWD/../../../tools/fairseq/examples/wav2vec/wav2vec_manifest.py $audio_data_dir --dest $data_dir --ext $audio_extension --valid 0
cp -r datas/dict.ltr.txt $data_dir/
mv $data_dir/train.tsv $data_dir/$subset.tsv
cat $data_dir/$subset.tsv | awk -F '' '{print $NF}' | tail -n +2 > $data_dir/$subset.ltr

# viterbi decoding
python $PWD/PL_generation/main.py $data_dir --task audio_pretraining \
--nbest 1 --path $finetuned_model --gen-subset $subset --results-path temp_pl --w2l-decoder viterbi \
 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter --quiet

rm -rf $data_dir/$subset.ltr

# generate pseudo label files
wrd_prefix=hypo.word
ltr_prefix=hypo.units

temp_dir=`date +%s`temp
mkdir -p $temp_dir
cp -r temp_pl/$wrd_prefix* $temp_dir/train_pl_temp.wrd
cp -r temp_pl/$ltr_prefix* $temp_dir/train_pl_temp.ltr

python3 $PWD/PL_generation/sort_decoded_data.py --wrd_file $temp_dir/train_pl_temp.wrd --ltr_file $temp_dir/train_pl_temp.ltr \
--wrd_save $temp_dir/generated_pl.wrd --ltr_save $temp_dir/generated_pl.ltr

mkdir -p $pl_data_dir
cp -r $data_dir/* $pl_data_dir/
rm -rf $pl_data_dir/generated_pl.wrd; cp -r $temp_dir/generated_pl.wrd $pl_data_dir/generated_pl.wrd
rm -rf $pl_data_dir/generated_pl.ltr; cp -r $temp_dir/generated_pl.ltr $pl_data_dir/generated_pl.ltr

rm -rf $temp_dir
rm -rf temp_pl
