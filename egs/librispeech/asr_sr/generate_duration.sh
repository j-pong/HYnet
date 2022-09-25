#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

audio_data_dir=/DB/LibriSpeech/dev-other
audio_extension=flac

data_dir=$PWD/datas/dev_other
finetuned_model=$PWD/models/SR_checkpoint_save.pt

. ./path.sh || exit 1;
. utils/parse_options.sh || exit 1;

subset=`echo $data_dir | awk -F '/' '{print $NF}'`

export PATH=$PWD/../../../tools/fairseq:$PATH
export PYTHONPATH=$PWD/../../../tools/fairseq:$PYTHONPATH
export FAIRSEQ_PATH=$PWD/../../../tools/fairseq
export MODELTYPE=wav2vec

# manifest tsv file
python $FAIRSEQ_PATH/examples/wav2vec/wav2vec_manifest.py $audio_data_dir --dest $data_dir --ext $audio_extension --valid 0
mv $data_dir/train.tsv $data_dir/$subset.tsv
cp -r datas/dict.ltr.txt $data_dir/

# generate label file
python $FAIRSEQ_PATH/examples/wav2vec/libri_labels.py $data_dir/$subset.tsv --output-dir $data_dir --output-name $subset

# generate duration file
python $PWD/segmentor/main.py \
    --config-dir $FAIRSEQ_PATH/examples/speech_recognition/new/conf --config-name infer \
    task=audio_pretraining task.data=$data_dir task.labels=ltr common.user_dir=$FAIRSEQ_PATH/examples/$MODELTYPE \
    decoding.type=viterbi dataset.gen_subset=$subset dataset.batch_size=1 \
    common_eval.path=$finetuned_model distributed_training.distributed_world_size=1

cp -r $PWD/None/$subset/train.dur $data_dir/$subset.dur
rm -rf $PWD/None
