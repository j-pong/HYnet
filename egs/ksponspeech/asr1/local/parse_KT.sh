#!/bin/bash

path=$1
temp=data/temp
data_set=`ls $path/* | awk -F '/' '{print $NF}'`

for set in ${data_set}; do
    to_do=`echo $set | awk -F '.list' '{print $1}'`
    dest_dir=data/${to_do}; mkdir -p $dest_dir
    mkdir -p $temp

    cat $path/$set | awk -F '/' '{print "/DB/"$6"/"$7"/"$7"/"$8"/"$9}' > $temp/wav_text
    cat $temp/wav_text | awk -F '\t' '{print $1}' > $temp/wav_path.txt
    cat $temp/wav_text | awk -F '\t' '{print $2}' > $temp/text.txt
    cat $temp/wav_path.txt | awk -F '/' '{print $(NF-1)"_"$NF}' | sed 's/.wav//g' > $temp/utt_id.txt
    cat $temp/utt_id.txt | awk -F '_' '{print $1"_"$2}' > $temp/spk_id.txt

    paste $temp/utt_id.txt $temp/wav_path.txt > $temp/uswav.scp
    paste $temp/utt_id.txt $temp/spk_id.txt > $temp/usutt2spk

    sort $temp/uswav.scp > $dest_dir/wav.scp
    sort $temp/usutt2spk > $dest_dir/utt2spk
    # grep

    utils/utt2spk_to_spk2utt.pl $dest_dir/utt2spk > $dest_dir/spk2utt || exit 1

    rm -rf $temp
done
