#!/bin/bash

dir=exp/KsponSpeech_tr_pytorch_train_rnn_specaug/decode_test_model.acc.best_decode_KT_

. utils/parse_options.sh || exit 1;

temp=temp

cat $dir/hyp.trn | sed 's/(.*)//g' > temp.txt
cat temp.txt | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' > hyp.txt
cat $dir/ref.trn | sed 's/(.*)//g' > temp.txt
cat temp.txt | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' > ref.txt

rm -rf temp.txt
python3 local/CER_WER.py
