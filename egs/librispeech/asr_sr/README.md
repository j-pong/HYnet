# HYnet_NST
Conventional / End-to-end / Representation Learning Combined Toolkit

## Make Augment dataset
### Data augmentation
```bash
# simple rir augmentation using np.convolve
# If you want to augment using kaldi-based augmentation using musan-dataset, contact JS
python mk_augment_data/aug_100h_rir_ratio.py --train_aug_100 ./data/fairseq/train_aug100_half --train_aug_DB /DB/LibriSpeech/LibriSpeech/train_aug100_half --aug_ratio 50   
```



## Make Pseudo label
### 1. Inference 860-unlabeled dataset
```bash
./w2v_ctc_infer.sh --data_dir [libri 860h] --subset train --finetuned_model [model path] --inference_result [inference path]

./w2v_ctc_infer.sh --data_dir ./data/fairseq/train_860/ --subset train --finetuned_model ./models/100h_finetuned_rir_aug_quarter_jpong3_conf/checkpoints/checkpoint_best.pt --inference_result 100h_finetunning_infer/100h_finetuned_rir_aug_quarter_jpong3_conf
```

### 2. Pseudo label generation for 860 dataset
```bash
./generate_pseudo_data.sh --pl_data_dir [inference path] --save_data_dir [save 860h data dir]
./generate_pseudo_data.sh --pl_data_dir 100h_finetunning_infer/100h_finetuned_rir_aug_quarter_jpong3_conf --save_data_dir ./data/fairseq/NST1_aug_quarter_ft_860_pl_viterbi
```

### 3. make 960h dataset
```bash
./generate_train_pl_960h.sh --pl860_dir [save 860h data dir] ----save_data_dir [save 960h data dir]
./generate_train_pl_960h.sh --pl860_dir ./data/fairseq/NST1_aug_quarter_ft_860_pl_viterbi ----save_data_dir ./data/fairseq/NST1_aug_quarter_ft_l100_pl860_viterbi

```


### Make Pseudo label with LM
```bash
bash get_public_lm.sh
bash 
```