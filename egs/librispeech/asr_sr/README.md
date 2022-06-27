# HYnet
Conventional / End-to-end / Representation Learning Combined Toolkit

## Make augment 860 
```bash
python ./mk_augment_data/aug_rir.py
```

## fine-tuning
```bash
./w2v_uda_finetune.sh --data_dir ./data/fairseq/train_aug_ul860/ --pretrained_model ./models/public_fairseq_models/960h_pretrained_no_finetuned/libri960_big.pt --config_name ft_w2v_large_l100h_ul860h_uda_aug_conf1
```