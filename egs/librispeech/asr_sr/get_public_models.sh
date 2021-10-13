mkdir -p models/public_fairseq_models/960h_pretrained_no_finetuned public_fairseq_models/960h_pretrained_100h_finetuned
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt -P models/public_fairseq_models/960h_pretrained_no_finetuned
wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_big_100h.pt -P models/public_fairseq_models/960h_pretrained_100h_finetuned
