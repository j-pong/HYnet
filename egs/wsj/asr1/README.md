<!-- Generated by ./scripts/utils/show_asr_result.sh -->
# RESULTS
## Update only RNN-LM:  [Transformer](./conf/tuning/train_asr_transformer2.yaml) with [Char-LM](./conf/tuning/train_lm_adam_layers4.yaml)
### Environments
- date: `Mon Aug 24 11:52:54 JST 2020`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.0`
- pytorch version: `pytorch 1.6.0`
- Git hash: `e7d278ade57d8fba9b4f709150c4c499c75f53de`
  - Commit date: `Mon Aug 24 09:45:54 2020 +0900`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|8234|93.3|5.9|0.8|0.8|7.5|58.1|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|5643|95.8|4.0|0.3|0.7|5.0|44.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_dev93|503|48634|97.6|1.1|1.3|0.6|3.0|62.8|
|inference_lm_lm_train_lm_char_optimadam_batch_size512_optim_conflr0.0005_lm_confnlayers4_valid.loss.ave_asr_model_valid.acc.ave/test_eval92|333|33341|98.5|0.7|0.8|0.5|2.0|53.2|


## FBANK without pitch, [Transformer, bs=32, accum_grad=8, warmup_steps=30000, 100epoch](./conf/tuning/train_asr_transformer2.yaml) with [Char-LM](./conf/tuning/train_lm_adagrad.yaml)
### Environments
- date: `Mon Mar 23 18:20:34 JST 2020`
- python version: `3.7.5 (default, Oct 25 2019, 15:51:11)  [GCC 7.3.0]`
- espnet version: `espnet 0.7.0`
- pytorch version: `pytorch 1.4.0`
- Git hash: `8f3a0ff172dac2cb887878cda42a918737df8b91`
  - Commit date: `Wed Mar 18 10:41:54 2020 +0900`

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|503|8234|92.2|6.9|0.9|1.3|9.1|62.2|
|decode_test_eval92_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|333|5643|95.1|4.6|0.3|0.8|5.7|49.2|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|503|48634|97.0|1.3|1.6|0.8|3.7|67.4|
|decode_test_eval92_decode_lm_train_char_valid.loss.best_asr_model_valid.acc.ave|333|33341|98.2|0.8|0.9|0.6|2.3|56.5|


## FBANK without pitch, [Transformer, bs=32, accum_grad=8, warmup_steps=60000, 200epoch](./conf/tuning/train_asr_transformer.yaml) with [Char-LM](./conf/tuning/train_lm_adagrad.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.ave|503|8234|92.2|6.9|0.9|1.1|8.9|63.2|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.ave|333|5643|94.3|5.3|0.4|1.0|6.7|54.7|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.ave|503|48634|97.1|1.4|1.5|0.7|3.6|67.6|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.ave|333|33341|98.1|1.0|1.0|0.7|2.6|61.6|


## FBANK without pitch, [VGG-BLSTMP](./conf/tuning/train_asr_rnn.yaml) with [Char-LM](./conf/tuning/train_lm_adagrad.yaml)
### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.best|503|8234|90.9|8.0|1.1|1.5|10.6|66.8|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.best|333|5643|94.1|5.3|0.6|1.0|6.9|54.4|

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_test_dev93decode_lm_valid.loss.best_asr_model_valid.acc.best|503|48634|96.5|1.8|1.7|0.9|4.4|69.8|
|decode_test_eval92decode_lm_valid.loss.best_asr_model_valid.acc.best|333|33341|97.8|1.1|1.0|0.7|2.9|62.2|