python moneynet/bin/unsup_train.py --batch-size 16 --accum-grad 5 --datamper 1 --opt sgd `
                                   --ngpu 1 --ncpu 6 --pin-memory 1 `
                                   --self-train 1 --encoder-type conv1d --energy-threshold 10 `
                                   --indir dump --outdir exp/train97_conv1d_split_selftrain_cdim32_hdim1024_HEactivation_residual_pass