python moneynet/bin/unsup_train.py --batch-size 15 --accum-grad 6 --datamper 1 --opt sgd `
                                   --ngpu 1 --ncpu 6 --pin-memory 1 `
                                   --self-train 1 --encoder-type conv1d --energy-threshold 10 `
                                   --indir dump --outdir exp/train97_conv1d_split_eth10_residual