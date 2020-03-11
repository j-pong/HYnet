python moneynet/bin/unsup_train.py --batch-size 19 --accum-grad 5 --datamper 1 `
                                   --ngpu 1 --ncpu 6 --pin-memory 0 `
                                   --self-train 0 --encoder-type conv1d `
                                   --indir dump --outdir exp/temperature0.01_srconly_conv1d_energymask