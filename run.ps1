python moneynet/bin/unsup_train.py --batch-size 24 --accum-grad 4 --datamper 1 `
                                   --ngpu 1 --ncpu 6 --pin-memory 0 `
                                   --self-train 0 --encoder-type linear `
                                   --indir dump --outdir exp/temperature0.01_srconly_linear_energymask