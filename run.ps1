python moneynet/bin/unsup_train.py --batch-size 22 --accum-grad 4 --datamper 1 --opt adam `
                                   --ngpu 1 --ncpu 6 --pin-memory 0 `
                                   --self-train 0 --encoder-type conv1d `
                                   --indir dump --outdir exp/train97_srconly_conv1d_hrelay_eth4_adm_residual