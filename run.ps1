python moneynet/bin/unsup_train.py --batch-size 16 --accum-grad 6 --datamper 1 /
                                   --ngpu 1 --ncpu 6 --pin-memory 0 /
                                   --self-train 0 --encoder-type conv1d /
                                   --indir dump --outdir exp/temperature0.01_srconly_pretrain_hiddenmask_hsrwithenergy_batchsizefix_conv1d