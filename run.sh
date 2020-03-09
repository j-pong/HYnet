#!/usr/bin/env bash
python moneynet/bin/unsup_train.py --ngpu 4 --batch-size 48 --accum-grad 2 \
                                   --ncpu 16 --datamper 1 --self-train 1 --pin-memory 1 \
                                   --indir dump --outdir exp/temperature0.01_srconly_pretrain_hiddenmask_hsrwithenergy