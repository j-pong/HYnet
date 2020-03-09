#!/usr/bin/env bash
python moneynet/bin/unsup_train.py --ngpu 4 --batch_size 97 --accum_grad 1 --ncpu 16 --indir dump --outdir exp/temperature0.01_srconly_pretrain_hiddenmask_hsrwithenergy