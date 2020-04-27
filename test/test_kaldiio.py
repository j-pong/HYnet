from kaldiio import ReadHelper

ali_path = 'exp/tri4b_ali_train_clean_100/ali.*.gz'
ark_path = 'dump/deltafalse/feats.*.ark'

with ReadHelper('ark: gunzip -c {} |'.format(ali_path)) as reader:
    for key, array in reader:
        print(key, array)

with ReadHelper('ark: gunzip -c {} |'.format(ark_path)) as reader:
    for key, array in reader:
        print(key, array)