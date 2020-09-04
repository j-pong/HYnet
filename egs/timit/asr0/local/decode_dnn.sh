#!/bin/bash


# Copyright 2013    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Decode the DNN model. The [srcdir] in this script should be the same as dir in
# build_nnet_pfile.sh. Also, the DNN model has been trained and put in srcdir.
# All these steps will be done automatically if you run the recipe file run-dnn.sh

# Modified 2018 Mirco Ravanelli Univeristé de Montréal - Mila

# Modified 2020 King God General Emperor Prof. Jun Hyuk Chang

# Reading the options in the cfg file
./path.sh
./cmd.sh

## Begin configuration section
cmd=${decode_cmd}
num_threads=4
acwt=0.10
beam=13.0
latbeam=8.0
min_active=200
max_active=7000
max_mem=50000000
max_arcs=-1

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: steps/decode_dnn.sh [options] <graph-dir> <ali-dir> <data-dir> <decode-dir>"
   echo " e.g.: steps/decode_dnn.sh exp/tri4/graph data/test exp/tri4_ali exp/tri4_dnn/decode ark_file"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --num-threads <n>                        # number of threads to use, default 4."
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --beam <beam>                            # default 20.0"
   echo "  --latbeam <latbeam>                      # default 12.0"
   echo "  --min-active <min>                       # default 200"
   echo "  --max-active <max>                       # default 7000"
   echo "  --max-mem <memory>                       # default 50000000"
   echo "  --max-arcs <arcs>                        # default -1"
   exit 1;
fi

graphdir=$1
alidir=$2
data=$3
out_folder=$4

dir=`echo $out_folder | sed 's:/$::g'` # remove any trailing slash.
srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/log

arr_ck=($(ls $out_folder/*.ark))

nj=${#arr_ck[@]}

echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

JOB=1
for ck_data in "${arr_ck[@]}"
do
    finalfeats="ark,s,cs: cat $ck_data |"
    latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $alidir/final.mdl $graphdir/HCLG.fst "$finalfeats" "ark:|gzip -c > $dir/lat.$JOB.gz" &> $dir/log/decode.$JOB.log &
    JOB=$((JOB+1))
done
wait

# Copy the source model in order for scoring
cp $alidir/final.mdl $srcdir

exit 0;
