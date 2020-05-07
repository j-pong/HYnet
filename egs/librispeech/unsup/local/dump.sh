#!/bin/bash

# Copyright 2017 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "$0 $*"  # Print the command line for logging
. ./path.sh

cmd=run.pl
nj=1
verbose=0
compress=true
write_utt2num_frames=true
filetype='mat'  # mat or hdf5
help_message="Usage: $0 <ark> <logdir> <dumpdir>"

. utils/parse_options.sh

arkdir=$1
logdir=$2
dumpdir=$3

if [ $# != 3 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

mkdir -p ${logdir}
mkdir -p ${dumpdir}

dumpdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${dumpdir} ${PWD})

if ${write_utt2num_frames}; then
    write_num_frames_opt="--write-num-frames=ark,t:$dumpdir/utt2num_frames.JOB"
else
    write_num_frames_opt=
fi

# dump features
${cmd} JOB=1:${nj} ${logdir}/dump_unsup_feature.JOB.log \
    copy-feats.py --verbose ${verbose} --out-filetype ${filetype} \
        --compress=${compress} --compression-method=2 ${write_num_frames_opt} \
        ark:${arkdir}/data.JOB.ark ark,scp:${dumpdir}/feats.JOB.ark,${dumpdir}/feats.JOB.scp \
    || exit 1

# concatenate scp files
for n in $(seq ${nj}); do
    cat ${dumpdir}/feats.${n}.scp || exit 1;
done > ${dumpdir}/feats.scp || exit 1

if ${write_utt2num_frames}; then
    for n in $(seq ${nj}); do
        cat ${dumpdir}/utt2num_frames.${n} || exit 1;
    done > ${dumpdir}/utt2num_frames || exit 1
    rm ${dumpdir}/utt2num_frames.* 2>/dev/null
fi

# Write the filetype, this will be used for data2json.sh
echo ${filetype} > ${dumpdir}/filetype

# remove temp scps
if [ ${verbose} -eq 1 ]; then
    echo "Succeeded dumping features for training"
fi
