#!/bin/bash
# Copyright 2020 j-pong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
ngpu=4               # The number of gpus ("0" uses cpu, otherwise use gpu).
ngpu_id=0,1,2,3
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands
resume=false

# Imgr model related
imgr_tag=
imgr_exp=
imgr_config=conf/train.yaml
imgr_decode_config=conf/decode.yaml
imgr_args=

# Feature extraction related
feats_type=raw         # Feature type.

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Set tag for naming of model directory
if [ -z "${imgr_tag}" ]; then
    if [ -n "${imgr_config}" ]; then
        imgr_tag="$(basename "${imgr_config}" .yaml)_${feats_type}"
    else
        imgr_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${imgr_args}" ]; then
        imgr_tag+="$(echo "${imgr_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for training commands
if [ -z "${imgr_exp}" ]; then
    imgr_exp="${expdir}/imgr_${imgr_tag}"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: IMGR Training"

    _opts=
    if [ -n "${imgr_config}" ]; then
        _opts+="--config ${imgr_config} "
    fi

    mkdir -p "${imgr_exp}"

    log "IMGR training started... log: '${imgr_exp}/train.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${imgr_exp})"
    else
        jobname="${imgr_exp}/train.log"
    fi

    # shellcheck disable=SC2086
    ${python} -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${jobname}" \
        --log "${imgr_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${imgr_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        CUDA_VISIBLE_DEVICES=${ngpu_id} ${python} -m hynet.bin.imgr_train \
                                                  --resume ${resume} \
                                                  --output_dir "${imgr_exp}" \
                                                  ${_opts} ${imgr_args}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: IMGR Inference"

    _opts=
    if [ -n "${imgr_decode_config}" ]; then
        _opts+="--config ${imgr_decode_config} "
    fi
    mkdir -p "${imgr_exp}"

    log "IMGR inference started... log: '${imgr_exp}/decode.log'"
    if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
        # SGE can't include "/" in a job name
        jobname="$(basename ${imgr_exp})"
    else
        jobname="${imgr_exp}/decode.log"
    fi

    # shellcheck disable=SC2086
    ${cuda_cmd} --gpu 1 "${imgr_exp}"/decode.log \
        CUDA_VISIBLE_DEVICES=0 ${python} -m hynet.bin.imgr_inference \
            --ngpu 1 \
            --output_dir "${imgr_exp}" \
            ${_opts} ${imgr_args}

fi

log "Successfully finished. [elapsed=${SECONDS}s]"
