#!/bin/bash

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
skip_train=false     # Skip training stages
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes
nj=32                # The number of parallel jobs.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Ingr model related
imgr_tag=
imgr_exp=
imgr_config=
imgr_args= 

# Feature extraction related
feats_type=img         # Feature type (raw or fbank_pitch).

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

# if [ $# -ne 0 ]; then
#     log "Error: No positional arguments are required."
#     exit 2
# fi

. ./path.sh
. ./cmd.sh

# # Check required arguments
# [ -z "${train_set}" ]; log "Error: --train_set is required"; exit 2;
# [ -z "${valid_set}" ]; log "Error: --valid_set is required"; exit 2;
# [ -z "${test_sets}" ]; log "Error: --test_sets is required"; exit 2;

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

# The directory used for collect-stats mode
imgr_stats_dir="${expdir}/asr_stats_${feats_type}"
# The directory used for training commands
if [ -z "${imgr_exp}" ]; then
    imgr_exp="${expdir}/imgr_${imgr_tag}"
fi


if ! "${skip_train}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: IMGR collect stats"

        _opts=
        if [ -n "${imgr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${imgr_config} "
        fi

        _logdir="${imgr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        ${train_cmd} JOB=1:"${nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m hynet.bin.imgr_train \
                --collect_stats true \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${imgr_args}

        _opts=
        for i in $(seq "${nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${imgr_stats_dir}"
    fi


    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "Stage 2: IMGR Training"

        _opts=
        if [ -n "${imgr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.asr_train --print_config --optim adam
            _opts+="--config ${imgr_config} "
        fi

        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${asr_stats_dir}/train/feats_stats.npz "
        fi

        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${asr_stats_dir}/splits${num_splits_asr}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_asr_train_dir}/${_scp}" \
                      "${_asr_train_dir}/text" \
                      "${asr_stats_dir}/train/speech_shape" \
                      "${asr_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_asr}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_asr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${asr_stats_dir}/train/text_shape.${token_type} "
        fi

        log "Generate '${asr_exp}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${asr_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${asr_exp}/run.sh"; chmod +x "${asr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ASR training started... log: '${asr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${asr_exp})"
        else
            jobname="${asr_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${asr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${asr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.asr_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_asr_valid_dir}/text,text,text" \
                --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
                --valid_shape_file "${asr_stats_dir}/valid/text_shape.${token_type}" \
                --resume true \
                --fold_length "${_fold_length}" \
                --fold_length "${asr_text_fold_length}" \
                --output_dir "${asr_exp}" \
                ${_opts} ${imgr_args}

    fi
else
    log "Skip the training stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
