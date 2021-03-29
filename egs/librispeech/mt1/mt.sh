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
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.
skip_upload=true     # Skip packing and uploading stages.
ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Text feature extraction related
min_text_length=1    # Minimum duration in second.
max_text_length=100  # Maximum duration in second.

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0  # character coverage when modeling BPE

# Language model related
use_lm=true       # Use language model for mt decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the direcotry path for LM experiment.
                  # If this option is specified, lm_tag is ignored.
lm_stats_dir=     # Specify the direcotry path for LM statistics.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# mt model related
mt_tag=       # Suffix to the result dir for mt model training.
mt_exp=       # Specify the direcotry path for mt experiment.
               # If this option is specified, mt_tag is ignored.
mt_stats_dir= # Specify the direcotry path for mt statistics.
mt_config=    # Config for mt model training.
mt_args=      # Arguments for mt model training, e.g., "--max_epoch 10".
               # Note that it will overwrite args in mt config.
feats_normalize=global_mvn # Normalizaton layer type.
num_splits_mt=1           # Number of splitting for lm corpus.

# Decoding related
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
inference_args=   # Arguments for decoding, e.g., "--lm_weight 0.1".
                  # Note that it will overwrite args in inference config.
inference_lm=valid.loss.ave.pth       # Language modle path for decoding.
inference_mt_model=valid.acc.ave.pth # mt model path for decoding.
                                      # e.g.
                                      # inference_mt_model=train.loss.best.pth
                                      # inference_mt_model=3epoch.pth
                                      # inference_mt_model=valid.acc.best.pth
                                      # inference_mt_model=valid.loss.ave.pth
download_model= # Download a model from Model Zoo and use it for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
pseudo_set=       # Name of pseudo labeled training set.
answer_set=       # Name of answer labeled training set for pseudo.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
bpe_train_text=  # Text file path of bpe training set.
bpe_train_text_small=  # Text file path of bpe small training set.
lm_train_text=   # Text file path of language model training set.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
mt_speech_fold_length=800 # fold_length for speech data during mt training.
mt_text_fold_length=150   # fold_length for text data during mt training.
lm_fold_length=150         # fold_length for LM training.

help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --skip_upload    # Skip packing and uploading stages (default="${skip_upload}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Text feature related
    --max_text_length # Maximum number of tokens in a sentence (default="${max_text_length}").
    --min_text_length # Minimum number of tokens in a sentence (default="${min_text_length}").


    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --nbpe                    # The number of BPE vocabulary (default="${nbpe}").
    --bpemode                 # Mode of BPE (unigram or bpe, default="${bpemode}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").
    --bpe_input_sentence_size # Size of input sentence for BPE (default="${bpe_input_sentence_size}").
    --bpe_nlsyms              # Non-linguistic symbol list for sentencepiece, separated by a comma. (default="${bpe_nlsyms}").
    --bpe_char_cover          # Character coverage when modeling BPE (default="${bpe_char_cover}").

    # Language model related
    --lm_tag          # Suffix to the result dir for language model training (default="${lm_tag}").
    --lm_exp          # Specify the direcotry path for LM experiment.
                      # If this option is specified, lm_tag is ignored (default="${lm_exp}").
    --lm_stats_dir    # Specify the direcotry path for LM statistics (default="${lm_stats_dir}").
    --lm_config       # Config for language model training (default="${lm_config}").
    --lm_args         # Arguments for language model training (default="${lm_args}").
                      # e.g., --lm_args "--max_epoch 10"
                      # Note that it will overwrite args in lm config.
    --use_word_lm     # Whether to use word language model (default="${use_word_lm}").
    --word_vocab_size # Size of word vocabulary (default="${word_vocab_size}").
    --num_splits_lm   # Number of splitting for lm corpus (default="${num_splits_lm}").

    # mt model related
    --mt_tag          # Suffix to the result dir for mt model training (default="${mt_tag}").
    --mt_exp          # Specify the direcotry path for mt experiment.
                       # If this option is specified, mt_tag is ignored (default="${mt_exp}").
    --mt_stats_dir    # Specify the direcotry path for mt statistics (default="${mt_stats_dir}").
    --mt_config       # Config for mt model training (default="${mt_config}").
    --mt_args         # Arguments for mt model training (default="${mt_args}").
                       # e.g., --mt_args "--max_epoch 10"
                       # Note that it will overwrite args in mt config.
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_mt   # Number of splitting for lm corpus  (default="${num_splits_mt}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --inference_args      # Arguments for decoding (default="${inference_args}").
                          # e.g., --inference_args "--lm_weight 0.1"
                          # Note that it will overwrite args in inference config.
    --inference_lm        # Language modle path for decoding (default="${inference_lm}").
    --inference_mt_model # mt model path for decoding (default="${inference_mt_model}").
    --download_model      # Download a model from Model Zoo and use it for decoding (default="${download_model}").

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).
    --bpe_train_text # Text file path of bpe training set.
    --bpe_train_text_small # Text file path of bpe small training set.
    --lm_train_text  # Text file path of language model training set.
    --lm_dev_text   # Text file path of language model development set (default="${lm_dev_text}").
    --lm_test_text  # Text file path of language model evaluation set (default="${lm_test_text}").
    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --mt_speech_fold_length # fold_length for speech data during mt training (default="${mt_speech_fold_length}").
    --mt_text_fold_length   # fold_length for text data during mt training (default="${mt_text_fold_length}").
    --lm_fold_length         # fold_length for LM training (default="${lm_fold_length}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Check feature type
data_feats=${dumpdir}/raw

# Use the same text as mt for bpe training if not specified.
[ -z "${bpe_train_text}" ] && bpe_train_text="${data_feats}/${train_set}/text"
# Use the same text as mt for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="${data_feats}/${train_set}/text"
# Use the same text as mt for lm training if not specified.
[ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
# Use the text of the 1st evaldir if lm_test is not specified
[ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"
# Use the same small text as mt for bpe training if not specified.
[ -z "${bpe_train_text_small}" ] && bpe_train_text_small="${data_feats}/${train_set}/text"

# Check tokenization type
token_listdir=data/token_list
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt

# Check tokenization type for small distribution
bpedir_small="${token_listdir}/bpe_${bpemode}${nbpe}_small"
bpeprefix_small="${bpedir_small}"/bpe
bpemodel_small="${bpeprefix_small}".model
token_list_small="${bpedir_small}"/tokens.txt

# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
if [ -z "${mt_tag}" ]; then
    if [ -n "${mt_config}" ]; then
        mt_tag="$(basename "${mt_config}" .yaml)_${token_type}"
    else
        mt_tag="train_${token_type}"
    fi
    if [ "${token_type}" = bpe ]; then
        mt_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${mt_args}" ]; then
        mt_tag+="$(echo "${mt_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)_${lm_token_type}"
    else
        lm_tag="train_${lm_token_type}"
    fi
    if [ "${lm_token_type}" = bpe ]; then
        lm_tag+="${nbpe}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi

# The directory used for collect-stats mode
if [ -z "${mt_stats_dir}" ]; then
    mt_stats_dir="${expdir}/mt_stats_${token_type}"
    if [ "${token_type}" = bpe ]; then
        mt_stats_dir+="${nbpe}"
    fi
fi
if [ -z "${lm_stats_dir}" ]; then
    lm_stats_dir="${expdir}/lm_stats_${lm_token_type}"
    if [ "${lm_token_type}" = bpe ]; then
        lm_stats_dir+="${nbpe}"
    fi
fi
# The directory used for training commands
if [ -z "${mt_exp}" ]; then
    mt_exp="${expdir}/mt_${mt_tag}"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi


if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_args}" ]; then
        inference_tag+="$(echo "${inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if "${use_lm}"; then
        inference_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_tag+="_mt_model_$(echo "${inference_mt_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi

# ========================== Main stages start from here. ==========================
if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # Data preparation not supported yet

        mkdir -p ${data_feats}
        for dset in "${train_set}" "${valid_set}"; do
            cp -r data/"${dset}"/text data/"${dset}"/text_input
            cp -r data/"${dset}"/text data/"${dset}"/text_output
            # Copy datadir
            rm -rf "${data_feats}"/org/"${dset}"
            rm -rf "${data_feats}"/"${dset}"
            cp -r data/"${dset}" "${data_feats}"/org/"${dset}"
            cp -r data/"${dset}" "${data_feats}"/"${dset}"
        done
        cp -r data/"${pseudo_set}"/text data/"${test_sets}"/text_input
        cp -r data/"${answer_set}"/text data/"${test_sets}"/text_output
        cp -r data/"${answer_set}"/text data/"${test_sets}"/text
        cp -r data/"${answer_set}"/utt2spk data/"${test_sets}"/utt2spk
        rm -rf "${data_feats}"/org/"${test_sets}"
        rm -rf "${data_feats}"/"${test_sets}"
        cp -r data/"${test_sets}" "${data_feats}"/org/"${test_sets}"
        cp -r data/"${test_sets}" "${data_feats}"/"${test_sets}"
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        data_feats=${dumpdir}/raw/org

        if [ "${token_type}" = bpe ]; then
            log "Stage 2: Generate token_list from ${train_set} and ${bpe_train_text} using BPE"
            # Task Independent: small distribution -> large distribution for here
            mkdir -p "${bpedir}"
            mkdir -p "${bpedir_small}"
            # shellcheck disable=SC2002
            cat ${bpe_train_text} | cut -f 2- -d" "  > "${bpedir}"/train.txt
            cat ${bpe_train_text_small} | cut -f 2- -d" "  > "${bpedir_small}"/train.txt

            if [ -n "${bpe_nlsyms}" ]; then
                _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
            else
                _opts_spm=""
            fi

            spm_train \
                --input="${bpedir}"/train.txt \
                --vocab_size="${nbpe}" \
                --model_type="${bpemode}" \
                --model_prefix="${bpeprefix}" \
                --character_coverage=${bpe_char_cover} \
                --input_sentence_size="${bpe_input_sentence_size}" \
                ${_opts_spm}

            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${token_list}"

            spm_train \
                --input="${bpedir_small}"/train.txt \
                --vocab_size="${nbpe}" \
                --model_type="${bpemode}" \
                --model_prefix="${bpeprefix_small}" \
                --character_coverage=${bpe_char_cover} \
                --input_sentence_size="${bpe_input_sentence_size}" \
                ${_opts_spm}

            {
            echo "${blank}"
            echo "${oov}"
            # Remove <unk>, <s>, </s> from the vocabulary
            <"${bpeprefix_small}".vocab awk '{ if( NR != 1 && NR != 2 && NR != 3 ){ print $1; } }'
            echo "${sos_eos}"
            } > "${token_list_small}"
        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi
    fi
    data_feats=${dumpdir}/raw

    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        log "Stage 3: Remove long/short data: ${data_feats}/org -> ${data_feats}"
        # Not Implemented yet

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do
            # Copy data dir
            rm -rf "${data_feats}/${dset}"
            cp -r "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        done

    fi
else
    log "Skip the stages for data preparation"
fi

# ========================== Data preparation is done here. ==========================


if ! "${skip_train}"; then
    if "${use_lm}"; then
        if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
            log "Stage 4: LM collect stats: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            # 1. Split the key file
            _logdir="${lm_stats_dir}/logdir"
            mkdir -p "${_logdir}"
            # Get the minimum number among ${nj} and the number lines of input files
            _nj=$(min "${nj}" "$(<${data_feats}/lm_train.txt wc -l)" "$(<${lm_dev_text} wc -l)")

            key_file="${data_feats}/lm_train.txt"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/train.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            key_file="${lm_dev_text}"
            split_scps=""
            for n in $(seq ${_nj}); do
                split_scps+=" ${_logdir}/dev.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Generate run.sh
            log "Generate '${lm_stats_dir}/run.sh'. You can resume the process from stage 6 using this script"
            mkdir -p "${lm_stats_dir}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${lm_stats_dir}/run.sh"; chmod +x "${lm_stats_dir}/run.sh"

            # 3. Submit jobs
            log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
            # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
            #       but it's used only for deciding the sample ids.
            # shellcheck disable=SC2086
            ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
                ${python} -m espnet2.bin.lm_train \
                    --collect_stats true \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --train_data_path_and_name_and_type "${data_feats}/lm_train.txt,text,text" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --train_shape_file "${_logdir}/train.JOB.scp" \
                    --valid_shape_file "${_logdir}/dev.JOB.scp" \
                    --output_dir "${_logdir}/stats.JOB" \
                    ${_opts} ${lm_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

            # 4. Aggregate shape files
            _opts=
            for i in $(seq "${_nj}"); do
                _opts+="--input_dir ${_logdir}/stats.${i} "
            done
            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

            # Append the num-tokens at the last dimensions. This is used for batch-bins count
            <"${lm_stats_dir}/train/text_output_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/train/text_output_shape.${lm_token_type}"

            <"${lm_stats_dir}/valid/text_output_shape" \
                awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
                >"${lm_stats_dir}/valid/text_output_shape.${lm_token_type}"
        fi


        if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
            log "Stage 5: LM Training: train_set=${data_feats}/lm_train.txt, dev_set=${lm_dev_text}"

            _opts=
            if [ -n "${lm_config}" ]; then
                # To generate the config file: e.g.
                #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
                _opts+="--config ${lm_config} "
            fi

            if [ "${num_splits_lm}" -gt 1 ]; then
                # If you met a memory error when parsing text files, this option may help you.
                # The corpus is split into subsets and each subset is used for training one by one in order,
                # so the memory footprint can be limited to the memory required for each dataset.

                _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
                if [ ! -f "${_split_dir}/.done" ]; then
                    rm -f "${_split_dir}/.done"
                    ${python} -m espnet2.bin.split_scps \
                      --scps "${data_feats}/lm_train.txt" "${lm_stats_dir}/train/text_output_shape.${lm_token_type}" \
                      --num_splits "${num_splits_lm}" \
                      --output_dir "${_split_dir}"
                    touch "${_split_dir}/.done"
                else
                    log "${_split_dir}/.done exists. Spliting is skipped"
                fi

                _opts+="--train_data_path_and_name_and_type ${_split_dir}/lm_train.txt,text,text "
                _opts+="--train_shape_file ${_split_dir}/text_output_shape.${lm_token_type} "
                _opts+="--multiple_iterator true "

            else
                _opts+="--train_data_path_and_name_and_type ${data_feats}/lm_train.txt,text,text "
                _opts+="--train_shape_file ${lm_stats_dir}/train/text_output_shape.${lm_token_type} "
            fi

            # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

            log "Generate '${lm_exp}/run.sh'. You can resume the process from stage 4 using this script"
            mkdir -p "${lm_exp}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${lm_exp}/run.sh"; chmod +x "${lm_exp}/run.sh"

            log "LM training started... log: '${lm_exp}/train.log'"
            if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
                # SGE can't include "/" in a job name
                jobname="$(basename ${lm_exp})"
            else
                jobname="${lm_exp}/train.log"
            fi

            # shellcheck disable=SC2086
            ${python} -m espnet2.bin.launch \
                --cmd "${cuda_cmd} --name ${jobname}" \
                --log "${lm_exp}"/train.log \
                --ngpu "${ngpu}" \
                --num_nodes "${num_nodes}" \
                --init_file_prefix "${lm_exp}"/.dist_init_ \
                --multiprocessing_distributed true -- \
                ${python} -m espnet2.bin.lm_train \
                    --ngpu "${ngpu}" \
                    --use_preprocessor true \
                    --bpemodel "${bpemodel}" \
                    --token_type "${lm_token_type}"\
                    --token_list "${lm_token_list}" \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --cleaner "${cleaner}" \
                    --g2p "${g2p}" \
                    --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                    --valid_shape_file "${lm_stats_dir}/valid/text_output_shape.${lm_token_type}" \
                    --fold_length "${lm_fold_length}" \
                    --resume true \
                    --output_dir "${lm_exp}" \
                    ${_opts} ${lm_args}

        fi


        if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            log "Stage 6: Calc perplexity: ${lm_test_text}"
            _opts=
            # TODO(kamo): Parallelize?
            log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
            # shellcheck disable=SC2086
            ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
                ${python} -m espnet2.bin.lm_calc_perplexity \
                    --ngpu "${ngpu}" \
                    --data_path_and_name_and_type "${lm_test_text},text,text" \
                    --train_config "${lm_exp}"/config.yaml \
                    --model_file "${lm_exp}/${inference_lm}" \
                    --output_dir "${lm_exp}/perplexity_test" \
                    ${_opts}
            log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

        fi

    else
        log "Stage 4-6: Skip lm-related stages: use_lm=${use_lm}"
    fi


    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        _mt_train_dir="${data_feats}/${train_set}"
        _mt_valid_dir="${data_feats}/${valid_set}"

        log "Stage 7: mt collect stats: train_set=${_mt_train_dir}, valid_set=${_mt_valid_dir}"

        _opts=
        if [ -n "${mt_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.mt_train --print_config --optim adam
            _opts+="--config ${mt_config} "
        fi

        _scp=text

        # 1. Split the key file
        _logdir="${mt_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_mt_train_dir}/${_scp} wc -l)" "$(<${_mt_valid_dir}/${_scp} wc -l)")

        key_file="${_mt_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_mt_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${mt_stats_dir}/run.sh'. You can resume the process from stage 7 using this script"
        mkdir -p "${mt_stats_dir}"; echo "${run_args} --stage 7 \"\$@\"; exit \$?" > "${mt_stats_dir}/run.sh"; chmod +x "${mt_stats_dir}/run.sh"

        # 3. Submit jobs
        log "mt collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m hynet.bin.mt_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --bpemodel_small "${bpemodel_small}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --token_list_small "${token_list_small}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --train_data_path_and_name_and_type "${_mt_train_dir}/text_input,text_input,text" \
                --train_data_path_and_name_and_type "${_mt_train_dir}/text_output,text_output,text" \
                --valid_data_path_and_name_and_type "${_mt_valid_dir}/text_input,text_input,text" \
                --valid_data_path_and_name_and_type "${_mt_valid_dir}/text_output,text_output,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${mt_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${mt_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${mt_stats_dir}/train/text_output_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${mt_stats_dir}/train/text_output_shape.${token_type}"

        <"${mt_stats_dir}/valid/text_output_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${mt_stats_dir}/valid/text_output_shape.${token_type}"
    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        _mt_train_dir="${data_feats}/${train_set}"
        _mt_valid_dir="${data_feats}/${valid_set}"
        log "Stage 8: mt Training: train_set=${_mt_train_dir}, valid_set=${_mt_valid_dir}"

        _opts=
        if [ -n "${mt_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.mt_train --print_config --optim adam
            _opts+="--config ${mt_config} "
        fi

        if [ "${num_splits_mt}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${mt_stats_dir}/splits${num_splits_mt}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_mt_train_dir}/text_input" \
                      "${_mt_train_dir}/text_output" \
                      "${mt_stats_dir}/train/text_input_shape" \
                      "${mt_stats_dir}/train/text_output_shape.${token_type}" \
                  --num_splits "${num_splits_mt}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text_input,text_input,text "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text_output,text_output,text "
            _opts+="--train_shape_file ${_split_dir}/text_input_shape "
            _opts+="--train_shape_file ${_split_dir}/text_output_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_mt_train_dir}/text_input,text_input,text "
            _opts+="--train_data_path_and_name_and_type ${_mt_train_dir}/text_output,text_output,text "
            _opts+="--train_shape_file ${mt_stats_dir}/train/text_input_shape "
            _opts+="--train_shape_file ${mt_stats_dir}/train/text_output_shape.${token_type} "
        fi

        log "Generate '${mt_exp}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${mt_exp}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${mt_exp}/run.sh"; chmod +x "${mt_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "mt training started... log: '${mt_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${mt_exp})"
        else
            jobname="${mt_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${mt_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${mt_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m hynet.bin.mt_train \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --bpemodel_small "${bpemodel_small}" \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --token_list_small "${token_list_small}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_mt_valid_dir}/text_input,text_input,text" \
                --valid_data_path_and_name_and_type "${_mt_valid_dir}/text_output,text_output,text" \
                --valid_shape_file "${mt_stats_dir}/valid/text_input_shape" \
                --valid_shape_file "${mt_stats_dir}/valid/text_output_shape.${token_type}" \
                --resume true \
                --output_dir "${mt_exp}" \
                ${_opts} ${mt_args}

    fi
else
    log "Skip the training stages"
fi

if ! "${skip_eval}"; then
    if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
        log "Stage 9: Decoding: training_dir=${mt_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi

        # 2. Generate run.sh
        log "Generate '${mt_exp}/${inference_tag}/run.sh'. You can resume the process from stage 9 using this script"
        mkdir -p "${mt_exp}/${inference_tag}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${mt_exp}/${inference_tag}/run.sh"; chmod +x "${mt_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${mt_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=text

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/mt_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/mt_inference.JOB.log \
                ${python} -m hynet.bin.mt_inference \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/text_input,text_input,text" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --mt_train_config "${mt_exp}"/config.yaml \
                    --mt_model_file "${mt_exp}"/"${inference_mt_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${inference_args}

            # 3. Concatenates the output files from each jobs
            for f in token token_int score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done
        done
    fi


    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        log "Stage 10: Scoring"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${mt_exp}/${inference_tag}/${dset}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"


                elif [ "${_type}" = cer ]; then
                    # Tokenize text to char level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                elif [ "${_type}" = ter ]; then
                    # Tokenize text using BPE
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                fi

                sclite \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done
        done

        [ -f local/score.sh ] && local/score.sh "${mt_exp}"

        # Show results in Markdown syntax
        scripts/utils/show_mt_result.sh "${mt_exp}" > "${mt_exp}"/RESULTS.md
        cat "${mt_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi


packed_model="${mt_exp}/${mt_exp##*/}_${inference_mt_model%.*}.zip"
if ! "${skip_upload}"; then
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "Stage 13: Pack model: ${packed_model}"

        _opts=
        if "${use_lm}"; then
            _opts+="--lm_train_config ${lm_exp}/config.yaml "
            _opts+="--lm_file ${lm_exp}/${inference_lm} "
            _opts+="--option ${lm_exp}/perplexity_test/ppl "
            _opts+="--option ${lm_exp}/images "
        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            _opts+="--option ${mt_stats_dir}/train/feats_stats.npz "
        fi
        if [ "${token_type}" = bpe ]; then
            _opts+="--option ${bpemodel} "
        fi
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.pack mt \
            --mt_train_config "${mt_exp}"/config.yaml \
            --mt_model_file "${mt_exp}"/"${inference_mt_model}" \
            ${_opts} \
            --option "${mt_exp}"/RESULTS.md \
            --option "${mt_exp}"/images \
            --outpath "${packed_model}"
    fi


    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 14: Upload model to Zenodo: ${packed_model}"

        # To upload your model, you need to do:
        #   1. Sign up to Zenodo: https://zenodo.org/
        #   2. Create access token: https://zenodo.org/account/settings/applications/tokens/new/
        #   3. Set your environment: % export ACCESS_TOKEN="<your token>"

        if command -v git &> /dev/null; then
            _creator_name="$(git config user.name)"
            _checkout="
git checkout $(git show -s --format=%H)"

        else
            _creator_name="$(whoami)"
            _checkout=""
        fi
        # /some/where/espnet/egs2/foo/mt1/ -> foo/mt1
        _task="$(pwd | rev | cut -d/ -f2 | rev)"
        # foo/mt1 -> foo
        _corpus="${_task%/*}"
        _model_name="${_creator_name}/${_corpus}_$(basename ${packed_model} .zip)"

        # Generate description file
        cat << EOF > "${mt_exp}"/description
This model was trained by ${_creator_name} using ${_task} recipe in <a href="https://github.com/espnet/espnet/">espnet</a>.
<p>&nbsp;</p>
<ul>
<li><strong>Python API</strong><pre><code class="language-python">See https://github.com/espnet/espnet_model_zoo</code></pre></li>
<li><strong>Evaluate in the recipe</strong><pre>
<code class="language-bash">git clone https://github.com/espnet/espnet
cd espnet${_checkout}
pip install -e .
cd $(pwd | rev | cut -d/ -f1-3 | rev)
./run.sh --skip_data_prep false --skip_train true --download_model ${_model_name}</code>
</pre></li>
<li><strong>Results</strong><pre><code>$(cat "${mt_exp}"/RESULTS.md)</code></pre></li>
<li><strong>mt config</strong><pre><code>$(cat "${mt_exp}"/config.yaml)</code></pre></li>
<li><strong>LM config</strong><pre><code>$(if ${use_lm}; then cat "${lm_exp}"/config.yaml; else echo NONE; fi)</code></pre></li>
</ul>
EOF

        # NOTE(kamo): The model file is uploaded here, but not published yet.
        #   Please confirm your record at Zenodo and publish it by yourself.

        # shellcheck disable=SC2086
        espnet_model_zoo_upload \
            --file "${packed_model}" \
            --title "ESPnet2 pretrained model, ${_model_name}, fs=${fs}, lang=${lang}" \
            --description_file "${mt_exp}"/description \
            --creator_name "${_creator_name}" \
            --license "CC-BY-4.0" \
            --use_sandbox false \
            --publish false
    fi
else
    log "Skip the uploading stages"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
