MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi
ESPNET_ROOT=$MAIN_ROOT/tools/espnet

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$ESPNET_ROOT/tools/chainer_ctc/ext/warp-ctc/build
if [ -e $ESPNET_ROOT/tools/venv/etc/profile.d/conda.sh ]; then
    source $ESPNET_ROOT/tools/venv/etc/profile.d/conda.sh && conda deactivate && conda activate
else
    source $ESPNET_ROOT/tools/venv/bin/activate
fi
export PATH=$ESPNET_ROOT/utils:$ESPNET_ROOT/espnet/bin:$MAIN_ROOT/moneynet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
