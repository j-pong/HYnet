#!/bin/bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

git clone https://github.com/kaldi-asr/kaldi kaldi

cd kaldi/tools && \
extras/install_mkl.sh -s && \
extras/check_dependencies.sh && \
make -j 4 && \
extras/install_irstlm.sh && \
cd ../src/ && \
./configure && \
make depend -j 4 && \
make -j 4