#!/bin/bash

rm -rf espnet
git clone https://github.com/espnet/espnet

# HYnet is extension of espnet
cp -r espnet/tools/* ./

# The espnet installation process https://espnet.github.io/espnet/installation.html
. ./setup_cuda_env.sh /usr/local/cuda
. ./setup_anaconda.sh venv base 3.8.5

# We needs just pyotrch if needs more dependency then install like this
make TH_VERSION=1.8.1 CHAINER_VERSION=6.0.0 CUDA_VERSION=10.2 pytorch.done

# The espent editable import to HYnet
. ./activate_python.sh && cd espnet/tools && python3 -m pip install -e "..[recipe]"

touch espnet.done

# install sentencepiece
cd ../../
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
