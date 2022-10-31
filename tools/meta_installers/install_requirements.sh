#!/bin/bash
. ./activate_python.sh

# install sentencepiece
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v

# install warp-rnnt
cd ../../
cd espnet/tools
. ./setup_cuda_env.sh /usr/local/cuda
./installers/install_warp-transducer.sh
