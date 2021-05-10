# HYnet
Fully Unsupervised Learning for Continual Sequence when High Local Correlation

## Installation Guide for SL

### 1. Installaion of docker for ssh env
- Basic of docker installation

- Docker for HYnet
```bash
cd docker

CUDA_VERSION=10 # if cuda version is 10.x
CUDA_VERSION=11 # if cuda version is 11.x

. cuda_img_name.sh $CUDA_VERSION

make _build

make run
```

### 2. Installaion of kaldi for speech
```bash
ssh jpong@192.168.0.104 -p 32770

cd /path/to/HYnet/tools
git clone https://github.com/kaldi-asr/kaldi kaldi

cd kaldi/tools
extras/install_mkl.sh -s
extras/check_dependencies.sh
make -j 28
extras/install_irstlm.sh

cd ../src/
./configure
make depend -j 28
make -j 28
```

### 3. Installaion of espnet for input pipelines
```bash
cd ../../tools

./meta_installers/install_espnet.sh
```

### 4. Installaion of hynet for customizing egs
```bash
cd tools

./meta_installers/install_hynet.sh
```

## Make custom example with task
```bash
cp -r tools/espnet/egs2/TEMPLATE/asr1/* egs/TEMPLATE/asr1/
```

## Bugfix

ctc_segmentation/ctc_segmentation_dyn.pyx error
- Remove ctc_segmentation at tools/espnet/setup.py that is included in requirements

ctc install error with pip version
```bash
pip install pip==19; pip install warpctc-pytorch==0.2.1+torch16.cuda102
```

matplotlib version error
- Remove matplotlib at tools/espnet/setup.py that is included in requirements
