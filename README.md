# HYnet
Fully Unsupervised Learning for Continual Sequence when High Local Correlation

## Installation
```
cd tools

source ./setup_cuda_env.sh /usr/local/cuda/

./installer/install_kaldi.sh
./setup_anaconda.sh venv base 3.7.3
make
```

## Bugfix
### ctc_segmentation/ctc_segmentation_dyn.pyx error
Remove ctc_segmentation at tools/espnet/setup.py that is included in requirements
### matplotlib version error
Remove matplotlib at tools/espnet/setup.py that is included in requirements