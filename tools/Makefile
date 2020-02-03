# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python3.7
PYTHON :=
# The python version installed in the conda setup
# NOTE(kan-bayashi): Use 3.7.3 to avoid sentencepiece installation error
PYTHON_VERSION := 3.7.3
TH_VERSION :=
WGET := wget --tries 3
OS := MaxOSX

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-$(OS)-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)


ifneq ($(shell which nvidia-smi),) # 'nvcc' found
CONDA_PYTORCH := pytorch=$(TH_VERSION) cudatoolkit=$(CUDA_VERSION)
else
CONDA_PYTORCH := pytorch-cpu=$(TH_VERSION)
endif


.PHONY: all

all: venv

miniconda.sh:
	test -f miniconda.sh || $(WGET) $(CONDA_URL) -O miniconda.sh
venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda install -y setuptools -c anaconda
	. venv/bin/activate && conda install -y pip -c anaconda
	. venv/bin/activate && conda update -y conda
	. venv/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. venv/bin/activate && conda info -a

	. venv/bin/activate && conda install pytorch torchvision -c pytorch
	. venv/bin/activate && pip install scikit-learn matplotlib seaborn librosa soundfile configargparse
	. venv/bin/activate && pip install -e ..