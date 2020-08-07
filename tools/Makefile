PYTHON_VERSION := 3.7.3
CUDA_VERSION := 10.2

# Pytorch version for fairseq
TH_VERSION := 1.5.0

# Use a prebuild Kaldi to omit the installation
KALDI := kaldi
WGET := wget --tries=3

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)

CONDA_PYTORCH := pytorch=$(TH_VERSION) torchvision cudatoolkit=$(CUDA_VERSION)

.PHONY: all clean

all: kaldi.done python

python: venv espnet.done fairseq.done moneynet.done

ifneq ($(strip $(KALDI)),)
kaldi.done:
	test -d $(KALDI)
	ln -s $(abspath $(KALDI)) kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi kaldi && \
    cd kaldi/tools && \
    extras/install_mkl.sh -s && \
    extras/check_dependencies.sh && \
    make -j 4 && \
    extras/install_irstlm.sh && \
    cd ../src/ && \
    ./configure && \
    make depend -j 4 && \
    make -j 4
	touch kaldi.done
endif

miniconda.sh:
	test -f miniconda.sh || $(WGET) $(CONDA_URL) -O miniconda.sh

venv: miniconda.sh
	test -d $(PWD)/venv || bash miniconda.sh -b -p $(PWD)/venv
	. venv/bin/activate && conda install -y setuptools -c anaconda
	. venv/bin/activate && conda install -y pip -c anaconda
	. venv/bin/activate && conda update -y conda
	. venv/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. venv/bin/activate && conda info -a

espnet.done: venv
	git clone https://github.com/espnet/espnet
	cd espnet/tools && ./setup_anaconda.sh venv && make
	# import the python package to current venv
	cd espnet/tools && . ../../venv/bin/activate && pip install -e ..
	# import moneynet to espnet venv
	cd espnet/tools && . venv/bin/activate && pip install -e ../../..
	touch espnet.done

fairseq.done: espnet.done
	rm -rf fairseq
	git clone https://github.com/pytorch/fairseq
	cd fairseq && . ../venv/bin/activate && pip uninstall -y typing && pip install -e .
	touch fairseq.done

moneynet.done: espnet.done
	. venv/bin/activate && pip install -e ..
	. espnet/tools/venv/bin/activate && pip install -e ..
	touch moneynet.done
