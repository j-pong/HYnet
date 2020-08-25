# PyTorch version: 0.4.1, 1.0.0, 1.0.1, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0
TH_VERSION := 1.6.0
WGET := wget --tries=3

# Use pip for pytorch installation even if you have anaconda
ifneq ($(shell test -f ./activate_python.sh && grep 'conda activate' ./activate_python.sh),)
IS_CONDA := 0
USE_PIP :=
else
IS_CONDA :=
USE_PIP := 0
endif


# Set if install binaries on CPU mode e.g. make CPU_ONLY=1
# If you don't have any GPUs, this value will be set automatically
ifeq ($(shell which nvcc),) # 'nvcc' not found
CPU_ONLY := 0
else
CPU_ONLY :=
endif

ifeq ($(strip $(CPU_ONLY)),)
# Derive CUDA version from nvcc
CUDA_VERSION = $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(CUDA_VERSION)))
CONDA_PYTORCH := pytorch=$(TH_VERSION) torchvision cudatoolkit=$(CUDA_VERSION)
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch_stable.html

else
CONDA_PYTORCH := pytorch=$(TH_VERSION) torchvision cpuonly
PIP_PYTORCH := torch==$(TH_VERSION) -f https://download.pytorch.org/whl/cpu/torch_stable.html
NO_CUPY := 0
endif

.PHONY: all

all: kaldi showenv sph2pipe.done python

python: activate_python.sh pytorch.done fairseq.done espnet.done hynet.done

kaldi:
	test -f kaldi/egs/wsj/s5/utils/parse_options.sh || { echo -e "Error: Put Kaldi here!\n    $$ ln -s <kaldi-root> kaldi"; exit 1; }

activate_python.sh:
	test -f activate_python.sh || { echo "Error: Run ./setup_python.sh or ./setup_anaconda.sh"; exit 1; }

################ Logging ################
showenv: activate_python.sh
ifeq ($(strip $(CPU_ONLY)),)
	[ -n "${CUDA_HOME}" ] || { echo -e "Error: CUDA_HOME is not set.\n    $$ . ./setup_cuda_env.sh <cuda-root>"; exit 1; }
	@echo CUDA_VERSION=$(CUDA_VERSION)
	. ./activate_python.sh && python3 check_pytorch_cuda_compatibility.py $(TH_VERSION) ${CUDA_VERSION} || { echo "Change TH_VERSION or CUDA_VERSION"; exit 1; }
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . ./activate_python.sh && command -v python3)
	@echo PYTHON_VERSION=$(shell . ./activate_python.sh && python3 --version)
	@echo IS_CONDA=$(IS_CONDA)
	@echo USE_PIP=$(USE_PIP)
	@echo TH_VERSION=$(TH_VERSION)
	@echo CONDA_PYTORCH=$(CONDA_PYTORCH)
	@echo PIP_PYTORCH=$(PIP_PYTORCH)
	@echo CHAINER_VERSION=$(CHAINER_VERSION)
	@echo PIP_CHAINER=$(PIP_CHAINER)
	@echo NO_CUPY=$(NO_CUPY)

#########################################

sph2pipe.done:
	./installer_espnet/install_sph2pipe.sh
	touch sph2pipe.done

pytorch.done: activate_python.sh
	# . ./activate_python.sh && conda install -y $(CONDA_PYTORCH) -c pytorch
	. ./activate_python.sh && pip install torch torchvision
	touch pytorch.done

fairseq.done: activate_python.sh
	rm -rf fairseq
	git clone https://github.com/pytorch/fairseq
	. ./activate_python.sh && cd fairseq && pip install -e .
	touch fairseq.done

espnet.done: pytorch.done
	rm -rf espnet
	git clone https://github.com/espnet/espnet

	rm -rf installer_espnet
	mkdir installer_espnet
	cp espnet/tools/installers/* installer_espnet

	. ./activate_python.sh && ./installer_espnet/install_warp-ctc.sh
	. ./activate_python.sh && cd espnet && python3 -m pip install -e .
	touch espnet.done

hynet.done: espnet.done
	. ./activate_python.sh && python3 -m pip install -e .. 
	touch hynet.done
