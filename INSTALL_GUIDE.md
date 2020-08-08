# HYnet (Hanyang Univ ASML 2020)
This reco consists of various ASR toolkits (kaldi, ESPnet, FairSeq). The goal of our code is to construct integrated speech recognition framework for E2E & Hybrid ASR systems.

### OS
Ubuntu 18.04

# installation guide #
### Requirements
cuda10.2, cudnn, apt-transport-https, ca-certificates, curl, gnupg-agent, software-properties-common, git, subversion, automake, autoconf, build-essential, zlib1g-dev, libtool, libatlas-base-dev, sox, flac, gfortran, python2.7

### INSTALL DOCKER ENGINE - COMMUNITY
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo apt-key fingerprint 0EBFCD88
  sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io

### INSTALL NVIDIA-DOCKER
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update
  sudo apt-get install nvidia-docker2
  sudo pkill -SIGHUP dockerd

### Build docker
  check installation file (moneynet/docker/Dockerfile : you can change your docker password & user name here)
  cd moneynet/docker
  sudo usermod -aG docker $USER
  make _build
  make run
  sudo reboot
  make info

### INSTALL HYnet
  ssh jpong@192.168.0.10 -p 32769 
  cd /home/Workspace/moneynet/tools
  edit Makefile (If kaldi is not installed : edit line 8, KALDI :=)
  sudo make 

### INSTALL srilm
  cd /home/Workspace/moneynet/tools/kaldi/tools/install_srilm.sh
