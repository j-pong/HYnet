#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
img_ver=$1
echo "install cuda version with ${img_ver}"

if [[ $img_ver == 10 ]]; then 
	cuda_img="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04"
elif [[ $img_ver == 11 ]]; then
	cuda_img="nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04"
fi

echo "FROM ${cuda_img}" > Dockerfile
cat Dockerfile_meta >> Dockerfile

