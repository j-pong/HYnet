docker run -v /path/to/Workspace:/home/Workspace -it -d --name KT-docker <image_id>
docker start KT-docker; docker attach KT-docker
apt-get upgrade; apt-get update
apt-get install -y git
apt-get install -y zlib1g-dev automake autoconf unzip sox gfortran libtool subversion python2.7 python3
make
