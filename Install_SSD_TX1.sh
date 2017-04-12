#!/bin/sh
# Script for installing Caffe SSD with cuDNN support on Jetson TX1 Development Kits
# Modified from JetsonHacks file and Dockerfiles:
# https://github.com/jetsonhacks/installCaffeJTX1/blob/master/installCaffeCuDNN.sh
# https://github.com/pool1892/docker/blob/master/caffe_pre/Dockerfile
# https://github.com/pool1892/docker/blob/master/ssd/Dockerfile
# Install and compile Caffe on NVIDIA Jetson TX1 Development Kit
# Prerequisites (which can be installed with JetPack 2):
# L4T 24.2 (Ubuntu 16.04)
# OpenCV4Tegra
# CUDA 8.0
# cuDNN v5.1

sudo add-apt-repository universe
sudo apt-get update -y
/bin/echo -e "\e[1;32mLoading Caffe Dependencies.\e[0m"
sudo apt-get install --no-install-recommends build-essential cmake git unzip wget -y
# General Dependencies
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev \
libhdf5-serial-dev protobuf-compiler -y
sudo apt-get install --no-install-recommends libboost-all-dev -y
# BLAS
# To Do: Switch to OPENBLAS
sudo apt-get install libatlas-base-dev -y
# Remaining Dependencies
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev -y
sudo apt-get install python-dev python-numpy python-pip python-setuptools \
python-scipy python-nose python-h5py python-skimage python-matplotlib \
python-pandas python-sklearn python-sympy python-scipy -y

sudo usermod -a -G video $USER
/bin/echo -e "\e[1;32mCloning Caffe-SSD into $HOME/git directory.\e[0m"
cd $HOME
# Git clone Caffe SSD
git clone https://github.com/weiliu89/caffe.git caffe-ssd
cd caffe-ssd
# Switch to SSD branch
git checkout ssd

/bin/echo -e "\e[1;32mInstalling OpenCV Libraries.\e[0m"
# Install OpenCV Libraries
cd $HOME
sudo apt-get install libopencv-dev
sudo ./OpenCV4Tegra/ocv.sh

/bin/echo -e "\e[1;32mOverlocking Jetson.\e[0m"
# save current settings
#sudo ./jetson_clocks.sh --store default-clocks
# load performance-optimized profile
#sudo ./jetson_clocks.sh

/bin/echo -e "\e[1;32mLoading Caffe pip Dependencies.\e[0m"
pip install --upgrade pip && \
pip --no-cache-dir install ipykernel jupyter sklearn && \
python -m ipykernel.kernelspec
cd caffe-ssd/python && for req in $(cat requirements.txt) pydot; do pip install $req; done && cd ..

/bin/echo -e "\e[1;32mPerforming CMake.\e[0m"
mkdir build && cd build
cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..

/bin/echo -e "\e[1;32mCompiling Caffe.\e[0m"
make -j"$(nproc)" all
# make install ???
# make symlink ???

#if [[ -z $(cat ~/.bashrc | grep "$HOME/caffe-ssd") ]] ; then
#    echo -e "\n# Adds Caffe to the PATH variable" >> ~/.bashrc
#    echo "export CAFFE_ROOT=$HOME/caffe-ssd" >> ~/.bashrc
#    echo "export PYCAFFE_ROOT=$CAFFE_ROOT/python" >> ~/.bashrc
#    echo "export PYTHONPATH=$PYCAFFE_ROOT:$PYTHONPATH" >> ~/.bashrc
#    echo "export PATH=$CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH" >> ~/.bashrc
#    source ~/.bashrc
#fi

# Run the tests to make sure everything works
#/bin/echo -e "\e[1;32mRunning Caffe Tests.\e[0m"
#make -j4 runtest
# The following is a quick timing test ...
# tools/caffe time --model=models/bvlc_alexnet/deploy.prototxt --gpu=0
