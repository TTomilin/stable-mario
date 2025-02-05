#!/bin/bash

echo "Installing required packages..."

. /home/$SUDO_USER/miniconda3/etc/profile.d/conda.sh

sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install zlib1g-dev
sudo apt-get install -y python3-opengl
conda install conda-forge::libcxx
conda install -c conda-forge libstdcxx-ng
sudo apt install libbz2-dev