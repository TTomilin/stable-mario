#!/bin/bash

echo "Installing required packages..."

. /home/$SUDO_USER/miniconda3/etc/profile.d/conda.sh

sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install zlib1g-dev
conda install conda-forge::libcxx
conda install -c conda-forge libstdcxx-ng