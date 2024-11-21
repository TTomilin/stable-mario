# Stable mario installation, step-by-step (20-11-2024):
1. Download and install miniconda3 by pasting the following commands into your terminal:
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm ~/miniconda3/miniconda.sh
2. Create a new virtual environment called ‘stable-mario’, Python version 3.9 by running:
        conda create -n stable-mario python==3.9
3. Clone the codebase:
        git clone https://github.com/TTomilin/stable-mario.git
4. Move into the codebase:
        cd stable-mario
5. Make the linux installation script executable:
        chmod +x ubuntu_install.sh
6. Run:
        sudo ./ubuntu_install.sh
7. Run:
        pip install -e .
8. Installation should now be complete.
