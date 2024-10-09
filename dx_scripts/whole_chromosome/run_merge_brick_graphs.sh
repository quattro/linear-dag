#!/bin/bash
linarg_dir=$1
load_dir=$2

# download python 3.9 and cython complier dependencies
sudo apt-get update
sudo apt -y install python3.9
sudo apt -y install python3.9-dev
python3.9 -m pip install --upgrade pip

python3.9 -m pip install --upgrade scipy
python3.9 -m pip install cyvcf2
python3.9 -m pip install git+https://github.com/quattro/linear-dag.git

export PATH=$PATH:/home/dnanexus/.local/bin/
kodama merge --linarg_dir $linarg_dir --load_dir $load_dir
