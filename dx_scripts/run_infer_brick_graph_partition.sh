#!/bin/bash
linarg_identifier=$1
load_dir=$2
partition_identifier=$3

linarg_dir="linear_arg_results/${linarg_identifier}"

set -euo pipefail

# download python 3.9 and cython complier dependencies
sudo apt-get update
sudo apt -y install python3.9
sudo apt -y install python3.9-dev
python3.9 -m pip install --upgrade pip

python3.9 -m pip install --upgrade scipy
python3.9 -m pip install cyvcf2
python3.9 -m pip install dxpy # for dna_nexus.py
python3.9 -m pip install pyspark # for dna_nexus.py
python3.9 -m pip install git+https://github.com/quattro/linear-dag.git@partition_and_merge

export PATH=$PATH:/home/dnanexus/.local/bin/
kodama infer-brick-graph --linarg_dir $linarg_dir --load_dir $load_dir --partition_identifier $partition_identifier