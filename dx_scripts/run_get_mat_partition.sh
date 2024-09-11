#!/bin/bash
vcf_path=$1
linarg_dir=$2
region=$3
partition_number=$4
phased=$5
flip_minor_alleles=$6

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
python3.9 -m pip install git+https://github.com/quattro/linear-dag.git

export PATH=$PATH:/home/dnanexus/.local/bin/
kodama make-geno --vcf_path "$vcf_path" \
                 --linarg_dir $linarg_dir \
                 --region $region \
                 --partition_number $partition_number \
                 --phased $phased \
                 --flip_minor_alleles $flip_minor_alleles 
