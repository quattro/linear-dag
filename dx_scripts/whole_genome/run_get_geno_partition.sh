#!/bin/bash
vcf_path=$1
linarg_dir=$2
region=$3
partition_number=$4
whitelist_path=$5

set -euo pipefail

# download python 3.10
sudo apt update
sudo apt install libffi-dev
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.10.13
pyenv global 3.10.13

pip install git+https://github.com/quattro/linear-dag.git@update_partition_merge
kodama make-geno --vcf_path "$vcf_path" \
                 --linarg_dir $linarg_dir \
                 --region $region \
                 --partition_number $partition_number \
                 --phased \
                 --flip_minor_alleles \
                 --whitelist_path $whitelist_path