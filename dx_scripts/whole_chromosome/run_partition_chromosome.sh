#!/bin/bash

set -euo pipefail

# download python 3.9 and cython complier dependencies
sudo apt-get update
sudo apt -y install python3.9
sudo apt -y install python3.9-dev
python3.9 -m pip install --upgrade pip
python3.9 -m pip install numpy

python3.9 /mnt/project/amber/scripts/partition_chromosome.py