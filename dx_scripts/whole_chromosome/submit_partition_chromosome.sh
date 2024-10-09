#!/bin/bash

dx run app-swiss-army-knife \
    -iin="/amber/scripts/run_partition_chromosome.sh" \
    -icmd="bash run_partition_chromosome.sh" \
    --destination "/" \
    --instance-type "mem1_ssd1_v2_x2" \
    --priority low \
    --name "partition_chromosome" \
    --brief \
    -y