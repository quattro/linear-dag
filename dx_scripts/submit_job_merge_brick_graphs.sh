#!/bin/bash
linarg_identifier=$1
load_dir=$2
instance_type=$3

dx run app-swiss-army-knife \
    -iin="/amber/scripts/run_merge_brick_graphs.sh" \
    -icmd="bash run_merge_brick_graphs.sh $linarg_identifier $load_dir" \
    --destination "/" \
    --instance-type $instance_type \
    --priority low \
    --name "linarg_merge_${linarg_identifier}_${partition_identifier}" \
    --brief \
    -y