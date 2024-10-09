#!/bin/bash
linarg_identifier=$1
load_dir=$2
instance_type=$3

mtx_list=($(dx ls "/linear_arg_results/${linarg_identifier}/genotype_matrices/"))
brick_graph_list=($(dx ls "/linear_arg_results/${linarg_identifier}/brick_graph_partitions/"))

for f in "${mtx_list[@]}"
do
    partition_identifier=$(echo "$f" | awk -F. '{print $1}')
    if [[ " ${brick_graph_list[@]} " =~ " ${partition_identifier}.npz " ]]; then # skip partitions that have already been inferred
        continue
    fi

    echo $f
    dx run app-swiss-army-knife \
        -iin="/amber/scripts/run_infer_brick_graph_partition.sh" \
        -icmd="bash run_infer_brick_graph_partition.sh $linarg_identifier $load_dir $partition_identifier" \
        --destination "/" \
        --instance-type $instance_type \
        --priority low \
        --name "brick_graph_${linarg_identifier}_${partition_identifier}" \
        --brief \
        -y
done