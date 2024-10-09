#!/bin/bash
instance_type=$1

load_dir='/mnt/project/'
out="linear_args"
data_identifier="ukb20279"
chrom=1
chrom_dir="${out}/${data_identifier}/chr${chrom}"

linarg_dir_list=($(dx ls $chrom_dir))

for dir in ${linarg_dir_list[@]}
do
    linarg_dir=${chrom_dir}/${dir}

    echo "${dir}, ${linarg_dir}, ${load_dir}"

    dx run app-swiss-army-knife \
        -iin="/amber/scripts/run_merge_brick_graphs.sh" \
        -icmd="bash run_merge_brick_graphs.sh $linarg_dir $load_dir" \
        --destination "/" \
        --instance-type $instance_type \
        --priority low \
        --name "linarg_merge_${dir}" \
        --brief \
        -y
done