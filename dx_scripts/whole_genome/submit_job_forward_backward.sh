#!/bin/bash
instance_type=$1

load_dir='/mnt/project/'
out="linear_args"
data_identifier="ukb20279"
chroms=({1..22})

for chrom in "${chroms[@]}"; do

    # test run
    if [[ $chrom != 15 ]]; then
        continue 
    fi

    chrom_dir="${out}/${data_identifier}/chr${chrom}"
    linarg_dir_list=($(dx ls $chrom_dir))

    for dir in ${linarg_dir_list[@]}; do
    
        linarg_dir=${chrom_dir}/${dir}
        mtx_list=($(dx ls "${linarg_dir}/genotype_matrices/"))
        forward_backward_list=($(dx ls "${linarg_dir}/forward_backward_graphs/"))

        for f in "${mtx_list[@]}"; do
            partition_identifier=$(echo "$f" | awk -F. '{print $1}')
            if [[ " ${forward_backward_list[@]} " =~ " ${partition_identifier}_forward_graph.h5 " ]]; then # skip partitions that have already been inferred
                echo "$${partition_identifier}_forward_graph.h5 already exists."
                continue
            fi

            echo "${linarg_dir} ${load_dir} ${partition_identifier}"

            echo $f
            dx run app-swiss-army-knife \
                -iin="/amber/scripts/run_forward_backward.sh" \
                -icmd="bash run_forward_backward.sh $linarg_dir $load_dir $partition_identifier" \
                --destination "/" \
                --instance-type $instance_type \
                --priority low \
                --name "forward_backward_${partition_identifier}" \
                --brief \
                -y
        done
    done
done