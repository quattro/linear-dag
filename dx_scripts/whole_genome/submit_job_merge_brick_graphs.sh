#!/bin/bash
instance_type=$1

load_dir='/mnt/project/'
out="linear_args"
data_identifier="ukb20279"
# chroms=({1..22})
chroms=(18 17 16)

for chrom in "${chroms[@]}"; do

    # test run
    # if [[ $chrom != 18 ]]; then
    #     continue 
    # fi

    chrom_dir="${out}/${data_identifier}/chr${chrom}"
    linarg_dir_list=($(dx ls $chrom_dir))

    for dir in ${linarg_dir_list[@]}; do
        linarg_dir=${chrom_dir}/${dir}
        output_list=($(dx ls "${linarg_dir}"))
        if [[ " ${output_list[@]} " =~ " linear_arg.npz " ]]; then # skip partitions that have already been inferred
                echo "${linarg_dir} has already been merged."
                continue
        fi

        echo "${dir}, ${linarg_dir}, ${load_dir}"

        dx run app-swiss-army-knife \
            -iin="/amber/scripts/run_merge_brick_graphs.sh" \
            -icmd="bash run_merge_brick_graphs.sh $linarg_dir $load_dir" \
            --destination "/" \
            --instance-type $instance_type \
            --priority high \
            --name "linarg_merge_${dir}" \
            --brief \
            -y
    done
done
