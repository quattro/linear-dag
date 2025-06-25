#!/bin/bash
instance_type=$1

load_dir='/mnt/project/'
out="linear_args"
data_identifier="ukb20279_maf_0.01"
# chroms=({1..22})
# chroms=(X)
chroms=({1..22} X)

for chrom in "${chroms[@]}"; do

    # test run
    # if [[ $chrom != 1 ]]; then
    #     continue 
    # fi

    chrom_dir="${out}/${data_identifier}/chr${chrom}"
    linarg_dir_list=($(dx ls $chrom_dir))

    for dir in ${linarg_dir_list[@]}; do
        linarg_dir=${chrom_dir}/${dir}
        output_list=($(dx ls "${linarg_dir}"))
        if [[ " ${output_list[@]} " =~ " linear_arg_individual.h5 " ]]; then # skip partitions that have already been inferred
                echo "${linarg_dir} already has individuals."
                continue
        fi

        echo "${dir}, ${linarg_dir}, ${load_dir}"

        dx run app-swiss-army-knife \
            -iin="/amber/scripts/run_add_individual_nodes.sh" \
            -icmd="bash run_add_individual_nodes.sh $linarg_dir $load_dir" \
            --destination "/" \
            --instance-type $instance_type \
            --priority high \
            --name "add_individuals_${dir}" \
            --brief \
            -y
    done
done
