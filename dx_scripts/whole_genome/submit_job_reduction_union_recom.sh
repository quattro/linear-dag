#!/bin/bash
instance_type=$1

load_dir='/mnt/project/'
out="linear_args"
data_identifier="ukb20279_maf_0.01"
# data_identifier="ukb20279"
# chroms=({1..22})
chroms=(X)

for chrom in "${chroms[@]}"; do

    # if [[ $chrom != 6 ]]; then
    #     continue 
    # fi

    chrom_dir="${out}/${data_identifier}/chr${chrom}"
    linarg_dir_list=($(dx ls $chrom_dir))

    for dir in ${linarg_dir_list[@]}; do
    
        linarg_dir=${chrom_dir}/${dir}
        mtx_list=($(dx ls "${linarg_dir}/genotype_matrices/"))
        brick_graph_list=($(dx ls "${linarg_dir}/brick_graph_partitions/"))

        for f in "${mtx_list[@]}"; do
            partition_identifier=$(echo "$f" | awk -F. '{print $1}')
            if [[ " ${brick_graph_list[@]} " =~ " ${partition_identifier}.h5 " ]]; then # skip partitions that have already been inferred
                echo "${partition_identifier}.h5 already exists."
                continue
            fi

            echo "${linarg_dir} ${load_dir} ${partition_identifier}"

            echo $f
            dx run app-swiss-army-knife \
                -iin="/amber/scripts/run_reduction_union_recom.sh" \
                -icmd="bash run_reduction_union_recom.sh $linarg_dir $load_dir $partition_identifier" \
                --destination "/" \
                --instance-type $instance_type \
                --priority high \
                --name "reduction_union_recom_${partition_identifier}" \
                --brief \
                -y
        done
    done
done