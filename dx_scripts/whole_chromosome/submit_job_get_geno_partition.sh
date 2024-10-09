#!/bin/bash
instance_type=$1

out="linear_args"
data_identifier="ukb20279"
chrom=1
chrom_dir="${out}/${data_identifier}/chr${chrom}"

linarg_dir_list=($(dx ls $chrom_dir))

vcf_path="/mnt/project/Bulk/Previous WGS releases/GATK and GraphTyper WGS/SHAPEIT Phased VCFs/ukb20279_c${chrom}_b0_v1.vcf.gz"

for dir in ${linarg_dir_list[@]}
do
    linarg_dir=${chrom_dir}/${dir}

    mtx_list=($(dx ls "${linarg_dir}/genotype_matrices/"))

    dx download -f "${linarg_dir}/partitions.txt"
    IFS=$'\n' read -r -d '' -a partitions < <(awk '{print $0}' partitions.txt)
    for partition in "${partitions[@]:1}"
        do
            p=($partition)
            partition_region="${p[0]}-${p[2]}-${p[3]}"
            partition_number=${p[1]}

            if [[ " ${mtx_list[@]} " =~ " ${partition_number}_${partition_region}.npz " ]]; then # skip partitions that have already been inferred
                echo "${partition_number}_${partition_region}.npz already exists."
                continue
            fi

            echo $partition_region
            dx run app-swiss-army-knife \
                -iin="/amber/scripts/run_get_geno_partition.sh" \
                -icmd="bash run_get_geno_partition.sh \"$vcf_path\" $linarg_dir $partition_region $partition_number" \
                --destination "/" \
                --instance-type $instance_type \
                --priority low \
                --name "get_mat_${partition_region}" \
                --brief \
                -y
        done
done

rm partitions.txt