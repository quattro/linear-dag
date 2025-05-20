#!/bin/bash
# instance_type=$1
instance_type="mem3_ssd1_v2_x2"

out="linear_args"
data_identifier="ukb20279_maf_0.01"
whitelist_path="/mnt/project/sample_metadata/ukb20279/250129_whitelist.txt"
chroms=({2..5})

for chrom in "${chroms[@]}"; do

    chrom_dir="${out}/${data_identifier}/chr${chrom}"
    linarg_dir_list=($(dx ls $chrom_dir))
    vcf_path="/mnt/project/Bulk/Previous WGS releases/GATK and GraphTyper WGS/SHAPEIT Phased VCFs/ukb20279_c${chrom}_b0_v1.vcf.gz"

    for dir in ${linarg_dir_list[@]}; do

        # if [[ $chrom != 1 ]]; then
        #     continue 
        # fi
        
        linarg_dir=${chrom_dir}/${dir}
        mtx_list=($(dx ls "${linarg_dir}/genotype_matrices/"))
        dx download -f "${linarg_dir}/partitions.txt"
        IFS=$'\n' read -r -d '' -a partitions < <(awk '{print $0}' partitions.txt)
        for partition in "${partitions[@]:1}"; do
            p=($partition)
            partition_region="${p[0]}-${p[2]}-${p[3]}"
            partition_number=${p[1]}

            if [[ " ${mtx_list[@]} " =~ " ${partition_number}_${partition_region}.h5 " ]]; then # skip partitions that have already been inferred
                echo "${partition_number}_${partition_region}.h5 already exists."
                continue
            fi

            echo $partition_region
            dx run app-swiss-army-knife \
                -iin="/amber/scripts/run_get_geno_partition_maf_0.01.sh" \
                -icmd="bash run_get_geno_partition_maf_0.01.sh \"$vcf_path\" $linarg_dir $partition_region $partition_number $whitelist_path" \
                --destination "/" \
                --instance-type $instance_type \
                --priority high \
                --name "get_mat_${partition_region}" \
                --brief \
                --extra-args '{"executionPolicy": {"maxRestarts": 5}}' \
                -y
        done
    done
    rm partitions.txt
done