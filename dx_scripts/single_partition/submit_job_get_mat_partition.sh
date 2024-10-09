#!/bin/bash
region=$1
out_prefix=$2
partition_size=$3
instance_type=$4

chr=$(echo "$region" | awk -F- '{print $1}')
start=$(echo "$region" | awk -F- '{print $2}')
end=$(echo "$region" | awk -F- '{print $3}')
integer=$((start))
integer=$((end))

vcf_path="/mnt/project/Bulk/Previous WGS releases/GATK and GraphTyper WGS/SHAPEIT Phased VCFs/ukb20279_c${chr:3}_b0_v1.vcf.gz"
linarg_dir="linear_arg_results/${out_prefix}_${region}/"

n_partitions=$(echo "scale=0; (($end - $start + 1) + $parition_size - 1) / $partition_size" | bc) # region is inclusive

for i in $(seq 0 $n_partitions); do
  partition_start=$(($start + i*$partition_size))
  partition_end=$(printf "%d\n%d" $(($partition_start + $partition_size - 1)) $end | sort -n | head -n1) # region is inclusive
  partition_region="$chr-$partition_start-$partition_end"
  echo $partition_region

  dx run app-swiss-army-knife \
      -iin="/amber/scripts/run_get_mat_partition.sh" \
      -icmd="bash run_get_mat_partition.sh \"$vcf_path\" $linarg_dir $partition_region $i" \
      --destination "/" \
      --instance-type $instance_type \
      --priority low \
      --name "get_mat_${out_prefix}_${region}_${i}_${partition_region}" \
      --brief \
      -y
done