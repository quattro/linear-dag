import numpy as np
import os


# start and end are inclusive
def get_partitions(interval_start, interval_end, partition_size):
    num_partitions = int((interval_end - interval_start + 1) / partition_size)
    ticks = np.linspace(interval_start, interval_end+1, num_partitions).astype(int)
    ends = ticks[1:] - 1
    starts = ticks[:-1]
    intervals = [(start, end) for start,end in zip(starts, ends)]
    print(f'size of intervals: {[x[1]-x[0]+1 for x in intervals]}')
    return intervals


def partition_chromosome(chrom, chrom_start, chrom_end, large_partition_size, small_partition_size, data_identifier, out):
    large_partitions = get_partitions(chrom_start, chrom_end, large_partition_size)
    small_partitions = [get_partitions(i[0], i[1], small_partition_size) for i in large_partitions]

    for i in range(len(large_partitions)):
        start, end = large_partitions[i]
        large_partition_dir = f'{out}/{data_identifier}/chr{chrom}/{i}_chr{chrom}-{start}-{end}'
        if not os.path.exists(large_partition_dir): os.makedirs(large_partition_dir)
        
        with open(f'{large_partition_dir}/partitions.txt', 'w') as f:
            f.write(' '.join(['chrom', 'partition_number', 'start', 'end'])+'\n')
            for j in range(len(small_partitions[i])):
                partition = [f'chr{chrom}', j, small_partitions[i][j][0], small_partitions[i][j][1]]
                f.write(' '.join([str(x) for x in partition])+'\n')
    

if __name__ == "__main__":
    
    chrom = 1
    chrom_start = 15916  # position of first variant found manually
    chrom_end = 248945618 # position of last variant found manually
    
    large_partition_size = 2e7
    small_partition_size = 1e6
    
    data_identifier = 'ukb20279'
    out = 'linear_args'
    
    partition_chromosome(chrom, chrom_start, chrom_end, large_partition_size, small_partition_size, data_identifier, out)