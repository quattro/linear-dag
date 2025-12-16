import gzip
import os
import time
import subprocess

from os import PathLike
from typing import Optional, Union

import h5py
import numpy as np
import polars as pl
import scipy.sparse as sp

from .core.brick_graph import BrickGraph, merge_brick_graphs, read_graph_from_disk
from .core.lineararg import LinearARG, make_triangular, remove_degree_zero_nodes
from .core.one_summed_cy import linearize_brick_graph
from .core.recombination import Recombination
from .genotype import read_vcf
from .memory_logger import MemoryLogger


def compress_vcf(
    input_vcf: Union[str, PathLike],
    output_h5: Union[str, PathLike],
    region: Optional[str] = None,
    keep_path: Optional[Union[str, PathLike]] = None,
    flip_minor_alleles: bool = False,
    maf_filter: Optional[float] = None,
    remove_indels: bool = False,
    add_individual_nodes: bool = False,
):
    """
    Compress a VCF file into a LinearARG HDF5 file.
    """
    logger = MemoryLogger(__name__)
    logger.info("Starting compression")

    if keep_path is not None:
        include_samples = load_sample_ids(keep_path)
    else:
        include_samples = None

    linarg = LinearARG.from_vcf(
        path=input_vcf,
        region=region,
        include_samples=include_samples,
        flip_minor_alleles=flip_minor_alleles,
        maf_filter=maf_filter,
        snps_only=remove_indels,
    )
    logger.info(f"Number of variants: {linarg.shape[1]}")
    logger.info(f"Number of samples: {linarg.shape[0]}")
    logger.info(f"Number of nonzeros: {linarg.nnz}")
    num_minor_alleles = linarg.shape[0] * np.sum(np.minimum(linarg.allele_frequencies, 1 - linarg.allele_frequencies))
    logger.info(f"Number of minor alleles: {num_minor_alleles}")
    logger.info(f"Compression ratio: {num_minor_alleles / linarg.nnz}")

    if add_individual_nodes:
        linarg = linarg.add_individual_nodes()
    else:
        linarg.calculate_nonunique_indices()

    if region:
        chrom, pos = region.split(":")
        start, end = pos.split("-")
        block_info = {"chrom": chrom.strip("chr"), "start": int(start), "end": int(end)}
    else:
        block_info = None

    logger.info("Writing to disk")
    linarg.write(output_h5, block_info=block_info)
    logger.info("Done!")
    

def msc_step0(
    vcf_metadata: Union[str, PathLike],
    large_partition_size: int,
    n_small_blocks: int,
    out: Union[str, PathLike],
    flip_minor_alleles: bool = False,
    keep: Optional[Union[str, PathLike]] = None,
    maf: Optional[float] = None,
    remove_indels: bool = False,
    sex_path: Optional[Union[str, PathLike]] = None,
    mount_point: Optional[Union[str, PathLike]] = None,
):
    """
    Partitions chromosomes into large blocks of size large_partition_size and smaller blocks of
    size partition_size / n_small_blocks. Output jobs_metadata.txt containing linear ARG parameters
    required for steps 1-5 in the multi-step compress pipeline.
    """
    os.makedirs(out, exist_ok=True)
    
    vcf_meta = pl.read_csv(vcf_metadata, separator=" ")
    # Initialize with proper dtypes
    job_meta = pl.DataFrame({
        "small_job_id": pl.Series(dtype=pl.Int64),
        "large_job_id": pl.Series(dtype=pl.Int64),
        "small_region": pl.Series(dtype=pl.Utf8),
        "large_region": pl.Series(dtype=pl.Utf8),
        "vcf_path": pl.Series(dtype=pl.Utf8)
    })
    
    n_large_jobs = 0
    for chrom in vcf_meta["chr"].unique():
        chrom_meta = vcf_meta.filter(pl.col("chr") == chrom)
        
        # accept chrom as either 22 or chr22 as cyvcf2 requires 'chr' as a prefix
        if chrom[:3] != 'chr': 
            chrom_name = f'chr{chrom}'
        else:
            chrom_name = chrom
        
        vcf_path = chrom_meta["vcf_path"].item()
        
        first_coord = int(subprocess.check_output(
            f"bcftools query -f '%POS\\n' \"{vcf_path}\" | head -n 1", 
            shell=True, text=True
        ).strip())
        
        last_coord = int(subprocess.check_output(
            f"bcftools query -f '%POS\\n' \"{vcf_path}\" | tail -n 1", 
            shell=True, text=True
        ).strip())
        
        large_partitions = get_partitions(first_coord, last_coord, large_partition_size)
        small_partition_size = int(large_partition_size / n_small_blocks)
        small_partitions = [get_partitions(start, end, small_partition_size) 
                          for start, end in large_partitions]
        
        for i, (large_start, large_end) in enumerate(large_partitions):
            for small_start, small_end in small_partitions[i]:
                new_row = pl.DataFrame({
                    "small_job_id": [job_meta.height],
                    "large_job_id": [n_large_jobs],
                    "small_region": [f"{chrom_name}:{small_start}-{small_end}"],
                    "large_region": [f"{chrom_name}:{large_start}-{large_end}"],
                    "vcf_path": [vcf_path]
                })
                job_meta = pl.concat([job_meta, new_row])
            n_large_jobs += 1
    
    params = {
        "large_partition_size": str(large_partition_size),
        "n_small_blocks": str(n_small_blocks),
        "out": str(out),
        "flip_minor_alleles": str(flip_minor_alleles),
        "keep": str(keep),
        "maf": str(maf),
        "remove_indels": str(remove_indels),    
        "sex_path": str(sex_path),
        "mount_point": "" if mount_point is None else str(mount_point),
    }
    
    print(f'n_small_jobs: {job_meta.height}, n_large_jobs: {n_large_jobs}')
    job_meta.write_parquet(f"{out}/job_metadata.parquet", metadata=params)


def msc_step1(
    jobs_metadata: Union[str, PathLike],
    small_job_id: int,
):
    
    job_meta = pl.read_parquet(jobs_metadata)
    job = job_meta.filter(pl.col("small_job_id") == small_job_id)
    vcf_path = job["vcf_path"].item()
    region = job["small_region"].item()
    
    params = pl.read_parquet_metadata(jobs_metadata)
    flip_minor_alleles = params['flip_minor_alleles']
    keep = None if params['keep'] == "None" else params['keep']
    maf = None if params['maf'] == "None" else float(params['maf'])
    remove_indels = True if params['remove_indels'] == "True" else False
    sex_path = None if params['sex_path'] == "None" else params['sex_path']
    mount_point = params['mount_point']
    out = params['out']
    
    if os.path.exists(f"{mount_point}{out}/genotype_matrices/{small_job_id}_{region}.h5"):
        print(f"Genotype matrix for {small_job_id}_{region} already exists. Skipping.")
    else:
        make_genotype_matrix(
            vcf_path=vcf_path,
            out=out,
            region=region,
            partition_number=small_job_id,
            flip_minor_alleles=flip_minor_alleles,
            samples_path=keep,
            maf_filter=maf,
            remove_indels=remove_indels,
            sex_path=sex_path,
        )
    
    if os.path.exists(f"{mount_point}{out}/forward_backward_graphs/{small_job_id}_{region}_forward_graph.h5"):
        print(f"Forward backward graph for {small_job_id}_{region} already exists. Skipping.")
    else:
        run_forward_backward(out, "", f"{small_job_id}_{region}") # no mount point needed since matrices are generated in the same job
    
    
def msc_step2(
    jobs_metadata: Union[str, PathLike],
    small_job_id: int,
):
    job_meta = pl.read_parquet(jobs_metadata)
    job = job_meta.filter(pl.col("small_job_id") == small_job_id)
    region = job["small_region"].item()
    
    params = pl.read_parquet_metadata(jobs_metadata)
    mount_point = params['mount_point']
    out = params['out']
    
    if not os.path.exists(f"{mount_point}{out}/forward_backward_graphs/{small_job_id}_{region}_forward_graph.h5"):
        raise FileNotFoundError(f"{mount_point}{out}/forward_backward_graphs/{small_job_id}_{region}_forward_graph.h5 does not exist. Please run step 1 before step 2.")
    if os.path.exists(f"{mount_point}{out}/brick_graph_partitions/{small_job_id}_{region}.h5"):
        print(f"Brick graph for {small_job_id}_{region} already exists. Skipping.")
    else:
        reduction_union_recom(out, mount_point, f"{small_job_id}_{region}") 


def msc_step3(
    jobs_metadata: Union[str, PathLike],
    large_job_id: int,
):
    job_meta = pl.read_parquet(jobs_metadata)
    jobs = job_meta.filter(pl.col("large_job_id") == large_job_id)
    large_region = jobs["large_region"].to_list()[0]
    
    params = pl.read_parquet_metadata(jobs_metadata)
    mount_point = params['mount_point']
    out = params['out']
    
    if os.path.exists(f"{mount_point}{out}/linear_args/{large_job_id}_{large_region}.h5"):
        print(f"Linear ARG for {large_job_id}_{large_region} already exists. Skipping.")
        return
    
    # check that all brick graphs have been generated before merging
    partition_identifiers = []
    for row in jobs.iter_rows(named=True):
        small_job_id = row['small_job_id']
        small_region = row['small_region']
        partition_identifier = f"{small_job_id}_{small_region}"
        if not os.path.exists(f"{mount_point}{out}/brick_graph_partitions/{partition_identifier}.h5"):
            raise FileNotFoundError(f"{mount_point}{out}/brick_graph_partitions/{partition_identifier}.h5 does not exist. Please run step 2 on all partitions before step 3.")
        with h5py.File(f"{mount_point}{out}/brick_graph_partitions/{partition_identifier}.h5", "r") as f:
            is_empty = f.attrs.get("is_empty", False)
        if is_empty:
            continue
        partition_identifiers.append(partition_identifier)
    large_partition_identifier = f"{large_job_id}_{large_region}"
    
    if len(partition_identifiers) == 0:
        with h5py.File(f"{out}/linear_args/{large_partition_identifier}.h5", "w") as f:
            f.attrs["is_empty"] = True
        return
    
    merge(out, mount_point, partition_identifiers, large_partition_identifier)    


def msc_step4(
    jobs_metadata: Union[str, PathLike],
    large_job_id: int,
):
    job_meta = pl.read_parquet(jobs_metadata)
    jobs = job_meta.filter(pl.col("large_job_id") == large_job_id)
    large_region = jobs["large_region"].to_list()[0]
    large_partition_identifier = f'{large_job_id}_{large_region}'
    
    params = pl.read_parquet_metadata(jobs_metadata)
    mount_point = params['mount_point']
    out = params['out']
    
    if os.path.exists(f"{mount_point}{out}/individual_linear_args/{large_job_id}_{large_region}.h5"):
        print(f"Individual linear ARG for {large_job_id}_{large_region} already exists. Skipping.")
        return
    
    partition_identifiers = []
    for row in jobs.iter_rows(named=True):
        small_job_id = row['small_job_id']
        small_region = row['small_region']
        p_id = f"{small_job_id}_{small_region}"
        if not os.path.exists(f"{mount_point}{out}/brick_graph_partitions/{p_id}.h5"):
            raise FileNotFoundError(f"{mount_point}{out}/brick_graph_partitions/{p_id}.h5 does not exist. Please run step 2 on all partitions before step 4.")
        with h5py.File(f"{mount_point}{out}/brick_graph_partitions/{p_id}.h5", "r") as f:
            is_empty = f.attrs.get("is_empty", False)
        if is_empty:
            continue
        partition_identifiers.append(p_id)
        
    if len(partition_identifiers) == 0:
        with h5py.File(f"{out}/individual_linear_args/{large_partition_identifier}.h5", "w") as f:
            f.attrs["is_empty"] = True
        return
            
    add_individuals_to_linarg(out, mount_point, partition_identifiers, large_partition_identifier)
  
    
def msc_step5(
    jobs_metadata: Union[str, PathLike],
):
    """
    Merge multiple LinearARG HDF5 files into a single file.
    
    Args:
        jobs_metadata: Path to the jobs metadata parquet file
    """
    params = pl.read_parquet_metadata(jobs_metadata)
    mount_point = params['mount_point']
    out = params['out']
    
    os.makedirs(f"{out}/logs", exist_ok=True)
    logger = MemoryLogger(__name__, log_file=f"{out}/logs/msc_step5.log")
    logger.info("Starting merge of LinearARG partitions")
    
    job_meta = pl.read_parquet(jobs_metadata)
    
    partition_identifiers = set(
        f"{i}_{region}" 
        for i, region in zip(
            job_meta["large_job_id"].to_list(), 
            job_meta["large_region"].to_list()
        )
    )
        
    params = pl.read_parquet_metadata(jobs_metadata)
    mount_point = params['mount_point']
    out = params['out']
    
    final_merge(out, mount_point, partition_identifiers, logger)
    if os.path.exists(f"{mount_point}{out}/individual_linear_args/"):
        final_merge(out, mount_point, partition_identifiers, logger, individual=True)


def final_merge(
    out,
    mount_point,
    partition_identifiers,
    logger,
    individual=False,
):
    
    non_empty_partition_identifiers = []
    
     # check that all linear ARGs have been inferred and are correct
    for part_id in partition_identifiers:
        stats_path = f"{mount_point}{out}/linear_args/{part_id}_stats.txt"
        linarg_path = f"{mount_point}{out}/linear_args/{part_id}.h5"
        if not os.path.exists(linarg_path):
            raise FileNotFoundError(
                f"Linear ARG not found: {linarg_path}. "
                "Please run step 4 on all partitions before step 5."
            )
        with h5py.File(f"{mount_point}{out}/linear_args/{part_id}.h5", "r") as f:
            is_empty = f.attrs.get("is_empty", False)
        if is_empty:
            continue
        non_empty_partition_identifiers.append(part_id)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Stats file not found: {stats_path}. "
                "Please run step 4 on all partitions before step 5."
            )
        if not is_allele_count_correct(stats_path):
            raise ValueError(f"Allele counts do not match for partition {part_id}.")

    if individual:
        output_path = f"{out}/linear_arg_individual.h5"
    else:
        output_path = f"{out}/linear_arg.h5"
    logger.info(f"Merging {len(non_empty_partition_identifiers)} partitions into {output_path}")
    with h5py.File(output_path, "w") as merged:
        iids_saved = False
        
        for part_id in non_empty_partition_identifiers:
            if individual:
                src_path = f"{mount_point}{out}/individual_linear_args/{part_id}.h5"
            else:
                src_path = f"{mount_point}{out}/linear_args/{part_id}.h5"
            
            with h5py.File(src_path, "r") as src:
                # handle iids specially
                if not iids_saved and 'iids' in src:
                    merged.create_dataset(
                        'iids', 
                        data=src['iids'][:],
                        compression='gzip',
                        shuffle=True
                    )
                    iids_saved = True
                
                # Copy everything except 'iids' using the recursive function
                for key in src.keys():
                    if key == 'iids':
                        continue
                    
                    if key not in merged:
                        merged.create_group(key)
                    copy_h5_group(src[key], merged[key])
        
        if not iids_saved:
            raise ValueError("No 'iids' dataset found in any of the input files")
    logger.info("Successfully merged all partitions")
    


def copy_h5_group(src_group, dest_group):
    """Recursively copy all datasets and groups from src_group to dest_group"""
    for key in src_group.keys():
        if isinstance(src_group[key], h5py.Dataset):
            if key in dest_group:
                del dest_group[key]  # Remove if exists to avoid conflicts
            src_group.copy(key, dest_group, name=key)
        elif isinstance(src_group[key], h5py.Group):
            # Recursively copy subgroups
            if key not in dest_group:
                dest_group.create_group(key)
            copy_h5_group(src_group[key], dest_group[key])
    
    # Copy attributes
    for attr_key, attr_val in src_group.attrs.items():
        dest_group.attrs[attr_key] = attr_val
    
    

def get_partitions(interval_start, interval_end, partition_size):
    """
    Returns a list of (start, end) tuples representing partitions of the interval [interval_start, interval_end]
    where start and end are inclusive.
    """
    num_partitions = int((interval_end - interval_start + 1) / partition_size)
    ticks = np.linspace(interval_start, interval_end+1, num_partitions+1).astype(int)
    ends = ticks[1:] - 1
    starts = ticks[:-1]
    intervals = [(start, end) for start,end in zip(starts, ends)]
    return intervals 


def make_genotype_matrix(
    vcf_path: Union[str, PathLike],
    out: Union[str, PathLike],
    region: str,
    partition_number: int,
    phased: bool = True,
    flip_minor_alleles: bool = False,
    samples_path: Optional[Union[str, PathLike]] = None,
    maf_filter: Optional[float] = None,
    remove_indels: bool = False,
    sex_path=None,
):
    """
    From a vcf file, save the genotype matrix and variant metadata for the given region.
    """
    os.makedirs(f"{out}/logs/", exist_ok=True)
    os.makedirs(f"{out}/variant_metadata/", exist_ok=True)
    os.makedirs(f"{out}/genotype_matrices/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{out}/logs/{partition_number}_{region}_make_genotype_matrix.log")

    if samples_path is not None:
        samples = load_sample_ids(samples_path)
    else:
        samples = None

    if sex_path is None:
        sex = None
    else:
        with open(sex_path, "r") as f:
            sex = np.array([int(line.strip()) for line in f])

    logger.info("Reading vcf as sparse matrix")
    t1 = time.time()
    genotypes, flip, v_info, iids = read_vcf(
        vcf_path,
        phased=phased,
        region=region,
        flip_minor_alleles=flip_minor_alleles,
        samples=samples,
        maf_filter=maf_filter,
        remove_indels=remove_indels,
        sex=sex,
    )
    if genotypes is None:
        logger.info("No variants found")
        with h5py.File(f"{out}/genotype_matrices/{partition_number}_{region}.h5", "w") as f:
            f.attrs["is_empty"] = True
        
        with open(f"{out}/variant_metadata/{partition_number}_{region}.txt", "w") as f:
            f.write("# No variants found in this region\n")
        
        return None
    
    if phased:
        iids = [id_ for id_ in iids for _ in range(2)]

    t2 = time.time()
    logger.info(f"vcf to sparse matrix completed in {np.round(t2 - t1, 3)} seconds")
    logger.info("Saving genotype matrix and variant metadata")
    
    with h5py.File(f"{out}/genotype_matrices/{partition_number}_{region}.h5", "w") as f:
        f.create_dataset("shape", data=genotypes.shape, compression="gzip", shuffle=True)
        f.create_dataset("indptr", data=genotypes.indptr, compression="gzip", shuffle=True)
        f.create_dataset("indices", data=genotypes.indices, compression="gzip", shuffle=True)
        f.create_dataset("data", data=genotypes.data, compression="gzip", shuffle=True)
        f.create_dataset("flip", data=flip, compression="gzip", shuffle=True)
        f.create_dataset("iids", data=iids, compression="gzip", shuffle=True)
        if sex_path is not None:
            f.create_dataset("sex", data=sex, compression="gzip", shuffle=True)
        f.attrs["is_empty"] = False
    
    v_info.write_csv(f"{out}/variant_metadata/{partition_number}_{region}.txt", separator=" ")


def run_forward_backward(out, mount_point, partition_identifier):
    """
    Run the forward and backward brick graph algorithms on the genotype matrix.
    
    Args:
        out: Output directory
        mount_point: Mount point for cloud storage (empty string if local)
        partition_identifier: Identifier for the partition being processed
    """
    os.makedirs(f"{out}/logs/", exist_ok=True)
    os.makedirs(f"{out}/forward_backward_graphs/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{out}/logs/{partition_identifier}_forward_backward.log")
    logger.info("Loading genotype matrix")
    t1 = time.time()
    with h5py.File(f"{mount_point}{out}/genotype_matrices/{partition_identifier}.h5", "r") as f:
        is_empty = f.attrs.get("is_empty", False)
        if is_empty:
            with h5py.File(f"{out}/forward_backward_graphs/{partition_identifier}_forward_graph.h5", "w") as f:
                f.attrs["is_empty"] = True
            with h5py.File(f"{out}/forward_backward_graphs/{partition_identifier}_backward_graph.h5", "w") as f:
                f.attrs["is_empty"] = True
            with open(f"{out}/forward_backward_graphs/{partition_identifier}_sample_indices.txt", "w") as f:
                f.write("# No variants found in this region\n")
            logger.info("Genotype matrix is empty")
            return None
        genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])
    t2 = time.time()
    logger.info(f"Genotype matrix loaded in {np.round(t2 - t1, 3)} seconds")
    logger.info("Running forward and backward brick graph algorithms")
    t3 = time.time()
    sample_indices = BrickGraph.forward_backward(
        genotypes,
        add_samples=True,
        save_to_disk=True,
        out=f"{out}/forward_backward_graphs/{partition_identifier}",
    )
    np.savetxt(f"{out}/forward_backward_graphs/{partition_identifier}_sample_indices.txt", sample_indices)
    t4 = time.time()
    logger.info(f"Forward and backward brick graph algorithms completed in {np.round(t4 - t3, 3)} seconds")


def reduction_union_recom(out, mount_point, partition_identifier):
    """
    Compute the transitive reduction of the union of the forward and backward graphs
    and find recombinations to obtain the brick graph.
    """
    os.makedirs(f"{out}/logs/", exist_ok=True)
    os.makedirs(f"{out}/brick_graph_partitions/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{out}/logs/{partition_identifier}_reduction_union_recom.log")

    logger.info("Loading genotypes, forward and backward graphs, and sample indices")
    t1 = time.time()
    with h5py.File(f"{mount_point}{out}/genotype_matrices/{partition_identifier}.h5", "r") as f:
        is_empty = f.attrs.get("is_empty", False)
        if is_empty:
            with h5py.File(f"{out}/brick_graph_partitions/{partition_identifier}.h5", "w") as f:
                f.attrs["is_empty"] = True
            return None
        genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])
    n, m = genotypes.shape
    forward_graph = read_graph_from_disk(
        f"{mount_point}{out}/forward_backward_graphs/{partition_identifier}_forward_graph.h5"
    )
    backward_graph = read_graph_from_disk(
        f"{mount_point}{out}/forward_backward_graphs/{partition_identifier}_backward_graph.h5"
    )
    sample_indices = np.loadtxt(
        f"{mount_point}{out}/forward_backward_graphs/{partition_identifier}_sample_indices.txt"
    )
    t2 = time.time()
    logger.info(f"Loaded in {np.round(t2 - t1, 3)} seconds")

    logger.info("Combining nodes and computing reduction union")
    t3 = time.time()
    brick_graph, variant_indices = BrickGraph.combine_graphs(forward_graph, backward_graph, m)
    t4 = time.time()
    logger.info(f"Combined nodes and computed reduction union in {np.round(t4 - t3, 3)} seconds")

    for i in sample_indices:
        assert len(list(brick_graph.successors(int(i)))) == 0

    logger.info("Finding recombinations")
    t5 = time.time()
    recom = Recombination.from_graph(brick_graph)
    recom.find_recombinations()
    t6 = time.time()
    logger.info(f"Found recombinations in {np.round(t6 - t5, 3)} seconds")

    for i in sample_indices:
        assert len(list(recom.successors(int(i)))) == 0

    adj_mat = recom.to_csc()

    logger.info("Saving brick graph")
    with h5py.File(f"{out}/brick_graph_partitions/{partition_identifier}.h5", "w") as f:
        f.attrs["n"] = adj_mat.shape[0]
        f.create_dataset("indptr", data=adj_mat.indptr, compression="gzip", shuffle=True)
        f.create_dataset("indices", data=adj_mat.indices, compression="gzip", shuffle=True)
        f.create_dataset("data", data=adj_mat.data, compression="gzip", shuffle=True)
        f.create_dataset("variant_indices", data=np.array(variant_indices), compression="gzip", shuffle=True)
        f.create_dataset("sample_indices", data=np.array(sample_indices), compression="gzip", shuffle=True)

    af = np.diff(genotypes.indptr) / n
    geno_nnz = np.sum(n * np.minimum(af, 1 - af))
    nnz_ratio = geno_nnz / adj_mat.nnz
    logger.info(f"Stats - n: {n}, m: {m}, geno_nnz: {geno_nnz}, brickgraph_nnz: {adj_mat.nnz}, nnz_ratio: {nnz_ratio}")


def infer_brick_graph(out, mount_point, partition_identifier):
    """
    From a genotype matrix, infer the brick graph and find recombinations.
    """
    os.makedirs(f"{out}/logs/", exist_ok=True)
    os.makedirs(f"{out}/brick_graph_partitions/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{out}/logs/{partition_identifier}_infer_brick_graph.log")
    logger.info("Loading genotype matrix")
    t1 = time.time()
    with h5py.File(f"{mount_point}{out}/genotype_matrices/{partition_identifier}.h5", "r") as f:
        genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])
    n, m = genotypes.shape
    t2 = time.time()
    logger.info(f"Genotype matrix loaded in {np.round(t2 - t1, 3)} seconds")
    logger.info("Inferring brick graph")
    brick_graph, sample_indices, variant_indices = BrickGraph.from_genotypes(genotypes)
    t3 = time.time()
    logger.info(f"Brick graph inferred in {np.round(t3 - t2, 3)} seconds")
    logger.info("Finding recombinations")
    recom = Recombination.from_graph(brick_graph)
    recom.find_recombinations()
    t4 = time.time()
    logger.info(f"Recombinations found in {np.round(t4 - t3, 3)} seconds")
    adj_mat = recom.to_csc()

    logger.info("Saving brick graph")
    with h5py.File(f"{out}/brick_graph_partitions/{partition_identifier}.h5", "w") as f:
        f.attrs["n"] = adj_mat.shape[0]
        f.create_dataset("indptr", data=adj_mat.indptr, compression="gzip", shuffle=True)
        f.create_dataset("indices", data=adj_mat.indices, compression="gzip", shuffle=True)
        f.create_dataset("data", data=adj_mat.data, compression="gzip", shuffle=True)
        f.create_dataset("variant_indices", data=np.array(variant_indices), compression="gzip", shuffle=True)
        f.create_dataset("sample_indices", data=np.array(sample_indices), compression="gzip", shuffle=True)

    af = np.diff(genotypes.indptr) / n
    geno_nnz = np.sum(n * np.minimum(af, 1 - af))
    nnz_ratio = geno_nnz / adj_mat.nnz
    logger.info(f"Stats - n: {n}, m: {m}, geno_nnz: {geno_nnz}, brickgraph_nnz: {adj_mat.nnz}, nnz_ratio: {nnz_ratio}")


def merge(out, mount_point, partition_identifiers, partition_identifier):
    """
    Merged partitioned brick graphs, find recombinations, and linearize.
    """

    os.makedirs(f"{out}/logs/", exist_ok=True)
    os.makedirs(f"{out}/linear_args/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{out}/logs/merge.log")
    logger.info("Merging brick graphs")
    t1 = time.time()

    merged_graph, variant_indices, num_samples, index_mapping = merge_brick_graphs(
        f"{mount_point}{out}/brick_graph_partitions", partition_identifiers
    )
    t2 = time.time()
    logger.info(f"Brick graphs merged in {np.round(t2 - t1, 3)} seconds")

    logger.info("Finding recombinations")
    t3 = time.time()
    merged_graph_recom = Recombination.from_graph(merged_graph)
    merged_graph_recom.find_recombinations()
    t4 = time.time()
    logger.info(f"Recombinations found in {np.round(t4 - t3, 3)} seconds")
    logger.info("Linearizing brick graph")
    t5 = time.time()
    A = sp.csc_matrix(linearize_brick_graph(merged_graph_recom))
    t6 = time.time()
    logger.info(f"Linearized brick graph in {np.round(t6 - t5, 3)} seconds")
    sample_indices = np.arange(num_samples)

    logger.info("Reading variant metadata and flip")
    
    df_list = [pl.read_csv(f"{mount_point}{out}/variant_metadata/{p_id}.txt", separator=" ") for p_id in partition_identifiers]
    df = pl.concat(df_list)
    var_info = df.lazy()
    flip = []
    iids = None
    sex = None
    for p_id in partition_identifiers:
        with h5py.File(f"{mount_point}{out}/genotype_matrices/{p_id}.h5", "r") as f:
            flip_partition = list(f["flip"][:])
            if iids is None:
                iids = [iid.decode('utf-8') for iid in list(f["iids"][:])]
            if "sex" in f and sex is None:
                sex = f["sex"][:]
        flip += flip_partition
    flip = np.array(flip)
    iids = pl.Series("iids", iids)

    logger.info("Triangularizing and computing nonunique indices")
    A_filt, variant_indices_reindexed, sample_indices_reindexed = remove_degree_zero_nodes(
        A, variant_indices, sample_indices
    )
    A_tri, variant_indices_tri = make_triangular(A_filt, variant_indices_reindexed, sample_indices_reindexed)
    linarg = LinearARG(A_tri, variant_indices_tri, flip, len(sample_indices), variants=var_info, sex=sex, iids=iids)
    linarg.calculate_nonunique_indices()
    logger.info("Saving linear ARG")
    
    block = {
        "chrom": partition_identifier.split("_")[1].split(":")[0],
        "start": partition_identifier.split("_")[1].split(":")[1].split("-")[0],
        "end": partition_identifier.split("_")[1].split(":")[1].split("-")[1],
    }
    
    linarg.write(f"{out}/linear_args/{partition_identifier}.h5", block_info=block)
    logger.info("Computing linear ARG stats")
    get_linarg_stats(out, mount_point, partition_identifiers, partition_identifier, linarg)

    return


def get_linarg_stats(out, mount_point, partition_identifiers, partition_identifier, linarg=None):
    """
    Get stats from linear ARG.
    """
    if linarg is None:
        linarg = LinearARG.read(f"{mount_point}{out}/linear_args/{partition_identifier}.h5", block=partition_identifier.split("_")[1])

    linarg.flip = np.zeros(linarg.shape[1], dtype=bool)

    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg
    
    genotypes_nnz = 0
    allele_counts = []
    for p_id in partition_identifiers:
        with h5py.File(f"{mount_point}{out}/genotype_matrices/{p_id}.h5", "r") as f:
            genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])

        genotypes_nnz += np.sum(
            [
                np.minimum(genotypes[:, i].nnz, genotypes.shape[0] - genotypes[:, i].nnz)
                for i in range(genotypes.shape[1])
            ]
        )
        v_0 = np.ones(genotypes.shape[0])
        allele_count_from_genotypes = v_0 @ genotypes
        allele_counts.append(allele_count_from_genotypes)
    allele_counts_from_genotype = np.concatenate(allele_counts)

    stats = [
        linarg.shape[0],
        linarg.shape[1],
        genotypes_nnz,
        linarg.nnz,
        np.round(genotypes_nnz / linarg.nnz, 3),
        np.all(allele_counts_from_genotype == allele_count_from_linarg),
    ]
    stats = [str(x) for x in stats]
    with open(f"{out}/linear_args/{partition_identifier}_stats.txt", "w") as file:
        file.write(
            " ".join(
                [
                    "n",
                    "m",
                    "genotypes_nnz",
                    "linarg_nnz",
                    "linarg_nnz_ratio",
                    "correct_allele_counts",
                ]
            )
            + "\n"
        )
        file.write(" ".join(stats) + "\n")
        
    return
    
    
def is_allele_count_correct(stats_file_path):
    with open(stats_file_path, 'r') as f:
        # Read header and data lines
        header = f.readline().strip().split()
        data = f.readline().strip().split()
        
        # Find the index of 'correct_allele_counts' in the header
        try:
            idx = header.index('correct_allele_counts')
            # Convert the string to boolean
            return data[idx].lower() == 'true'
        except (ValueError, IndexError):
            return False


def add_individuals_to_linarg(out, mount_point, partition_identifiers, partition_identifier):
    os.makedirs(f"{out}/logs/", exist_ok=True)
    os.makedirs(f"{out}/individual_linear_args/", exist_ok=True)    
    logger = MemoryLogger(__name__, log_file=f"{out}/logs/add_individual_nodes.log")

    logger.info("Loading linear ARG")
    t1 = time.time()
    with h5py.File(str(f"{mount_point}{out}/linear_args/{partition_identifier}.h5"), "r") as f:
        block_name = list(f.keys())[0]
    temp = LinearARG.read(f"{mount_point}{out}/linear_args/{partition_identifier}.h5", block=block_name)
    t2 = time.time()
    logger.info(f"Linear ARG loaded in {np.round(t2 - t1, 3)} seconds")
    
    logger.info("Adding individual nodes to the linear ARG")
    t3 = time.time()
    linarg = temp.add_individual_nodes()
    t4 = time.time()
    logger.info(f"Individual nodes added in {np.round(t4 - t3, 3)} seconds")

    logger.info("Saving linear ARG")
    block_info = {
        "chrom": block_name.split(":")[0],
        "start": block_name.split(":")[1].split('-')[0],
        "end": block_name.split(":")[1].split('-')[1],
    }
    
    linarg.write(f"{out}/individual_linear_args/{partition_identifier}", block_info=block_info)
    
    logger.info("Computing linear ARG stats")
    get_linarg_individual_stats(out, mount_point, partition_identifiers, partition_identifier, linarg)


def get_linarg_individual_stats(out, mount_point, partition_identifiers, partition_identifier, linarg=None):
    
    if linarg is None:
        linarg = LinearARG.read(f"{out}/individual_linear_args/{partition_identifier}.h5", block=partition_identifier.split("_")[1])

    linarg.flip = np.zeros(linarg.shape[1], dtype=bool)

    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg

    genotypes_nnz = 0
    allele_counts = []
    carrier_counts = []
    for p_id in partition_identifiers:
        with h5py.File(f"{mount_point}{out}/genotype_matrices/{p_id}.h5", "r") as f:
            genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])

        genotypes_nnz += np.sum(
            [
                np.minimum(genotypes[:, i].nnz, genotypes.shape[0] - genotypes[:, i].nnz)
                for i in range(genotypes.shape[1])
            ]
        )

        v_0 = np.ones(genotypes.shape[0])
        allele_counts.append(v_0 @ genotypes)
        carrier_counts.append(get_carrier_counts(genotypes, sex=linarg.sex))

    allele_counts_from_genotype = np.concatenate(allele_counts)
    carrier_counts_from_genotype = np.concatenate(carrier_counts)

    stats = [
        linarg.shape[0],
        linarg.shape[1],
        genotypes_nnz,
        linarg.nnz,
        np.round(genotypes_nnz / linarg.nnz, 3),
        np.allclose(allele_counts_from_genotype, allele_count_from_linarg),
        np.allclose(carrier_counts_from_genotype, linarg.number_of_carriers()),
    ]
    stats = [str(x) for x in stats]
    with open(f"{out}/individual_linear_args/{partition_identifier}_stats.txt", "w") as file:
        file.write(
            " ".join(
                [
                    "n",
                    "m",
                    "genotypes_nnz",
                    "linarg_nnz",
                    "linarg_nnz_ratio",
                    "correct_allele_counts",
                    "correct_carrier_counts",
                ]
            )
            + "\n"
        )
        file.write(" ".join(stats) + "\n")


def get_carrier_counts(genotypes, sex=None):
    n_haplotypes, n_variants = genotypes.shape

    if sex is None:
        assert n_haplotypes % 2 == 0, "Expected even number of haplotypes for diploid individuals"
        sex = np.zeros(n_haplotypes // 2, dtype=np.uint8)  # Treat all as female (diploid)

    n_individuals = sex.shape[0]
    row_indices = []
    col_indices = []
    data = []

    haplo_idx = 0
    for i in range(n_individuals):
        if sex[i] == 0:
            row_indices += [i, i]
            col_indices += [haplo_idx, haplo_idx + 1]
            data += [1, 1]
            haplo_idx += 2
        else:
            row_indices.append(i)
            col_indices.append(haplo_idx)
            data.append(1)
            haplo_idx += 1

    P = sp.csc_matrix((data, (row_indices, col_indices)), shape=(n_individuals, n_haplotypes))
    individual_genotypes = P @ genotypes
    carriers = (individual_genotypes > 0).sum(axis=0).A1

    return carriers


def load_sample_ids(sample_path: Union[str, PathLike]) -> list[str]:
    """
    Helper function to load sample IDs from a path-like object. The file should be
    whitespace delimited and either contain a column with a `IID` header, or no header.
    If the file contains multiple columns `IID` header is required.
    """
    opener = gzip.open if str(sample_path).endswith(".gz") else open

    sample_ids = []
    with opener(sample_path, "rt") as f:
        first_line = f.readline().rstrip("\n")
        cols = first_line.split()
        # check if "IID" is in the first row
        if "IID" not in cols and "#IID" not in cols:
            if len(cols) == 1:
                # there is exactly 1 column, no header, first_line contains the first sample id
                iid_idx = 0
                sample_ids.append(first_line)
            elif len(cols) > 1:
                # we have multiple columns and none of them are denoted with `IID`
                raise ValueError("Multi-column sample ID file does not contain `IID` column")
            else:
                # we have no columns
                raise ValueError("Empty line in sample ID file")
        else:
            # Header mode: find the IID column index
            if "IID" in cols:
                iid_idx = cols.index("IID")
            elif "#IID" in cols:
                iid_idx = cols.index("#IID")

        # read each line and extract the IID column
        # iid_idx should be defined by here
        for line in f:
            parts = line.rstrip("\n").split()
            if len(parts) > iid_idx:
                sample_ids.append(parts[iid_idx])
            else:
                # malformed line
                raise ValueError(f"Sample ID file contains misshaped row: {line}")

    return sample_ids
