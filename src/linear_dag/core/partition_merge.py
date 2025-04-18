from ..memory_logger import MemoryLogger
import os
import time
import numpy as np
import polars as pl
import scipy.sparse as sp
import h5py

from ..genotype import read_vcf
from .brick_graph import BrickGraph, read_graph_from_disk, merge_brick_graphs
from .lineararg import LinearARG, VariantInfo, make_triangular, remove_degree_zero_nodes
from .one_summed_cy import linearize_brick_graph
from .recombination import Recombination


def make_genotype_matrix(vcf_path, linarg_dir, region, partition_number, phased=True, flip_minor_alleles=False, whitelist_path=None, maf_filter=None, remove_indels=False):
    """
    From a vcf file, save the genotype matrix and variant metadata for the given region.
    """
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/variant_metadata/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/genotype_matrices/", exist_ok=True)
    
    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_number}_{region}_make_genotype_matrix.log")
        
    region_formatted = f'{region.split("-")[0]}:{region.split("-")[1]}-{region.split("-")[2]}'
    
    if whitelist_path is None:
        whitelist = None
    else:
        with open(whitelist_path, 'r') as f:
            whitelist = [line.strip() for line in f]
    
    logger.info("Reading vcf as sparse matrix")
    t1 = time.time()
    genotypes, flip, v_info = read_vcf(vcf_path, phased=phased, region=region_formatted, flip_minor_alleles=flip_minor_alleles, whitelist=whitelist, maf_filter=maf_filter, remove_indels=remove_indels)
    if genotypes is None:
        logger.info(f"No variants found")
        return None
    t2 = time.time()
    logger.info(f"vcf to sparse matrix completed in {np.round(t2 - t1, 3)} seconds")
    logger.info("Saving genotype matrix and variant metadata")
    with h5py.File(f"{linarg_dir}/genotype_matrices/{partition_number}_{region}.h5", "w") as f:
        f.create_dataset('shape', data=genotypes.shape, compression='gzip', shuffle=True)
        f.create_dataset('indptr', data=genotypes.indptr, compression='gzip', shuffle=True)
        f.create_dataset('indices', data=genotypes.indices, compression='gzip', shuffle=True)
        f.create_dataset('data', data=genotypes.data, compression='gzip', shuffle=True)
        f.create_dataset('flip', data=flip, compression='gzip', shuffle=True)    
    v_info.write_csv(f"{linarg_dir}/variant_metadata/{partition_number}_{region}.txt", separator=" ")


def run_forward_backward(linarg_dir, load_dir, partition_identifier):
    """
    Run the forward and backward brick graph algorithms on the genotype matrix.
    """
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/forward_backward_graphs/", exist_ok=True)
    
    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_identifier}_forward_backward.log")
    logger.info("Loading genotype matrix")
    t1 = time.time()
    with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.h5", 'r') as f:
        genotypes = sp.csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=f['shape'][:])    
    t2 = time.time()
    logger.info(f"Genotype matrix loaded in {np.round(t2 - t1, 3)} seconds")
    logger.info("Running forward and backward brick graph algorithms")
    t3 = time.time()
    sample_indices = BrickGraph.forward_backward(genotypes, add_samples=True, save_to_disk=True, out=f"{linarg_dir}/forward_backward_graphs/{partition_identifier}")
    np.savetxt(f"{linarg_dir}/forward_backward_graphs/{partition_identifier}_sample_indices.txt", sample_indices)
    t4 = time.time()
    logger.info(f"Forward and backward brick graph algorithms completed in {np.round(t4 - t3, 3)} seconds")
    

def reduction_union_recom(linarg_dir, load_dir, partition_identifier):
    """
    Compute the transitive reduction of the union of the forward and backward graphs and find recombinations to obtain the brick graph.
    """
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/brick_graph_partitions/", exist_ok=True)
    
    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_identifier}_reduction_union_recom.log")
    
    logger.info("Loading genotypes, forward and backward graphs, and sample indices")
    t1 = time.time()
    with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.h5", 'r') as f:
        genotypes = sp.csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=f['shape'][:]) 
    n, m = genotypes.shape
    forward_graph = read_graph_from_disk(f"{load_dir}{linarg_dir}/forward_backward_graphs/{partition_identifier}_forward_graph.h5")
    backward_graph = read_graph_from_disk(f"{load_dir}{linarg_dir}/forward_backward_graphs/{partition_identifier}_backward_graph.h5")
    sample_indices = np.loadtxt(f"{load_dir}{linarg_dir}/forward_backward_graphs/{partition_identifier}_sample_indices.txt")
    t2 = time.time()
    logger.info(f"Loaded in {np.round(t2 - t1, 3)} seconds")
    
    logger.info("Combining nodes and computing reduction union")
    t3 = time.time()
    brick_graph, variant_indices = BrickGraph.combine_graphs(forward_graph, backward_graph, m)
    t4 = time.time()
    logger.info(f"Combined nodes and computed reduction union in {np.round(t4 - t3, 3)} seconds")
    
    logger.info("Finding recombinations")
    t5 = time.time()
    recom = Recombination.from_graph(brick_graph)
    recom.find_recombinations()
    t6 = time.time()
    logger.info(f"Found recombinations in {np.round(t6 - t5, 3)} seconds")
    adj_mat = recom.to_csc()
    
    logger.info("Saving brick graph")
    with h5py.File(f"{linarg_dir}/brick_graph_partitions/{partition_identifier}.h5", "w") as f:
        f.attrs['n'] = adj_mat.shape[0]
        f.create_dataset('indptr', data=adj_mat.indptr, compression='gzip', shuffle=True)
        f.create_dataset('indices', data=adj_mat.indices, compression='gzip', shuffle=True)
        f.create_dataset('data', data=adj_mat.data, compression='gzip', shuffle=True)
        f.create_dataset('variant_indices', data=np.array(variant_indices), compression='gzip', shuffle=True)
        f.create_dataset('sample_indices', data=np.array(sample_indices), compression='gzip', shuffle=True)

    af = np.diff(genotypes.indptr) / n
    geno_nnz = np.sum(n * np.minimum(af, 1 - af))
    nnz_ratio = geno_nnz / adj_mat.nnz
    logger.info(f"Stats - n: {n}, m: {m}, geno_nnz: {geno_nnz}, brickgraph_nnz: {adj_mat.nnz}, nnz_ratio: {nnz_ratio}")
        

def infer_brick_graph(linarg_dir, load_dir, partition_identifier):
    """
    From a genotype matrix, infer the brick graph and find recombinations.
    """
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/brick_graph_partitions/", exist_ok=True)
    
    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_identifier}_infer_brick_graph.log")
    logger.info("Loading genotype matrix")
    t1 = time.time()
    with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.h5", 'r') as f:
        genotypes = sp.csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=f['shape'][:])   
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
    with h5py.File(f"{linarg_dir}/brick_graph_partitions/{partition_identifier}.h5", "w") as f:
        f.attrs['n'] = adj_mat.shape[0]
        f.create_dataset('indptr', data=adj_mat.indptr, compression='gzip', shuffle=True)
        f.create_dataset('indices', data=adj_mat.indices, compression='gzip', shuffle=True)
        f.create_dataset('data', data=adj_mat.data, compression='gzip', shuffle=True)
        f.create_dataset('variant_indices', data=np.array(variant_indices), compression='gzip', shuffle=True)
        f.create_dataset('sample_indices', data=np.array(sample_indices), compression='gzip', shuffle=True)

    af = np.diff(genotypes.indptr) / n
    geno_nnz = np.sum(n * np.minimum(af, 1 - af))
    nnz_ratio = geno_nnz / adj_mat.nnz
    logger.info(f"Stats - n: {n}, m: {m}, geno_nnz: {geno_nnz}, brickgraph_nnz: {adj_mat.nnz}, nnz_ratio: {nnz_ratio}")


def merge(linarg_dir, load_dir):
    """
    Merged partitioned brick graphs, find recombinations, and linearize.
    """
    print(linarg_dir)
    
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/", exist_ok=True)
    
    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/merge.log")
    logger.info("Merging brick graphs")
    t1 = time.time()
    merged_graph, variant_indices, num_samples, index_mapping = merge_brick_graphs(
        f"{load_dir}{linarg_dir}/brick_graph_partitions"
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
    files = os.listdir(f"{load_dir}{linarg_dir}/variant_metadata/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    df_list = [pl.read_csv(f"{load_dir}{linarg_dir}/variant_metadata/{f}", separator=" ") for f in files]
    df = pl.concat(df_list)
    var_info = VariantInfo(df)
    files = os.listdir(f"{load_dir}{linarg_dir}/genotype_matrices/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    flip = []
    for file in files:
        with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{file}", 'r') as f:
            flip_partition = list(f['flip'][:])
        flip += flip_partition    
    flip = np.array(flip)
    
    logger.info("Triangularizing and computing nonunique indices")
    A_filt, variant_indices_reindexed, sample_indices_reindexed  = remove_degree_zero_nodes(A, variant_indices, sample_indices)
    A_tri, variant_indices_tri = make_triangular(A_filt, variant_indices_reindexed, sample_indices_reindexed)
    linarg = LinearARG(A_tri, variant_indices_tri, flip, len(sample_indices), var_info)
    linarg.calculate_nonunique_indices()
    logger.info("Saving linear ARG")
    linarg.write(f"{linarg_dir}/linear_arg")
    logger.info("Computing linear ARG stats")
    get_linarg_stats(linarg_dir, load_dir)


def get_linarg_stats(linarg_dir, load_dir):
    """
    Get stats from linear ARG.
    """
    linarg = LinearARG.read(f"{linarg_dir}/linear_arg.h5")
    
    linarg.flip = np.array([False for i in range(linarg.shape[1])])

    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg

    files = os.listdir(f"{load_dir}{linarg_dir}/genotype_matrices/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    genotypes_nnz = 0
    allele_counts = []
    for f in files:
        
        with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{f}", 'r') as f:
            genotypes = sp.csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=f['shape'][:]) 
        
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

    brick_graph_nnz = np.sum(linarg.A.data > 0)

    stats = [
        linarg.shape[0],
        linarg.shape[1],
        genotypes_nnz,
        linarg.nnz,
        brick_graph_nnz,
        np.round(genotypes_nnz / linarg.nnz, 3),
        np.round(genotypes_nnz / brick_graph_nnz, 3),
        all(allele_counts_from_genotype == allele_count_from_linarg),
    ]
    stats = [str(x) for x in stats]
    with open(f"{linarg_dir}/linear_arg_stats.txt", "w") as file:
        file.write(
            " ".join(
                [
                    "n",
                    "m",
                    "genotypes_nnz",
                    "linarg_nnz",
                    "brick_graph_nnz",
                    "linarg_nnz_ratio",
                    "brick_graph_nnz_ratio",
                    "correct_allele_counts",
                ]
            )
            + "\n"
        )
        file.write(" ".join(stats) + "\n")
