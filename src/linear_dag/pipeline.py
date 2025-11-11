import gzip
import os
import time

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


def make_genotype_matrix(
    vcf_path: Union[str, PathLike],
    linarg_dir: Union[str, PathLike],
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
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/variant_metadata/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/genotype_matrices/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_number}_{region}_make_genotype_matrix.log")

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
        return None
    
    if phased:
        iids = [id_ for id_ in iids for _ in range(2)]

    t2 = time.time()
    logger.info(f"vcf to sparse matrix completed in {np.round(t2 - t1, 3)} seconds")
    logger.info("Saving genotype matrix and variant metadata")
    with h5py.File(f"{linarg_dir}/genotype_matrices/{partition_number}_{region}.h5", "w") as f:
        f.create_dataset("shape", data=genotypes.shape, compression="gzip", shuffle=True)
        f.create_dataset("indptr", data=genotypes.indptr, compression="gzip", shuffle=True)
        f.create_dataset("indices", data=genotypes.indices, compression="gzip", shuffle=True)
        f.create_dataset("data", data=genotypes.data, compression="gzip", shuffle=True)
        f.create_dataset("flip", data=flip, compression="gzip", shuffle=True)
        f.create_dataset("iids", data=iids, compression="gzip", shuffle=True)
        if sex_path is not None:
            f.create_dataset("sex", data=sex, compression="gzip", shuffle=True)
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
    with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.h5", "r") as f:
        genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])
    t2 = time.time()
    logger.info(f"Genotype matrix loaded in {np.round(t2 - t1, 3)} seconds")
    logger.info("Running forward and backward brick graph algorithms")
    t3 = time.time()
    sample_indices = BrickGraph.forward_backward(
        genotypes,
        add_samples=True,
        save_to_disk=True,
        out=f"{linarg_dir}/forward_backward_graphs/{partition_identifier}",
    )
    np.savetxt(f"{linarg_dir}/forward_backward_graphs/{partition_identifier}_sample_indices.txt", sample_indices)
    t4 = time.time()
    logger.info(f"Forward and backward brick graph algorithms completed in {np.round(t4 - t3, 3)} seconds")


def reduction_union_recom(linarg_dir, load_dir, partition_identifier):
    """
    Compute the transitive reduction of the union of the forward and backward graphs
    and find recombinations to obtain the brick graph.
    """
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/brick_graph_partitions/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_identifier}_reduction_union_recom.log")

    logger.info("Loading genotypes, forward and backward graphs, and sample indices")
    t1 = time.time()
    with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.h5", "r") as f:
        genotypes = sp.csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=f["shape"][:])
    n, m = genotypes.shape
    forward_graph = read_graph_from_disk(
        f"{load_dir}{linarg_dir}/forward_backward_graphs/{partition_identifier}_forward_graph.h5"
    )
    backward_graph = read_graph_from_disk(
        f"{load_dir}{linarg_dir}/forward_backward_graphs/{partition_identifier}_backward_graph.h5"
    )
    sample_indices = np.loadtxt(
        f"{load_dir}{linarg_dir}/forward_backward_graphs/{partition_identifier}_sample_indices.txt"
    )
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


def infer_brick_graph(linarg_dir, load_dir, partition_identifier):
    """
    From a genotype matrix, infer the brick graph and find recombinations.
    """
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    os.makedirs(f"{linarg_dir}/brick_graph_partitions/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/{partition_identifier}_infer_brick_graph.log")
    logger.info("Loading genotype matrix")
    t1 = time.time()
    with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.h5", "r") as f:
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
    with h5py.File(f"{linarg_dir}/brick_graph_partitions/{partition_identifier}.h5", "w") as f:
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


def merge(linarg_dir, load_dir):
    """
    Merged partitioned brick graphs, find recombinations, and linearize.
    """

    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)

    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/merge.log")
    logger.info("Merging brick graphs")
    t1 = time.time()
    if not os.path.exists(f"{load_dir}{linarg_dir}/brick_graph_partitions"):
        raise ValueError(
            f"Path '{load_dir}{linarg_dir}/brick_graph_partitions' does not exist! Please provide valid path."
        )

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
    var_info = df.lazy()
    flip = []
    iids = None
    sex = None
    for file in files:
        with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{file[:-3]}h5", "r") as f:
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

    # pull block info before saving
    block = (
        var_info.select(
            [
                pl.col("CHROM").first().alias("chrom"),
                pl.col("POS").min().alias("start"),
                pl.col("POS").max().alias("end"),
            ]
        )
        .collect()
        .to_dicts()[0]
    )
    linarg.write(f"{linarg_dir}/linear_arg", block_info=block)
    logger.info("Computing linear ARG stats")
    get_linarg_stats(linarg_dir, load_dir, linarg)

    return


def get_linarg_stats(linarg_dir, load_dir, linarg=None):
    """
    Get stats from linear ARG.
    """
    if linarg is None:
        linarg = LinearARG.read(f"{linarg_dir}/linear_arg.h5")

    linarg.flip = np.zeros(linarg.shape[1], dtype=bool)

    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg

    files = os.listdir(f"{load_dir}{linarg_dir}/genotype_matrices/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    genotypes_nnz = 0
    allele_counts = []
    for f in files:
        with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{f}", "r") as f:
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
    with open(f"{linarg_dir}/linear_arg_stats.txt", "w") as file:
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


def add_individuals_to_linarg(linarg_dir, load_dir):
    os.makedirs(f"{linarg_dir}/logs/", exist_ok=True)
    logger = MemoryLogger(__name__, log_file=f"{linarg_dir}/logs/add_individual_nodes.log")

    logger.info("Loading linear ARG")
    t1 = time.time()
    with h5py.File(str(f"{load_dir}{linarg_dir}/linear_arg.h5"), "r") as f:
        block_name = list(f.keys())[0]
    temp = LinearARG.read(f"{load_dir}{linarg_dir}/linear_arg.h5", block=block_name)
    t2 = time.time()
    logger.info(f"Linear ARG loaded in {np.round(t2 - t1, 3)} seconds")

    logger.info("Adding individual nodes to the linear ARG")
    t3 = time.time()
    linarg = temp.add_individual_nodes()
    t4 = time.time()
    logger.info(f"Individual nodes added in {np.round(t4 - t3, 3)} seconds")

    logger.info("Saving linear ARG")
    block_info = {
        "chrom": block_name.split("_")[0],
        "start": block_name.split("_")[1],
        "end": block_name.split("_")[2],
    }
    linarg.write(f"{linarg_dir}/linear_arg_individual", block_info=block_info)

    # logger.info("Computing linear ARG stats")
    # get_linarg_individual_stats(linarg_dir, load_dir)


def get_linarg_individual_stats(linarg_dir, load_dir):
    linarg = LinearARG.read(f"{linarg_dir}/linear_arg_individual.h5", load_metadata=True)

    linarg.flip = np.zeros(linarg.shape[1], dtype=bool)

    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg

    files = os.listdir(f"{load_dir}{linarg_dir}/genotype_matrices/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    genotypes_nnz = 0
    allele_counts = []
    carrier_counts = []
    for f in files:
        with h5py.File(f"{load_dir}{linarg_dir}/genotype_matrices/{f}", "r") as f:
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
        all(allele_counts_from_genotype == allele_count_from_linarg),
        all(carrier_counts_from_genotype == linarg.number_of_carriers),
    ]
    stats = [str(x) for x in stats]
    with open(f"{linarg_dir}/linear_arg_individual_stats.txt", "w") as file:
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
