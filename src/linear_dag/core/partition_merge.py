import os
import time

import numpy as np
import polars as pl
import scipy.sparse

from cyvcf2 import VCF

from .brick_graph import BrickGraph, merge_brick_graphs
from .lineararg import LinearARG, VariantInfo
from .one_summed_cy import linearize_brick_graph
from .recombination import Recombination


def make_genotype_matrix(vcf_path, linarg_dir, region, partition_number, phased=True, flip_minor_alleles=False):
    """
    From vcf file, saves the genotype matrix as csc sparse matrix, variant metadata, and filtered out variants.
    Codes unphased genotypes as 0/1/2/3, where 3 means that at least one of the two alleles is unknown.
    Codes phased genotypes as 0/1, and there are 2n rows, where rows 2*k and 2*k+1 correspond to individual k.
    """
    if not os.path.exists(f"{linarg_dir}/variant_metadata/"):
        os.makedirs(f"{linarg_dir}/variant_metadata/")
    # if not os.path.exists(f'{linarg_dir}/filtered_variants/'): os.makedirs(f'{linarg_dir}/filtered_variants/')
    if not os.path.exists(f"{linarg_dir}/genotype_matrices/"):
        os.makedirs(f"{linarg_dir}/genotype_matrices/")

    chrom = region.split("chr")[1].split("-")[0]
    start = int(region.split("-")[1])
    end = int(region.split("-")[2])
    region_formatted = f'{region.split("-")[0]}:{region.split("-")[1]}-{region.split("-")[2]}'
    vcf = VCF(vcf_path, gts012=True, strict_gt=True)

    data = []
    idxs = []
    ptrs = [0]
    flip = False
    ploidy = 1 if phased else 2

    f_var = open(f"{linarg_dir}/variant_metadata/{partition_number}_{region}.txt", "w")
    f_var.write(" ".join(["CHROM", "POS", "ID", "REF", "ALT", "FLIP", "IDX"]) + "\n")

    # f_filt = open(f'{linarg_dir}/filtered_variants/{partition_number}_{region}.txt', 'w')
    # f_filt.write(' '.join(['CHROM', 'POS', 'ID', 'REF', 'ALT', 'AF'])+'\n')

    var_index = 0
    for var in vcf(region_formatted):
        if (var.POS < start) or (var.POS > end):
            continue  # ignore indels that are outside of region
        if phased:
            gts = np.ravel(np.asarray(var.genotype.array())[:, :2])
        else:
            gts = var.gt_types
        if flip_minor_alleles:
            af = np.mean(gts) / ploidy
            if af > 0.5:
                gts = ploidy - gts
                flip = True
            else:
                flip = False

        af = np.mean(gts) / ploidy
        # if (af == 0) or (af == 1): # filter out variants with af=0 or af=1
        #     f_filt.write(' '.join([chrom, str(var.POS), '.', var.REF, ','.join(var.ALT), str(af)])+'\n')
        #     continue

        (idx,) = np.where(gts != 0)
        data.append(gts[idx])
        idxs.append(idx)
        ptrs.append(ptrs[-1] + len(idx))
        f_var.write(
            " ".join([chrom, str(var.POS), ".", var.REF, ",".join(var.ALT), str(int(flip)), str(var_index)]) + "\n"
        )
        var_index += 1

    f_var.close()
    # f_filt.close()

    data = np.concatenate(data)
    idxs = np.concatenate(idxs)
    ptrs = np.array(ptrs)
    genotypes = scipy.sparse.csc_matrix((data, idxs, ptrs), shape=(gts.shape[0], len(ptrs) - 1))

    scipy.sparse.save_npz(f"{linarg_dir}/genotype_matrices/{partition_number}_{region}.npz", genotypes)


def infer_brick_graph(linarg_dir, load_dir, partition_identifier):
    """
    From a genotype matrix, infer the brick graph and find recombinations.
    """
    if not os.path.exists(f"{linarg_dir}/brick_graph_partitions/"):
        os.makedirs(f"{linarg_dir}/brick_graph_partitions/")
    if not os.path.exists(f"{linarg_dir}/brick_graph_partition_stats/"):
        os.makedirs(f"{linarg_dir}/brick_graph_partition_stats/")

    t1 = time.time()
    genotypes = scipy.sparse.load_npz(f"{load_dir}{linarg_dir}/genotype_matrices/{partition_identifier}.npz")
    n, m = genotypes.shape
    t2 = time.time()
    brick_graph, samples_idx, variants_idx = BrickGraph.from_genotypes(genotypes)
    t3 = time.time()
    recom = Recombination.from_graph(brick_graph)
    recom.find_recombinations()
    t4 = time.time()
    adj_mat = recom.to_csr()

    np.savez(
        f"{linarg_dir}/brick_graph_partitions/{partition_identifier}.npz",
        brick_graph_data=adj_mat.data,
        brick_graph_indices=adj_mat.indices,
        brick_graph_indptr=adj_mat.indptr,
        brick_graph_shape=adj_mat.shape,
        sample_indices=np.array(samples_idx),
        variant_indices=np.array(variants_idx),
    )

    af = np.diff(genotypes.indptr) / n
    geno_nnz = np.sum(n * np.minimum(af, 1 - af))
    nnz_ratio = geno_nnz / adj_mat.nnz
    stats = [
        partition_identifier,
        str(n),
        str(m),
        str(np.round(t2 - t1, 3)),
        str(np.round(t3 - t2, 3)),
        str(np.round(t4 - t3, 3)),
        str(geno_nnz),
        str(adj_mat.nnz),
        str(np.round(nnz_ratio, 3)),
    ]

    with open(f"{linarg_dir}/brick_graph_partition_stats/{partition_identifier}.txt", "w") as file:
        file.write(
            " ".join(
                [
                    "partition_identifier",
                    "n",
                    "m",
                    "mtx_load_time",
                    "brickgraph_inference_time",
                    "find_recombinations_time",
                    "geno_nnz",
                    "brickgraph_nnz",
                    "nnz_ratio",
                ]
            )
            + "\n"
        )
        file.write(" ".join(stats) + "\n")


def merge(linarg_dir, load_dir):
    """
    Merged partitioned brick graphs, find recombinations, and linearize.
    """
    if not os.path.exists(f"{linarg_dir}/"):
        os.makedirs(f"{linarg_dir}/")
    merged_graph, variant_indices, num_samples, index_mapping = merge_brick_graphs(
        f"{load_dir}{linarg_dir}/brick_graph_partitions"
    )
    merged_graph_recom = Recombination.from_graph(merged_graph)
    merged_graph_recom.find_recombinations()
    linear_arg_adjacency_matrix = linearize_brick_graph(merged_graph_recom)
    sample_indices = np.arange(num_samples)

    # get VariantInfo
    files = os.listdir(f"{load_dir}{linarg_dir}/variant_metadata/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    df_list = [pl.read_csv(f"{load_dir}{linarg_dir}/variant_metadata/{f}", separator=" ") for f in files]
    df = pl.concat(df_list)
    variant_indices = pl.Series(variant_indices)
    df = df.with_columns(variant_indices.alias("IDX"))  # replace old indices with new merged ones
    var_info = VariantInfo(df)

    linarg = LinearARG(linear_arg_adjacency_matrix, sample_indices, var_info)
    linarg.write(f"{linarg_dir}/linear_arg")
    get_linarg_stats(linarg_dir, load_dir)


def get_linarg_stats(linarg_dir, load_dir):
    """
    Get stats from linear ARG.
    """
    linarg = LinearARG.read(
        f"{linarg_dir}/linear_arg.npz", f"{linarg_dir}/linear_arg.pvar", f"{linarg_dir}/linear_arg.psam"
    )

    df_flip = linarg.variants.table.clone()
    df_flip = df_flip.with_columns(pl.Series([0] * df_flip.shape[0]).alias("FLIP"))
    linarg.variants.table = df_flip

    linarg_triangular = linarg.make_triangular()
    v = np.ones(linarg.shape[0])
    allele_count_from_linarg = v @ linarg_triangular

    files = os.listdir(f"{load_dir}{linarg_dir}/genotype_matrices/")
    ind_arr = np.array([int(f.split("_")[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist()  # sort files by index
    genotypes_nnz = 0
    allele_counts = []
    for f in files:
        genotypes = scipy.sparse.load_npz(f"{load_dir}{linarg_dir}/genotype_matrices/{f}")
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

    coo = linarg.A.tocoo()
    brick_graph_nnz = np.sum(coo.data > 0)

    stats = [
        linarg.shape[0],
        linarg.shape[1],
        genotypes_nnz,
        linarg.nnz,
        brick_graph_nnz,
        np.round(genotypes_nnz / linarg.nnz, 3),
        np.round(genotypes_nnz / brick_graph_nnz, 3),
        all(allele_counts_from_genotype == allele_count_from_linarg[0]),
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
