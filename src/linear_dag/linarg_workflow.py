import os

from time import time
from typing import Optional

import numpy as np

from scipy.io import mmread
from scipy.sparse import csc_matrix

from .genotype_processing import apply_maf_threshold, binarize, flip_alleles
from .lineararg import LinearARG


def run_linarg_workflow(
    input_file_prefix: str,
    output_file_prefix: Optional[str] = None,
    flip_minor_alleles: bool = False,
    maf_threshold: Optional[float] = None,
    rsq_threshold: Optional[float] = None,
    statistics_file_path: Optional[str] = None,
    recombination_method: str = "old",
    brick_graph_method: str = "old",
    make_triangular: bool = False,
) -> tuple:
    start_time = time()
    # TODO ingest a SNP info file

    # Check and select input files
    mtx_file = f"{input_file_prefix}.mtx"
    txt_file = f"{input_file_prefix}.txt"
    if os.path.exists(mtx_file):
        genotype_file = mtx_file
        input_type = "mtx"
    elif os.path.exists(txt_file):
        genotype_file = txt_file
        input_type = "txt"
    else:
        raise FileNotFoundError(f"No genotype matrix file found with prefix: {input_file_prefix}")

    # Initialize Linarg based on input file type
    if input_type == "mtx":
        genotypes = csc_matrix(mmread(genotype_file))
    else:
        genotypes = np.loadtxt(genotype_file)

    if rsq_threshold is None:
        well_imputed_variants = np.arange(genotypes.shape[1])
    else:
        genotypes, well_imputed_variants = binarize(genotypes, rsq_threshold)

    ploidy = np.max(genotypes).astype(int)
    if maf_threshold is None:
        common_variants = np.arange(genotypes.shape[1])
    else:
        genotypes, common_variants = apply_maf_threshold(genotypes, ploidy, maf_threshold)

    # TODO output which variants were kept
    kept_variants = well_imputed_variants[common_variants]
    print("kept_variants:", kept_variants.shape)

    if flip_minor_alleles:
        genotypes, flipped = flip_alleles(genotypes, ploidy)

    genotype_stats = (*genotypes.shape, genotypes.nnz)

    linarg = LinearARG.from_genotypes(genotypes, brick_graph_method=brick_graph_method)

    if recombination_method == "old":
        linarg = linarg.unweight()
        linarg = linarg.find_recombinations()
    elif recombination_method == "new":
        raise NotImplementedError

    if make_triangular:
        linarg = linarg.make_triangular()
        kept_variants[np.argsort(linarg.variant_indices)] = kept_variants

    if output_file_prefix is not None:
        output_file_path = os.path.join(output_file_prefix)
        linarg.write(output_file_path, variant_metadata={"original_index": 1 + kept_variants})

    runtime = time() - start_time

    # Handle statistics file
    line = (input_file_prefix, *genotype_stats, linarg.nnz, genotype_stats[2] / linarg.nnz, runtime)
    print(
        f"file_name: {line[0]},num_samples: {line[1]}, num_variants: {line[2]}, nnz_X: {line[3]}, "
        f"nnz_A: : {line[4]}, nnz_ratio: {line[5]}, runtime: {line[6]}"
    )
    if statistics_file_path:
        if not os.path.exists(statistics_file_path):
            with open(statistics_file_path, "w") as stats_file:
                stats_file.write("file_name,num_samples,num_variants,nnz_X,nnz_A,nnz_ratio,runtime\n")

        with open(statistics_file_path, "a") as stats_file:
            # Assuming stats is a tuple, write it to the end of the file
            stats_file.write(",".join(map(str, line)) + "\n")

    return linarg, genotypes
