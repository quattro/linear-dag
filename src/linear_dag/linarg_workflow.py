import os

from time import time
from typing import Optional

import numpy as np

from .genotype import load_genotypes
from .lineararg import LinearARG


def run_linarg_workflow(
    input_file_prefix: str,
    output_file_prefix: Optional[str] = None,
    flip_minor_alleles: bool = False,
    maf_threshold: Optional[float] = None,
    rsq_threshold: Optional[float] = None,
    statistics_file_path: Optional[str] = None,
    recombination_method: Optional[str] = None,
    brick_graph_method: str = "old",
    make_triangular: bool = False,
    skiprows: int = 0,
) -> tuple:
    start_time = time()
    # TODO ingest a SNP info file

    genotypes, kept_variants, flipped_variants = load_genotypes(
        input_file_prefix,
        flip_minor_alleles=flip_minor_alleles,
        maf_threshold=maf_threshold,
        rsq_threshold=rsq_threshold,
        skiprows=skiprows,
    )

    # TODO handle flipped variants appropriately

    genotype_stats = (*genotypes.shape, genotypes.nnz)

    linarg = LinearARG.from_genotypes(
        genotypes, brick_graph_method=brick_graph_method, recombination_method=recombination_method
    )

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
