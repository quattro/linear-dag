import os

from time import time
from typing import Optional

import numpy as np

from scipy.io import mmread
from scipy.sparse import csc_matrix

from .lineararg import LinearARG
from .utils import apply_maf_threshold, binarize, flip_alleles


def run_linarg_workflow(
    input_file_prefix: str,
    output_file_prefix: str,
    flip_minor_alleles: bool = False,
    maf_threshold: Optional[float] = None,
    rsq_threshold: Optional[float] = None,
    max_sample_size: Optional[int] = None,
    statistics_file_path: Optional[str] = None,
) -> None:
    start_time = time()

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

    # Check SNP info file
    # snpinfo_file = f"{input_file_prefix}.snpinfo.csv"
    # if not os.path.exists(snpinfo_file):
    #     raise FileNotFoundError(f"SNP info file not found: {snpinfo_file}")

    # Check output directory
    # if not os.path.isdir(output_directory):
    #     raise NotADirectoryError(f"Output directory does not exist: {output_directory}")

    # Initialize Linarg based on input file type
    if input_type == "mtx":
        genotypes = csc_matrix(mmread(genotype_file))
    else:
        genotypes = np.loadtxt(genotype_file)

    # Read SNP info file and check the number of lines
    # with open(snpinfo_file, 'r') as file:
    #     snpinfo_lines = file.readlines()
    #     if len(snpinfo_lines) != linarg.variants.shape[0]:
    #         raise ValueError("The number of lines in the SNP info file does not match the number of variants.")

    if rsq_threshold is not None:
        genotypes = binarize(genotypes, rsq_threshold)

    ploidy = np.max(genotypes).astype(int)
    if maf_threshold is not None:
        genotypes = apply_maf_threshold(genotypes, ploidy, maf_threshold)

    if flip_minor_alleles:
        genotypes, flipped = flip_alleles(genotypes, ploidy)

    linarg = LinearARG.from_genotypes(genotypes)
    linarg = linarg.find_recombinations()

    # Calculate and handle statistics
    stats = (*genotypes.shape, genotypes.nnz, linarg.nnz)

    # Write Linarg output with SNP info file
    output_file_path = os.path.join(output_file_prefix)  # Modify as needed
    linarg.write(output_file_path)

    runtime = time() - start_time

    # Handle statistics file
    if statistics_file_path:
        if not os.path.exists(statistics_file_path):
            with open(statistics_file_path, "w") as stats_file:
                stats_file.write("file_name,num_samples,num_variants,nnz_X,nnz_A,nnz_ratio,runtime\n")

        with open(statistics_file_path, "a") as stats_file:
            # Assuming stats is a tuple, write it to the end of the file
            line = (input_file_prefix, *stats, stats[2] / stats[3], runtime)
            stats_file.write(",".join(map(str, line)) + "\n")
