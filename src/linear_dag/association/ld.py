import numpy as np
import polars as pl

from scipy import sparse
from scipy.sparse import csc_matrix

from linear_dag.core.lineararg import LinearARG


def get_ldm(linarg: LinearARG, variants: np.ndarray) -> csc_matrix:
    """Compute an LD matrix for a selected variant set.

    **Arguments:**

    - `linarg`: [`linear_dag.core.lineararg.LinearARG`][] object in haplotype space.
    - `variants`: Variant indices to include.

    **Returns:**

    - Sparse LD matrix as a `csc_matrix`.
    """
    carriers: csc_matrix = linarg.get_carriers_subset(variants)
    return carriers.T @ carriers / linarg.shape[0]


def write_ld_files(linarg: LinearARG, variants: np.ndarray, variant_info: pl.DataFrame, out_prefix: str) -> None:
    """Write LD matrix and SNP metadata files for downstream tooling.

    **Arguments:**

    - `linarg`: [`linear_dag.core.lineararg.LinearARG`][] object in haplotype space.
    - `variants`: Variant indices to export.
    - `variant_info`: DataFrame with `POS`, `REF`, `ALT`, and `SNP` columns.
    - `out_prefix`: Output file prefix (creates `.snplist` and `.npz` outputs).

    **Returns:**

    - `None`.
    """
    # Get carriers matrix
    carriers = linarg.get_carriers_subset(variants)

    # Calculate allele frequencies (mean of haplotypes)
    allele_freq = np.array(carriers.mean(axis=0)).flatten()

    # Calculate missingness (assuming no missing data in LinearARG, set to 0)
    missingness = np.zeros(len(variants))

    # Compute sparse LD matrix
    # Normalize by number of haplotypes
    num_haplotypes = linarg.shape[0]
    sparse_ld = sparse.csr_matrix((carriers.T @ carriers) / num_haplotypes)

    # Create SNP info DataFrame
    snp_info = variant_info.select(["POS", "REF", "ALT", "SNP"]).with_columns(
        [pl.lit(missingness).alias("MISSINGNESS"), pl.lit(allele_freq).alias("AF")]
    )

    # Write to files
    snp_info.write_csv(f"{out_prefix}.snplist", separator="\t")
    sparse.save_npz(f"{out_prefix}.npz", sparse_ld)
