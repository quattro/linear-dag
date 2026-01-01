from linear_dag.core.lineararg import LinearARG

from scipy.sparse import csc_matrix
from scipy import sparse

import numpy as np
import polars as pl

def get_ldm(linarg: LinearARG, variants: np.ndarray) -> csc_matrix:
    """
    Compute LD matrix X'X / n for a set of variants.
    
    Args:
        linarg: LinearARG object (haploid)
        variants: Array of variant indices
    
    Returns:
        Sparse LD matrix as a csc_matrix
    """
    carriers: csc_matrix = linarg.get_carriers_subset(variants)    
    return carriers.T @ carriers / linarg.shape[0]


def write_ld_files(linarg: LinearARG, variants: np.ndarray, variant_info: pl.DataFrame, 
                   out_prefix: str) -> None:
    """
    Write LD matrix and SNP info files in the same format as sparse_ld_from_bed.
    
    Args:
        linarg: LinearARG object (haploid)
        variants: Array of variant indices
        variant_info: Polars DataFrame with columns ['POS', 'REF', 'ALT', 'SNP'] for the variants
        out_prefix: Output file prefix (will create {out_prefix}.snplist and {out_prefix}.npz)
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
    snp_info = variant_info.select(['POS', 'REF', 'ALT', 'SNP']).with_columns([
        pl.lit(missingness).alias('MISSINGNESS'),
        pl.lit(allele_freq).alias('AF')
    ])
    
    # Write to files
    snp_info.write_csv(f"{out_prefix}.snplist", separator='\t')
    sparse.save_npz(f"{out_prefix}.npz", sparse_ld)

    