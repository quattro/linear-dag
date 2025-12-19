from pathlib import Path
import tempfile

import numpy as np
import polars as pl
from scipy import sparse

from linear_dag.association.ld import write_ld_files, get_ldm
from linear_dag.core.lineararg import LinearARG, list_blocks

TEST_DATA_DIR = Path(__file__).parent / "testdata"


def test_write_ld_files():
    """
    Test that write_ld_files creates LD matrix files in the correct format.
    """
    # Setup
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    
    # Get first block
    blocks_df = list_blocks(hdf5_path)
    block_name = blocks_df['block_name'][0]
    
    linarg = LinearARG.read(hdf5_path, block=block_name)
    
    # Select first 10 variants
    num_variants = 10
    variants = np.arange(num_variants)
    
    # Get variant info
    variant_info = LinearARG.read_variant_info(hdf5_path, block=block_name).collect()
    variant_info = variant_info.head(num_variants).rename({'ID': 'SNP'})
    
    # Write LD files to temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = f"{tmpdir}/test_ld"
        write_ld_files(linarg, variants, variant_info, out_prefix)
        
        # Check that files were created
        snplist_path = Path(f"{out_prefix}.snplist")
        npz_path = Path(f"{out_prefix}.npz")
        assert snplist_path.exists()
        assert npz_path.exists()
        
        # Read and verify SNP info file
        snp_info = pl.read_csv(snplist_path, separator='\t')
        assert snp_info.shape[0] == num_variants
        assert set(snp_info.columns) == {'POS', 'REF', 'ALT', 'SNP', 'MISSINGNESS', 'AF'}
        
        # Check that allele frequencies are in valid range [0, 1]
        af = snp_info['AF'].to_numpy()
        assert np.all((af >= 0) & (af <= 1))
        
        # Check that missingness is all zeros
        missingness = snp_info['MISSINGNESS'].to_numpy()
        assert np.all(missingness == 0)
        
        # Read and verify sparse LD matrix
        ld_matrix = sparse.load_npz(npz_path)
        assert ld_matrix.shape == (num_variants, num_variants)
        
        # LD matrix should be symmetric
        assert np.allclose(ld_matrix.toarray(), ld_matrix.T.toarray())
        
        # Diagonal should be allele frequencies (or close to them)
        # Since LD = carriers.T @ carriers / n, diagonal is sum of squared carriers / n
        carriers = linarg.get_carriers_subset(variants)
        expected_diag = np.array((carriers.T @ carriers / linarg.shape[0]).diagonal()).flatten()
        actual_diag = ld_matrix.diagonal()
        assert np.allclose(expected_diag, actual_diag)


def test_get_ldm():
    """
    Test that get_ldm computes LD matrix correctly.
    """
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    
    # Get first block
    blocks_df = list_blocks(hdf5_path)
    block_name = blocks_df['block_name'][0]
    
    linarg = LinearARG.read(hdf5_path, block=block_name)
    
    # Select first 5 variants
    variants = np.arange(5)
    
    # Compute LD matrix
    ld_matrix = get_ldm(linarg, variants)
    
    # Check shape
    assert ld_matrix.shape == (5, 5)
    
    # Check symmetry
    assert np.allclose(ld_matrix.toarray(), ld_matrix.T.toarray())
    
    # Check that values are in valid range
    ld_array = ld_matrix.toarray()
    assert np.all((ld_array >= 0) & (ld_array <= 1))
