import pytest
from linear_dag.core.lineararg import LinearARG, list_blocks
from linear_dag.genotype import read_vcf
import polars as pl
from pathlib import Path
from scipy.sparse import csc_matrix
import numpy as np
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag.core.operators import get_diploid_operator

TEST_DATA_DIR = Path(__file__).parent / "testdata"

def test_lineararg():
    """Tests for lineararg module."""
    # Test list_blocks()
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    print(f"\nTesting list_blocks with HDF5 file: {hdf5_path}")
    blocks_df = list_blocks(hdf5_path)
    print("Blocks found:")
    print(blocks_df)
    assert isinstance(blocks_df, pl.DataFrame)
    assert not blocks_df.is_empty()
    assert 'block_name' in blocks_df.columns

def test_read_vcf():
    """Test reading a VCF file."""
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf"
    genotypes, flip, v_info, iids = read_vcf(vcf_path)

    assert isinstance(genotypes, csc_matrix)
    assert isinstance(flip, np.ndarray)
    assert isinstance(v_info, pl.DataFrame)
    assert isinstance(iids, list)

    assert genotypes.shape[0] == len(iids) * 2 # phased
    assert genotypes.shape[1] == len(v_info)
    assert len(flip) == len(v_info)


def test_from_vcf():
    """Test creating a LinearARG from a VCF file."""
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf"
    linarg, genotypes = LinearARG.from_vcf(vcf_path, return_genotypes=True)

    assert isinstance(linarg, LinearARG)
    assert isinstance(genotypes, csc_matrix)
    assert isinstance(linarg.iids, pl.Series)
    assert not linarg.iids.is_empty()
    assert linarg.shape[0] == len(linarg.iids)
    assert linarg.shape[1] > 0
    assert linarg.variants is not None
    assert linarg.variants.collect().height > 0


def test_zero_ac_variants():
    """Test handling of variants with 0 allele count (all-zero columns)."""
    genotypes = csc_matrix(np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ]))
    flip = np.array([False, False, False, False])
    
    linarg = LinearARG.from_genotypes(genotypes, flip, find_recombinations=True)
    
    assert isinstance(linarg, LinearARG)
    assert linarg.shape[0] == genotypes.shape[0]


def test_samples_with_no_variants():
    """Test handling of samples with no variants (all-zero rows)."""
    genotypes = csc_matrix(np.array([
        [1, 1, 1],
        [0, 0, 0],
        [1, 0, 1],
        [0, 0, 0]
    ]))
    flip = np.array([False, False, False])
    
    linarg = LinearARG.from_genotypes(genotypes, flip, find_recombinations=True)
    
    assert isinstance(linarg, LinearARG)
    assert linarg.shape[0] == genotypes.shape[0]


def test_read_write_matmul(tmp_path):
    """
    Test that a written and then read LinearARG object gives the same
    matrix-vector product as the original object and the raw genotype matrix.
    Also tests all matrix multiplication variants.
    """
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf"
    linarg, genotypes = LinearARG.from_vcf(vcf_path, return_genotypes=True)

    # 1. Save the linear arg to a temporary file
    temp_h5_path = str(tmp_path / "test_linarg")
    linarg = linarg.add_individual_nodes()
    block_info = {"chrom": 21, "start": 0, "end": 1000000}
    linarg.write(temp_h5_path, block_info=block_info)

    # 2. Read it from the temp file
    loaded_linarg = LinearARG.read(temp_h5_path, block=f'{block_info["chrom"]}_{block_info["start"]}_{block_info["end"]}')
    assert loaded_linarg.shape == linarg.shape
    print(loaded_linarg.iids)
    print(linarg.iids)
    assert loaded_linarg.iids.len() == linarg.iids.len()

    # 3. Test multiplications
    np.random.seed(42)

    # Test vector multiplication (matvec)
    vec = np.random.rand(linarg.shape[1])
    res_original = linarg @ vec
    res_loaded = loaded_linarg @ vec
    res_genotypes = genotypes @ vec
    np.testing.assert_allclose(res_original, res_genotypes, rtol=1e-6)
    np.testing.assert_allclose(res_loaded, res_genotypes, rtol=1e-6)

    # Test transpose vector multiplication (rmatvec)
    vec_left = np.random.rand(linarg.shape[0])
    res_original_T = linarg.T @ vec_left
    res_loaded_T = loaded_linarg.T @ vec_left
    res_genotypes_T = genotypes.T @ vec_left
    np.testing.assert_allclose(res_original_T, res_genotypes_T, rtol=1e-6)
    np.testing.assert_allclose(res_loaded_T, res_genotypes_T, rtol=1e-6)

    # Test matrix multiplication (matmat)
    k = 5
    mat = np.random.rand(linarg.shape[1], k)
    res_mat_original = linarg @ mat
    res_mat_loaded = loaded_linarg @ mat
    res_mat_genotypes = genotypes @ mat
    np.testing.assert_allclose(res_mat_original, res_mat_genotypes, rtol=1e-6)
    np.testing.assert_allclose(res_mat_loaded, res_mat_genotypes, rtol=1e-6)

    # Test transpose matrix multiplication
    mat_left = np.random.rand(linarg.shape[0], k)
    res_mat_original_T = linarg.T @ mat_left
    res_mat_loaded_T = loaded_linarg.T @ mat_left
    res_mat_genotypes_T = genotypes.T @ mat_left
    np.testing.assert_allclose(res_mat_original_T, res_mat_genotypes_T, rtol=1e-6)
    np.testing.assert_allclose(res_mat_loaded_T, res_mat_genotypes_T, rtol=1e-6)

    # Number of carriers
    diploid_genotypes = get_diploid_operator(genotypes) @ np.eye(genotypes.shape[1])
    num_carriers = np.sum(diploid_genotypes > 0, axis=0)
    assert np.all(num_carriers == loaded_linarg.number_of_carriers())


def test_get_carriers_subset():
    """Test get_carriers_subset method against linarg @ indicator."""
    vcf_path = TEST_DATA_DIR / "1kg_small.vcf"
    linarg, genotypes = LinearARG.from_vcf(vcf_path, return_genotypes=True)
    
    # Test with a subset of variant indices
    np.random.seed(42)
    n_variants = linarg.shape[1]
    subset_size = min(10, n_variants)
    user_indices = np.sort(np.random.choice(n_variants, subset_size, replace=False))
    
    # Get carriers using the new method
    carriers_matrix = linarg.get_carriers_subset(user_indices)
    
    # Verify shape
    assert carriers_matrix.shape == (linarg.shape[0], len(user_indices))
    
    # Verify against linarg @ indicator for each variant
    for i, variant_idx in enumerate(user_indices):
        indicator = np.zeros(n_variants)
        indicator[variant_idx] = 1
        expected = linarg @ indicator
        actual = carriers_matrix[:, i].toarray().ravel()
        np.testing.assert_array_equal(actual, expected, 
            err_msg=f"Mismatch for variant index {variant_idx}")
    
    # Test unphased option
    carriers_unphased = linarg.get_carriers_subset(user_indices, unphased=True)
    
    # Verify shape: should have half the rows (diploid)
    assert carriers_unphased.shape == (linarg.shape[0] // 2, len(user_indices))
    
    # Verify that unphased carriers are the sum of consecutive haplotype pairs
    carriers_phased = carriers_matrix.toarray()
    for i in range(carriers_unphased.shape[0]):
        expected_diploid = carriers_phased[2*i, :] + carriers_phased[2*i + 1, :]
        actual_diploid = carriers_unphased[i, :].toarray().ravel()
        np.testing.assert_array_equal(actual_diploid, expected_diploid,
            err_msg=f"Mismatch for individual {i}")
