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


def test_parallel_operator():
    """
    Test that ParallelOperator gives the same result as serial processing.
    """
    # 1. Setup
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    num_traits = 5

    # 2. Parallel version
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as operator:
        n, m = operator.shape

        # Test transpose multiplication
        y = np.random.rand(n, num_traits)
        parallel_result_T = operator.T @ y

        # Test forward multiplication
        b = np.random.rand(m, num_traits)
        parallel_result = operator @ b

    # 3. Serial version
    blocks = list_blocks(hdf5_path)['block_name']

    # Transpose multiplication
    serial_results_T = []
    for block_name in blocks:
        linarg = LinearARG.read(hdf5_path, block=block_name)
        serial_results_T.append(linarg.T @ y)
    serial_result_T = np.vstack(serial_results_T)

    # Forward multiplication
    serial_result = np.zeros((n, num_traits))
    variant_offset = 0
    for block_name in blocks:
        linarg = LinearARG.read(hdf5_path, block=block_name)
        num_block_variants = linarg.shape[1]
        block_b = b[variant_offset : variant_offset + num_block_variants, :]
        serial_result += linarg @ block_b
        variant_offset += num_block_variants

    # 4. Assertions
    np.testing.assert_allclose(parallel_result_T, serial_result_T, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(parallel_result, serial_result, rtol=1e-3, atol=1e-2)
