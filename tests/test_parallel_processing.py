# import pytest
import numpy as np
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse import diags
from pathlib import Path
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag.core.grm_parallel import GRMOperator

from linear_dag import LinearARG, list_blocks
from linear_dag.core.metadata import read_metadata

def test_parallel_matvec():
    """Test parallel matrix-vector multiplication."""
    hdf5_path = "data/test/test_chr21_100_haplotypes.h5"
    with ParallelOperator.from_hdf5(hdf5_path, max_num_traits=2, num_processes=8) as linarg_op:
        # Create random input vector of appropriate size (num_samples)
        x = np.random.randn(linarg_op.shape[1], 5).astype(np.float32)
        result_parallel = linarg_op @ x

    # Compute serial version for comparison
    metadata = list_blocks(hdf5_path)
    result_serial = 0
    
    variant_counter = 0
    for block in metadata.iter_rows(named=True):
        linarg = LinearARG.read(hdf5_path, block['block_name'])
        result = linarg @ x[variant_counter:variant_counter+linarg.shape[1], :]
        variant_counter += linarg.shape[1]
        result_serial += result

    # Check that parallel and serial results match
    assert np.allclose(result_parallel, result_serial, rtol=1e-2, atol=1e-2), "Left multiplication results do not match"
    print("Left multiplication results match")


def test_parallel_rmatvec():
    """Test parallel matrix-vector multiplication."""
    hdf5_path = "data/test/test_chr21_100_haplotypes.h5"
    with ParallelOperator.from_hdf5(hdf5_path, max_num_traits=2, num_processes=8) as linarg_op:
        # Create random input vector of appropriate size (num_samples)
        x = np.random.randn(linarg_op.shape[0], 5).astype(np.float32)
        result_parallel = x.T @ linarg_op
        print(result_parallel.shape)

    # Compute serial version for comparison
    metadata = list_blocks(hdf5_path)
    result_serial = []

    for block in metadata.iter_rows(named=True):
        linarg = LinearARG.read(hdf5_path, block['block_name'])
        result = x.T @ linarg
        result_serial.append(result)

    result_serial = np.hstack(result_serial)

    # Check that parallel and serial results match
    assert np.allclose(result_parallel, result_serial, rtol=1e-2, atol=1e-2), "Right multiplication results do not match"
    print("Right multiplication results match")


def test_grm():
    """Test GRM computation."""

    hdf5_path = "data/test/test_chr21_100_haplotypes.h5"
    with GRMOperator.from_hdf5(hdf5_path, max_num_traits=10, num_processes=1) as grm_op:
        # Create random input vector of appropriate size (num_samples)
        x = np.random.randn(grm_op.shape[0], 22).astype(np.float32)
        # x = np.ones((grm_op.shape[0], 1))

        # Compute left and right multiplication
        result_parallel = grm_op @ x

        assert result_parallel.shape == x.shape

    # Compute serial version for comparison
    result_serial = np.zeros_like(x)
    metadata = list_blocks(hdf5_path)

    # For each block we need to compute its contribution to the full GRM
    for block in metadata.iter_rows(named=True):
        linarg = LinearARG.read(hdf5_path, block['block_name'])
        result = linarg.normalized @ linarg.normalized.T @ x
        result_serial += result

    # Check that parallel and serial results match
    # print(f"first 10 elements of result and result_serial: {result[:10,0]} {result_serial[:10,0]}")
    assert np.allclose(result_parallel, result_serial, rtol=1e-2, atol=1e-2), "GRM multiplication results do not match"
    print("GRM multiplication results match")

def test_grm_per_allele():
    """Test GRM computation."""

    hdf5_path = "data/test/test_chr21_100_haplotypes.h5"
    alpha = 0
    with GRMOperator.from_hdf5(hdf5_path, max_num_traits=10, num_processes=1, alpha=alpha) as grm_op:
        # Create random input vector of appropriate size (num_samples)
        x = np.random.randn(grm_op.shape[0], 22).astype(np.float32)
        # x = np.ones((grm_op.shape[0], 1))

        # Compute left and right multiplication
        result_parallel = grm_op @ x

        assert result_parallel.shape == x.shape

    # Compute serial version for comparison
    result_serial = np.zeros_like(x)
    metadata = list_blocks(hdf5_path)

    # For each block we need to compute its contribution to the full GRM
    for block in metadata.iter_rows(named=True):
        linarg = LinearARG.read(hdf5_path, block['block_name'])
        result = linarg.mean_centered @ linarg.mean_centered.T @ x
        result_serial += result

    # Check that parallel and serial results match
    print(f"first 10 elements of result_parallel and result_serial: {result_parallel[:10,0]} {result_serial[:10,0]}")
    assert np.allclose(result_parallel, result_serial, rtol=1e-2, atol=1e-2), "GRM multiplication results do not match"
    print("GRM multiplication results match")

if __name__ == "__main__":
    test_parallel_matvec()
    test_parallel_rmatvec()
    test_grm()
    test_grm_per_allele()
