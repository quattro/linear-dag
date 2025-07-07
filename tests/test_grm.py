import pytest
import numpy as np
from pathlib import Path
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag.core.grm_parallel import GRMOperator

TEST_DATA_DIR = Path(__file__).parent / "testdata"

def test_grm_operator():
    """
    Test that the GRMOperator gives the same result as direct computation.
    """
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"

    # 1. Compute GRM directly
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as genotypes:
        G = genotypes @ np.eye(genotypes.shape[1])
        # Standardize genotypes
        G_std = (G - G.mean(axis=0)) / G.std(axis=0)
        G_std[np.isnan(G_std)] = 0
        # Compute GRM
        grm_direct = G_std @ G_std.T

    # 2. Use GRMOperator
    with GRMOperator.from_hdf5(hdf5_path, num_processes=2) as grm_op:
        # 3. Compare results
        np.random.seed(42)
        vec = np.random.randn(grm_op.shape[1])
        res_op = grm_op @ vec
        res_direct = grm_direct @ vec

    np.testing.assert_allclose(res_op, res_direct, rtol=1e-3)
