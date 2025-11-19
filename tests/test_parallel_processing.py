from pathlib import Path

import numpy as np
import pytest

from linear_dag.core.lineararg import list_blocks, LinearARG
from linear_dag.core.parallel_processing import ParallelOperator, GRMOperator


TEST_DATA_DIR = Path(__file__).parent / "testdata"


def _load_serial_blocks(hdf5_path: Path):
    blocks_df = list_blocks(hdf5_path)
    block_names = blocks_df.get_column("block_name").to_list()
    n_variants = blocks_df.get_column("n_variants").to_list()
    linargs = [LinearARG.read(hdf5_path, block) for block in block_names]
    return linargs, n_variants


def test_matmat_matches_serial():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, nvars = _load_serial_blocks(hdf5_path)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as par:
        n, m = par.shape
        rng = np.random.default_rng(42)
        X = rng.standard_normal((m, 3)).astype(np.float32)

        # Parallel result
        Y_par = par @ X

    # Serial result: sum over blocks
    Y_ser = np.zeros((n, 3), dtype=np.float32)
    offset = 0
    for la, nv in zip(linargs, nvars):
        Y_ser += la @ X[offset : offset + nv, :]
        offset += nv

    assert np.allclose(Y_par, Y_ser, rtol=1e-5, atol=1e-5)


def test_rmatmat_matches_serial():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, nvars = _load_serial_blocks(hdf5_path)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as par:
        n, m = par.shape
        rng = np.random.default_rng(123)
        Y = rng.standard_normal((n, 2)).astype(np.float32)

        # Parallel result
        Z_par = par.T @ Y

    # Serial result: vertical concatenation over blocks
    Z_parts = []
    for la in linargs:
        Z_parts.append(la.T @ Y)
    Z_ser = np.vstack(Z_parts)

    assert np.allclose(Z_par, Z_ser, rtol=1e-5, atol=1e-5)


def test_number_of_carriers_matches_serial():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, nvars = _load_serial_blocks(hdf5_path)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as par:
        # Parallel: default includes all individuals
        carriers_par = par.number_of_heterozygotes()

    # Serial: concatenate per-block results
    carriers_parts = []
    for la in linargs:
        carriers_parts.append(la.number_of_heterozygotes().reshape(-1, 1))
    carriers_ser = np.vstack(carriers_parts)

    # Cast and compare
    carriers_par_int = np.rint(carriers_par).astype(np.int64)
    assert carriers_par_int.shape == carriers_ser.shape
    assert np.array_equal(carriers_par_int, carriers_ser.astype(np.int64))


def test_rmatmat_in_place_view_and_correctness():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, _ = _load_serial_blocks(hdf5_path)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, max_num_traits=10) as par:
        n, _ = par.shape
        rng = np.random.default_rng(7)
        Y = rng.standard_normal((n, 3)).astype(np.float32)

        # In-place result is a view into shared memory
        Z_view = par._rmatmat(Y, in_place=True)

        # Serial result for comparison
        Z_parts = [la.T @ Y for la in linargs]
        Z_ser = np.vstack(Z_parts)
        assert np.allclose(Z_view, Z_ser, rtol=1e-5, atol=1e-5)

        # Aliasing check: running again with new Y should mutate Z_view contents
        Y2 = rng.standard_normal((n, 3)).astype(np.float32)
        _ = par._rmatmat(Y2, in_place=True)
        Z2_ser = np.vstack([la.T @ Y2 for la in linargs])
        assert np.allclose(Z_view, Z2_ser, rtol=1e-5, atol=1e-5)



def test_matmat_in_place_uses_shared_variant_data():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, nvars = _load_serial_blocks(hdf5_path)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, max_num_traits=10) as par:
        _, m = par.shape
        rng = np.random.default_rng(11)
        n = 3
        X_data = rng.standard_normal((m, n)).astype(np.float32)

        # Pre-populate shared variant_data directly without copying
        view = par.borrow_variant_data_view()
        view[:, :n] = X_data

        # Dummy X only supplies n via shape; contents are ignored in in_place mode
        X_dummy = np.empty_like(X_data)
        Y_par = par._matmat(X_dummy, in_place=True)

    # Serial result: sum over blocks
    Y_ser = np.zeros((par.shape[0], n), dtype=np.float32)
    offset = 0
    for la, nv in zip(linargs, nvars):
        Y_ser += la @ X_data[offset : offset + nv, :]
        offset += nv

    assert np.allclose(Y_par, Y_ser, rtol=1e-5, atol=1e-5)


def test_in_place_raises_when_exceeds_max_traits():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, max_num_traits=2) as par:
        n, m = par.shape
        Y = np.ones((n, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            _ = par._rmatmat(Y, in_place=True)
        X = np.ones((m, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            _ = par._matmat(X, in_place=True)


def test_grm_matmat_matches_serial():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, _ = _load_serial_blocks(hdf5_path)

    with GRMOperator.from_hdf5(hdf5_path, num_processes=2) as grm:
        n = grm.shape[0]
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3)).astype(np.float32)

        # Parallel result
        Y_par = grm @ X

    # Serial result: sum over blocks of L @ L.T @ X
    Y_ser = np.zeros((n, 3), dtype=np.float32)
    for la in linargs:
        Y_ser += la @ la.T @ X

    assert np.allclose(Y_par, Y_ser, rtol=1e-5, atol=1e-5)


def test_grm_matmat_with_alpha():
    from scipy.sparse import diags
    from scipy.sparse.linalg import aslinearoperator
    
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    linargs, _ = _load_serial_blocks(hdf5_path)
    
    alpha = 0.5
    
    with GRMOperator.from_hdf5(hdf5_path, num_processes=2, alpha=alpha) as grm:
        n = grm.shape[0]
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3)).astype(np.float32)
        
        # Parallel result
        Y_par = grm @ X
    
    # Serial result: sum over blocks of L.normalized @ K @ L.normalized.T @ X
    Y_ser = np.zeros((n, 3), dtype=np.float32)
    for la in linargs:
        pq = la.allele_frequencies * (1 - la.allele_frequencies)
        K = aslinearoperator(diags(pq ** (1 + alpha)))
        Y_ser += la.normalized @ K @ la.normalized.T @ X
    
    assert np.allclose(Y_par, Y_ser, rtol=1e-5, atol=1e-5)
