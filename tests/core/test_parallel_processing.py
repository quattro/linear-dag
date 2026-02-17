import os
import tempfile

from inspect import signature
from pathlib import Path

import numpy as np
import pytest

from linear_dag.core.lineararg import LinearARG, list_blocks
from linear_dag.core.parallel_processing import GRMOperator, ParallelOperator


def test_from_hdf5_signature_constructor_contract_parameter_order():
    expected = [
        "hdf5_file",
        "num_processes",
        "max_num_traits",
        "maf_log10_threshold",
        "block_metadata",
        "bed_file",
        "bed_maf_log10_threshold",
        "alpha",
    ]

    parallel_params = list(signature(ParallelOperator.from_hdf5).parameters)
    grm_params = list(signature(GRMOperator.from_hdf5).parameters)

    assert parallel_params == expected
    assert grm_params == expected
    assert parallel_params == grm_params


def test_from_hdf5_signature_constructor_contract_default_values():
    parallel_signature = signature(ParallelOperator.from_hdf5)
    grm_signature = signature(GRMOperator.from_hdf5)

    assert parallel_signature.parameters["max_num_traits"].default == 8
    assert grm_signature.parameters["max_num_traits"].default == 8

    for name in (
        "maf_log10_threshold",
        "block_metadata",
        "bed_file",
        "bed_maf_log10_threshold",
    ):
        assert parallel_signature.parameters[name].default is None
        assert grm_signature.parameters[name].default is None

    assert parallel_signature.parameters["alpha"].default == -1.0
    assert grm_signature.parameters["alpha"].default == -1.0


def test_parallel_operator(linarg_h5_path: Path):
    """
    Test that ParallelOperator gives the same result as serial processing.
    """
    # 1. Setup
    hdf5_path = linarg_h5_path
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
    blocks = list_blocks(hdf5_path)["block_name"]

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


def _load_serial_blocks(hdf5_path: Path):
    blocks_df = list_blocks(hdf5_path)
    block_names = blocks_df.get_column("block_name").to_list()
    n_variants = blocks_df.get_column("n_variants").to_list()
    linargs = [LinearARG.read(hdf5_path, block) for block in block_names]
    return linargs, n_variants


def test_matmat_matches_serial(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
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

    assert np.allclose(Y_par, Y_ser, rtol=1e-3, atol=1e-3)


def test_rmatmat_matches_serial(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
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


def test_parallel_operator_alpha_is_no_op(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
    rng = np.random.default_rng(2026)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, alpha=-1.0) as op_alpha_neg:
        _, m = op_alpha_neg.shape
        x = rng.standard_normal((m, 3)).astype(np.float32)
        y_alpha_neg = op_alpha_neg @ x

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, alpha=0.75) as op_alpha_pos:
        y_alpha_pos = op_alpha_pos @ x

    assert y_alpha_neg.shape == y_alpha_pos.shape
    assert y_alpha_neg.dtype == y_alpha_pos.dtype
    np.testing.assert_allclose(y_alpha_neg, y_alpha_pos, rtol=1e-5, atol=1e-5)


def test_number_of_carriers_matches_serial(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
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


def test_rmatmat_in_place_view_and_correctness(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
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


def test_matmat_in_place_uses_shared_variant_data(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
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


def test_in_place_raises_when_exceeds_max_traits(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, max_num_traits=2) as par:
        n, m = par.shape
        Y = np.ones((n, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            _ = par._rmatmat(Y, in_place=True)
        X = np.ones((m, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            _ = par._matmat(X, in_place=True)


def test_grm_matmat_matches_serial(linarg_h5_path: Path):
    hdf5_path = linarg_h5_path
    linargs, _ = _load_serial_blocks(hdf5_path)

    with GRMOperator.from_hdf5(hdf5_path, num_processes=1) as grm:
        n = grm.shape[0]
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3)).astype(np.float32)

        # Parallel result
        Y_par = grm @ X

    # Serial result: sum over blocks of L.normalized @ L.normalized.T @ X
    # GRMOperator aggregates unscaled block contributions.
    Y_ser = np.zeros((n, 3), dtype=np.float32)
    for la in linargs:
        la_norm = la.normalized
        Y_ser += la_norm @ la_norm.T @ X

    print(f"Y_par: {Y_par[:, 0]}, Y_ser: {Y_ser[:, 0]}")
    assert np.allclose(Y_par, Y_ser, rtol=1e-4, atol=1e-4)


def test_grm_matmat_with_alpha(linarg_h5_path: Path):
    from scipy.sparse import diags
    from scipy.sparse.linalg import aslinearoperator

    hdf5_path = linarg_h5_path
    linargs, _ = _load_serial_blocks(hdf5_path)

    alpha = 0.5

    with GRMOperator.from_hdf5(hdf5_path, num_processes=2, alpha=alpha) as grm:
        n = grm.shape[0]
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3)).astype(np.float32)

        # Parallel result
        Y_par = grm @ X

    # Serial result: sum over blocks of L.normalized @ K @ L.normalized.T @ X
    # GRMOperator aggregates unscaled block contributions.
    Y_ser = np.zeros((n, 3), dtype=np.float32)
    for la in linargs:
        pq = la.allele_frequencies * (1 - la.allele_frequencies)
        K = aslinearoperator(diags(pq ** (1 + alpha)))
        Y_ser += la.normalized @ K @ la.normalized.T @ X

    assert np.allclose(Y_par, Y_ser, rtol=1e-3, atol=1e-3)


def test_maf_threshold_filtering(linarg_h5_path: Path):
    """Test MAF threshold filtering in ParallelOperator."""
    hdf5_path = linarg_h5_path

    # Create a temporary file with threshold data
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Read blocks and rewrite with threshold data
        metadata = list_blocks(hdf5_path)
        for block_info in metadata.iter_rows(named=True):
            linarg = LinearARG.read(hdf5_path, block_info["block_name"])
            block_dict = {
                "chrom": str(block_info["chrom"]),
                "start": int(block_info["start"]),
                "end": int(block_info["end"]),
            }
            linarg.write(tmp_path, block_info=block_dict)

        # Test with MAF > 0.01 threshold
        maf_log10_threshold = -2

        # Test with parallel operator
        with ParallelOperator.from_hdf5(
            tmp_path, max_num_traits=2, num_processes=2, maf_log10_threshold=maf_log10_threshold
        ) as linarg_op:
            x = np.random.randn(linarg_op.shape[1], 3).astype(np.float32)
            result_parallel = linarg_op @ x

        # Compute serial version with filtering
        metadata_filtered = list_blocks(tmp_path)
        result_serial = 0
        variant_counter = 0
        total_filtered_variants = 0

        threshold_value = 10**maf_log10_threshold

        for block_info in metadata_filtered.iter_rows(named=True):
            linarg = LinearARG.read(tmp_path, block_info["block_name"])

            # Apply same filtering as worker
            linarg.filter_variants_by_maf(threshold_value)
            n_filtered = linarg.shape[1]
            total_filtered_variants += n_filtered

            result = linarg @ x[variant_counter : variant_counter + n_filtered, :]
            variant_counter += n_filtered
            result_serial += result

        # Check shapes match
        assert (
            linarg_op.shape[1] == total_filtered_variants
        ), f"Shape mismatch: operator has {linarg_op.shape[1]} variants, expected {total_filtered_variants}"

        # Check that parallel and serial results match
        assert np.allclose(
            result_parallel, result_serial, rtol=1e-2, atol=1e-2
        ), "MAF filtered multiplication results do not match"

        # Test that filtering actually reduces variant count
        with ParallelOperator.from_hdf5(tmp_path, max_num_traits=2, num_processes=1) as linarg_op_unfiltered:
            unfiltered_variants = linarg_op_unfiltered.shape[1]

        assert total_filtered_variants < unfiltered_variants, "Filtering should reduce variant count"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
