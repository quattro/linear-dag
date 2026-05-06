# pattern: Functional Core

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG

_USE_CASE_NONUNIQUE = object()


def _src_of_edge(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(indptr.shape[0] - 1, dtype=np.int32), np.diff(indptr))


def _operator_from_case(oracle_case, *, nonunique_indices=_USE_CASE_NONUNIQUE) -> JaxLinearARG:
    linarg = oracle_case.linarg
    if nonunique_indices is _USE_CASE_NONUNIQUE:
        nonunique_indices = linarg.nonunique_indices
    return JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
        src_of_edge=_src_of_edge(linarg.A.indptr),
        variant_indices=linarg.variant_indices,
        flip=linarg.flip,
        sample_indices=linarg.sample_indices,
        nonunique_indices=nonunique_indices,
        n_variants=linarg.shape[1],
        n_samples=linarg.shape[0],
        backend=Backend.PURE_JAX,
        dtype=jnp.float32,
    )


def test_jax_lineararg_shape_and_forward_product_match_oracle(oracle_case):
    op = _operator_from_case(oracle_case)

    assert op.shape == oracle_case.linarg.shape
    np.testing.assert_allclose(
        np.asarray(op.matmat(oracle_case.w)),
        oracle_case.Xw,
        rtol=1e-5,
        atol=1e-5,
    )


def test_jax_lineararg_reverse_product_match_oracle(oracle_case):
    op = _operator_from_case(oracle_case)

    np.testing.assert_allclose(
        np.asarray(op.rmatmat(oracle_case.y)),
        oracle_case.XTy,
        rtol=1e-5,
        atol=1e-5,
    )


def test_jax_lineararg_forward_product_handles_flipped_variants(linarg_h5_path, first_block_name):
    from tests.jax.oracle import make_oracle_cases

    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    case = cases["flipped_k3"]
    op = _operator_from_case(case)

    assert case.flip_prob > 0
    np.testing.assert_allclose(np.asarray(op.matmat(case.w)), case.Xw, rtol=1e-5, atol=1e-5)


def test_jax_lineararg_one_dimensional_inputs_restore_vector_outputs(oracle_case):
    op = _operator_from_case(oracle_case)
    w = np.asarray(oracle_case.w).reshape(oracle_case.linarg.shape[1], -1)[:, 0]
    y = np.asarray(oracle_case.y).reshape(oracle_case.linarg.shape[0], -1)[:, 0]

    xw = op.matmat(w)
    xty = op.rmatmat(y)
    matvec = op.matvec(w)
    rmatvec = op.rmatvec(y)

    assert xw.shape == (oracle_case.linarg.shape[0],)
    assert xty.shape == (oracle_case.linarg.shape[1],)
    assert matvec.shape == (oracle_case.linarg.shape[0],)
    assert rmatvec.shape == (oracle_case.linarg.shape[1],)
    np.testing.assert_allclose(np.asarray(xw), np.asarray(matvec), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(xty), np.asarray(rmatvec), rtol=1e-5, atol=1e-5)


def test_jax_lineararg_transpose_view_dispatches_reverse_product(oracle_case):
    op = _operator_from_case(oracle_case)

    np.testing.assert_allclose(
        np.asarray(op.T.matmat(oracle_case.y)),
        oracle_case.XTy,
        rtol=1e-5,
        atol=1e-5,
    )


def test_jax_lineararg_rejects_wrong_leading_dimensions(oracle_case):
    op = _operator_from_case(oracle_case)

    with pytest.raises(ValueError, match="expected leading dimension"):
        op.matmat(np.zeros((op.shape[1] + 1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="expected leading dimension"):
        op.rmatmat(np.zeros((op.shape[0] + 1, 1), dtype=np.float32))


def test_from_lineararg_arrays_synthesizes_identity_nonunique_mapping(oracle_case):
    op = _operator_from_case(oracle_case, nonunique_indices=None)

    np.testing.assert_array_equal(
        np.asarray(op.nonunique_indices),
        np.arange(oracle_case.linarg.A.shape[0], dtype=np.int32),
    )
