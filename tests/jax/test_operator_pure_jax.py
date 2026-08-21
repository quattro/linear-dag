# pattern: Functional Core

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.typing import ArrayLike

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG

_USE_CASE_NONUNIQUE = object()


def _minimal_operator_kwargs() -> dict:
    return {
        "indptr": np.array([0, 1, 1], dtype=np.int32),
        "indices": np.array([1], dtype=np.int32),
        "data": np.ones(1, dtype=np.float32),
        "variant_indices": np.array([0], dtype=np.int32),
        "flip": np.array([False]),
        "sample_indices": np.array([1], dtype=np.int32),
        "nonunique_indices": np.array([0, 1], dtype=np.int32),
        "n_variants": 1,
        "n_samples": 1,
        "backend": Backend.PURE_JAX,
        "dtype": jnp.float32,
    }


def _operator_from_case(oracle_case, *, nonunique_indices=_USE_CASE_NONUNIQUE) -> JaxLinearARG:
    linarg = oracle_case.linarg
    if nonunique_indices is _USE_CASE_NONUNIQUE:
        nonunique_indices = linarg.nonunique_indices
    return JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
        variant_indices=linarg.variant_indices,
        flip=linarg.flip,
        sample_indices=linarg.sample_indices,
        nonunique_indices=cast(ArrayLike | None, nonunique_indices),
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


def test_jax_lineararg_caches_flipped_variant_indices(linarg_h5_path, first_block_name):
    from tests.jax.oracle import make_oracle_cases

    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    case = cases["flipped_k3"]
    op = _operator_from_case(case)

    np.testing.assert_array_equal(np.asarray(op._flipped_variant_indices), np.flatnonzero(case.linarg.flip))
    np.testing.assert_allclose(np.asarray(op.matmat(case.w)), case.Xw, rtol=1e-5, atol=1e-5)


def test_jax_lineararg_vmapped_matvec_matches_matmat_for_flipped_variants(linarg_h5_path, first_block_name):
    from tests.jax.oracle import make_oracle_cases

    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    case = cases["flipped_k3"]
    op = _operator_from_case(case)
    w = jnp.asarray(case.w)

    vmapped_matvec = jax.vmap(op.matvec, in_axes=1, out_axes=1)(w)

    assert case.flip_prob > 0
    np.testing.assert_allclose(
        np.asarray(vmapped_matvec),
        np.asarray(op.matmat(w)),
        rtol=1e-5,
        atol=1e-5,
    )


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


def test_jax_lineararg_transpose_view_transposes_back_to_parent(oracle_case):
    op = _operator_from_case(oracle_case)
    w = np.asarray(oracle_case.w).reshape(oracle_case.linarg.shape[1], -1)

    assert op.T.T is op
    assert op.T.T.shape == op.shape
    np.testing.assert_allclose(
        np.asarray(op.T.T.matmat(w)),
        np.asarray(oracle_case.Xw).reshape(op.shape[0], -1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_jax_lineararg_forward_product_uses_nonunique_buffer(monkeypatch, oracle_case):
    linarg = oracle_case.linarg
    expected_rows = int(np.max(linarg.nonunique_indices)) + 1
    assert expected_rows < linarg.A.shape[0]
    op = _operator_from_case(oracle_case)

    def fake_solve(*args):
        b = args[-1]
        assert b.shape[0] == expected_rows
        return b

    monkeypatch.setattr(jaxlinarg_operator, "_solve_forward", fake_solve)

    op.matmat(np.zeros((op.shape[1], 1), dtype=np.float32))


def test_jax_lineararg_reverse_product_uses_nonunique_buffer(monkeypatch, oracle_case):
    linarg = oracle_case.linarg
    expected_rows = int(np.max(linarg.nonunique_indices)) + 1
    assert expected_rows < linarg.A.shape[0]
    op = _operator_from_case(oracle_case)

    def fake_solve(*args):
        b = args[-1]
        assert b.shape[0] == expected_rows
        return b

    monkeypatch.setattr(jaxlinarg_operator, "_solve_backward", fake_solve)

    op.rmatmat(np.zeros((op.shape[0], 1), dtype=np.float32))


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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("indices", np.array([-1], dtype=np.int32)),
        ("variant_indices", np.array([-1], dtype=np.int32)),
        ("sample_indices", np.array([-1], dtype=np.int32)),
        ("nonunique_indices", np.array([-1, 1], dtype=np.int32)),
    ],
)
def test_from_lineararg_arrays_rejects_negative_indices(field, bad_value):
    kwargs = _minimal_operator_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=f"{field} contains a negative index"):
        JaxLinearARG.from_lineararg_arrays(**kwargs)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("indptr", np.array([1, 2, 2], dtype=np.int32), "indptr must start at 0"),
        ("indptr", np.array([0, 2, 1], dtype=np.int32), "indptr must be monotonic"),
        ("indices", np.array([0], dtype=np.int32), "indices must be greater than their source nodes"),
    ],
)
def test_from_lineararg_arrays_rejects_malformed_graph_structure(field, bad_value, message):
    kwargs = _minimal_operator_kwargs()
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=message):
        JaxLinearARG.from_lineararg_arrays(**kwargs)
