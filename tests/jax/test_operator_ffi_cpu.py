# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.kernels import ffi_cpu


def _src_of_edge(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(indptr.shape[0] - 1, dtype=np.int32), np.diff(indptr))


def _operator_from_case(oracle_case, *, backend: Backend) -> JaxLinearARG:
    linarg = oracle_case.linarg
    return JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
        src_of_edge=_src_of_edge(linarg.A.indptr),
        variant_indices=linarg.variant_indices,
        flip=linarg.flip,
        sample_indices=linarg.sample_indices,
        nonunique_indices=linarg.nonunique_indices,
        n_variants=linarg.shape[1],
        n_samples=linarg.shape[0],
        backend=backend,
        dtype=jnp.float32,
    )


def test_jax_lineararg_ffi_cpu_forward_product_matches_oracle(oracle_case):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = _operator_from_case(oracle_case, backend=Backend.FFI_CPU)

    assert op.backend is Backend.FFI_CPU
    np.testing.assert_allclose(np.asarray(op.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)


def test_jax_lineararg_ffi_cpu_vmapped_matvec_matches_matmat_for_flipped_variants(
    linarg_h5_path,
    first_block_name,
):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    from tests.jax.oracle import make_oracle_cases

    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    case = cases["flipped_k3"]
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = _operator_from_case(case, backend=Backend.FFI_CPU)
    w = jnp.asarray(case.w)

    vmapped_matvec = jax.vmap(op.matvec, in_axes=1, out_axes=1)(w)

    assert op.backend is Backend.FFI_CPU
    assert case.flip_prob > 0
    np.testing.assert_allclose(
        np.asarray(vmapped_matvec),
        np.asarray(op.matmat(w)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_jax_lineararg_ffi_cpu_reverse_product_matches_oracle(oracle_case):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = _operator_from_case(oracle_case, backend=Backend.FFI_CPU)

    assert op.backend is Backend.FFI_CPU
    np.testing.assert_allclose(np.asarray(op.rmatmat(oracle_case.y)), oracle_case.XTy, rtol=1e-5, atol=1e-5)


def test_jax_lineararg_auto_resolves_to_ffi_cpu_when_handler_is_available(oracle_case):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = _operator_from_case(oracle_case, backend=Backend.AUTO)

    assert op.backend is Backend.FFI_CPU
    np.testing.assert_allclose(np.asarray(op.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)


def test_jax_lineararg_explicit_ffi_cpu_fallback_warns_and_runs_pure_jax(monkeypatch, oracle_case):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)

    with pytest.warns(UserWarning, match="FFI_CPU backend is unavailable"):
        op = _operator_from_case(oracle_case, backend=Backend.FFI_CPU)

    assert op.backend is Backend.PURE_JAX
    np.testing.assert_allclose(np.asarray(op.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)
