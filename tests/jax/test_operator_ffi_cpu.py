# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.kernels import ffi_cpu


def _direct_ffi_constructor_kwargs() -> dict:
    return {
        "indptr": np.array([0, 1, 1], dtype=np.int32),
        "indices": np.array([1], dtype=np.int32),
        "data": np.ones(1, dtype=np.float32),
        "variant_indices": np.array([0], dtype=np.int32),
        "flip": np.array([False]),
        "sample_indices": np.array([1], dtype=np.int32),
        "nonunique_indices": np.array([0, 1], dtype=np.int32),
        "allele_counts": np.array([-1], dtype=np.int32),
        "n_variants": 1,
        "n_samples": 1,
        "n_nonunique_indices": 2,
        "min_index_to_keep": 1,
        "backend": Backend.FFI_CPU,
        "dtype": jnp.float32,
    }


def _operator_from_case(oracle_case, *, backend: Backend) -> JaxLinearARG:
    linarg = oracle_case.linarg
    return JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
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


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="JAX x64 is disabled")
def test_jax_lineararg_ffi_cpu_float64_products_match_oracle(oracle_case):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    linarg = oracle_case.linarg
    op = JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
        variant_indices=linarg.variant_indices,
        flip=linarg.flip,
        sample_indices=linarg.sample_indices,
        nonunique_indices=linarg.nonunique_indices,
        n_variants=linarg.shape[1],
        n_samples=linarg.shape[0],
        backend=Backend.FFI_CPU,
        dtype=jnp.float64,
    )
    w = np.asarray(oracle_case.w, dtype=np.float64)
    y = np.asarray(oracle_case.y, dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(op.matmat(w)),
        linarg @ w,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(op.rmatmat(y)),
        linarg.T @ y,
        rtol=1e-10,
        atol=1e-10,
    )


def test_direct_ffi_cpu_constructor_retains_float_edge_compatibility():
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = JaxLinearARG(**_direct_ffi_constructor_kwargs())

    assert op.data.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(op.matmat(np.array([3.0], dtype=np.float32))), np.array([3.0]))
    np.testing.assert_array_equal(np.asarray(op.rmatmat(np.array([4.0], dtype=np.float32))), np.array([4.0]))


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="JAX x64 is disabled")
@pytest.mark.parametrize(
    "field",
    ["indptr", "indices", "variant_indices", "sample_indices", "nonunique_indices"],
)
def test_direct_ffi_cpu_constructor_rejects_int64_structural_buffers(field):
    kwargs = _direct_ffi_constructor_kwargs()
    kwargs[field] = np.asarray(kwargs[field], dtype=np.int64)

    with pytest.raises(ValueError, match=rf"{field} must have dtype int32"):
        JaxLinearARG(**kwargs)


def test_direct_ffi_cpu_constructor_rejects_nonboolean_flip_buffer():
    kwargs = _direct_ffi_constructor_kwargs()
    kwargs["flip"] = np.array([0], dtype=np.uint8)

    with pytest.raises(ValueError, match="flip must have dtype bool"):
        JaxLinearARG(**kwargs)


def test_direct_ffi_cpu_constructor_rejects_float_edge_dtype_mismatch():
    kwargs = _direct_ffi_constructor_kwargs()
    kwargs["data"] = np.ones(1, dtype=np.float16)

    with pytest.raises(ValueError, match="floating data dtype must match computation dtype"):
        JaxLinearARG(**kwargs)


def test_direct_ffi_cpu_constructor_rejects_unsupported_computation_dtype():
    kwargs = _direct_ffi_constructor_kwargs()
    kwargs["data"] = np.ones(1, dtype=np.float16)
    kwargs["dtype"] = jnp.float16

    with pytest.raises(ValueError, match="supports only float32 and float64 computation dtypes"):
        JaxLinearARG(**kwargs)


def test_factory_ffi_cpu_constructor_rejects_unsupported_computation_dtype(oracle_case):
    linarg = oracle_case.linarg

    with pytest.raises(ValueError, match="supports only float32 and float64 computation dtypes"):
        JaxLinearARG.from_lineararg_arrays(
            indptr=linarg.A.indptr,
            indices=linarg.A.indices,
            data=linarg.A.data,
            variant_indices=linarg.variant_indices,
            flip=linarg.flip,
            sample_indices=linarg.sample_indices,
            nonunique_indices=linarg.nonunique_indices,
            n_variants=linarg.shape[1],
            n_samples=linarg.shape[0],
            backend=Backend.FFI_CPU,
            dtype=jnp.float16,
        )


def test_direct_auto_constructor_uses_pure_jax_for_unsupported_ffi_dtype():
    kwargs = _direct_ffi_constructor_kwargs()
    kwargs["data"] = np.ones(1, dtype=np.float16)
    kwargs["backend"] = Backend.AUTO
    kwargs["dtype"] = jnp.float16

    op = JaxLinearARG(**kwargs)

    assert op.backend is Backend.PURE_JAX


def test_factory_auto_constructor_uses_pure_jax_for_unsupported_ffi_dtype(oracle_case):
    linarg = oracle_case.linarg
    op = JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
        variant_indices=linarg.variant_indices,
        flip=linarg.flip,
        sample_indices=linarg.sample_indices,
        nonunique_indices=linarg.nonunique_indices,
        n_variants=linarg.shape[1],
        n_samples=linarg.shape[0],
        backend=Backend.AUTO,
        dtype=jnp.float16,
    )

    assert op.backend is Backend.PURE_JAX


@pytest.mark.parametrize(
    ("operation", "target"),
    [
        ("matmat", "linear_dag_jaxlinarg_matmat_f32"),
        ("rmatmat", "linear_dag_jaxlinarg_rmatmat_f32"),
    ],
)
def test_jax_lineararg_ffi_cpu_lowers_each_product_to_one_fused_call(oracle_case, operation, target):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = _operator_from_case(oracle_case, backend=Backend.FFI_CPU)
    values = jnp.asarray(oracle_case.w if operation == "matmat" else oracle_case.y)

    stablehlo = jax.jit(lambda operator, rhs: getattr(operator, operation)(rhs)).lower(op, values).as_text()

    assert target in stablehlo
    assert stablehlo.count("stablehlo.custom_call") == 1
    assert "stablehlo.scatter" not in stablehlo
    assert "stablehlo.gather" not in stablehlo
    assert "stablehlo.reduce" not in stablehlo
    assert "stablehlo.select" not in stablehlo


def test_jax_lineararg_ffi_cpu_keeps_graph_arrays_as_operands(oracle_case):
    if jax.default_backend() != "cpu":
        pytest.skip("FFI_CPU operator dispatch is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    op = _operator_from_case(oracle_case, backend=Backend.FFI_CPU)
    values = jnp.asarray(oracle_case.w)

    closed_jaxpr = jax.make_jaxpr(lambda operator, rhs: operator.matmat(rhs))(op, values)
    constant_bytes = sum(
        int(constant.size * constant.dtype.itemsize)
        for constant in closed_jaxpr.consts
        if isinstance(constant, (jax.Array, np.ndarray))
    )

    assert constant_bytes == 0


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
