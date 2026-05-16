# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend
from linear_dag.core.jaxlinarg.kernels import ffi_cpu
from linear_dag.core.jaxlinarg.kernels.pure_jax import (
    pure_jax_solve_backward_compressed,
    pure_jax_solve_forward_compressed,
)


def _solve_args(dtype=np.float32, n_cols: int = 1) -> tuple:
    indptr = jnp.asarray(np.array([0, 1, 1], dtype=np.int32))
    indices = jnp.asarray(np.array([1], dtype=np.int32))
    data = jnp.asarray(np.ones(1, dtype=dtype))
    nonunique_indices = jnp.asarray(np.array([0, 1], dtype=np.int32))
    b = jnp.asarray(np.vstack([np.arange(2, n_cols + 2), np.zeros(n_cols)]).astype(dtype))
    return indptr, indices, data, nonunique_indices, 0, b


@pytest.mark.parametrize(
    ("wrapper_name", "expected_target"),
    [
        ("ffi_cpu_solve_forward", "linear_dag_jaxlinarg_solve_forward_f32"),
        ("ffi_cpu_solve_backward", "linear_dag_jaxlinarg_solve_backward_f32"),
    ],
)
def test_ffi_solve_wrapper_uses_explicit_result_metadata_and_vmap_method(monkeypatch, wrapper_name, expected_target):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_available", lambda: True)
    captured = {}

    def fake_ffi_call(target_name, result_shape_dtypes, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args, **attrs):
            captured["args"] = args
            captured["attrs"] = attrs
            return args[-1] + 1

        return call

    monkeypatch.setattr(ffi_cpu.jax.ffi, "ffi_call", fake_ffi_call)

    result = getattr(ffi_cpu, wrapper_name)(*_solve_args())

    assert captured["target_name"] == expected_target
    assert captured["result_shape_dtypes"] == jax.ShapeDtypeStruct((2, 1), jnp.float32)
    assert captured["kwargs"]["vmap_method"] == "sequential"
    assert captured["kwargs"]["input_output_aliases"] == {4: 0}
    assert captured["attrs"]["min_index_to_keep"] == 0
    np.testing.assert_allclose(np.asarray(result), np.array([[3.0], [1.0]], dtype=np.float32))


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="JAX x64 is disabled")
def test_ffi_solve_wrapper_uses_float64_target_when_enabled(monkeypatch):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_available", lambda: True)
    captured = {}

    def fake_ffi_call(target_name, result_shape_dtypes, **kwargs):
        del kwargs
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        return lambda *args, **attrs: args[-1]

    monkeypatch.setattr(ffi_cpu.jax.ffi, "ffi_call", fake_ffi_call)

    result = ffi_cpu.ffi_cpu_solve_forward(*_solve_args(np.float64))

    assert captured["target_name"] == "linear_dag_jaxlinarg_solve_forward_f64"
    assert captured["result_shape_dtypes"] == jax.ShapeDtypeStruct((2, 1), result.dtype)


def test_ffi_solve_wrapper_does_not_call_ffi_when_unavailable(monkeypatch):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_available", lambda: False)

    def fail_ffi_call(*args, **kwargs):
        raise AssertionError("ffi_call should not be invoked when the handler is absent")

    monkeypatch.setattr(ffi_cpu.jax.ffi, "ffi_call", fail_ffi_call)

    with pytest.raises(RuntimeError, match="FFI CPU backend is unavailable"):
        ffi_cpu.ffi_cpu_solve_forward(*_solve_args())


def test_native_ffi_cpu_handler_is_available_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert ffi_cpu.is_ffi_cpu_available()


def test_native_ffi_cpu_blas_flag_is_boolean_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert isinstance(ffi_cpu.is_ffi_cpu_blas_enabled(), bool)


def test_native_ffi_cpu_build_metadata_has_expected_types_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu._import_ffi_cpu_impl.cache_clear()
    ffi_cpu._load_ffi_cpu_impl.cache_clear()
    ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert isinstance(ffi_cpu.is_ffi_cpu_built(), bool)
    assert ffi_cpu.ffi_cpu_blas_backend() in {"accelerate", "openblas", "blas", "none"}
    assert isinstance(ffi_cpu.is_ffi_cpu_native_tuning_enabled(), bool)


def test_auto_backend_resolves_to_ffi_cpu_when_native_handler_is_available():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert jaxlinarg_operator.resolve_backend(Backend.AUTO, platform="cpu") is Backend.FFI_CPU


@pytest.mark.parametrize("n_cols", [1, 64])
def test_native_ffi_cpu_forward_and_backward_match_pure_jax_on_cpu(n_cols):
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    indptr, indices, data, nonunique_indices, min_index_to_keep, b = _solve_args(n_cols=n_cols)
    backward_b = jnp.asarray(np.vstack([np.zeros(n_cols), np.arange(3, n_cols + 3)]).astype(np.float32))

    forward = ffi_cpu.ffi_cpu_solve_forward(
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )
    expected_forward = pure_jax_solve_forward_compressed(
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )
    backward = ffi_cpu.ffi_cpu_solve_backward(
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        backward_b,
    )
    expected_backward = pure_jax_solve_backward_compressed(
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        backward_b,
    )

    np.testing.assert_allclose(np.asarray(forward), np.asarray(expected_forward), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(backward), np.asarray(expected_backward), rtol=1e-6, atol=1e-6)
