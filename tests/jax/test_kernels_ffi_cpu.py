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


def _packed_solve_args(
    dtype=np.float32,
    *,
    n_blocks: int = 2,
    descriptor_capacity: int = 3,
    n_cols: int = 2,
) -> tuple:
    """Return rebased flat graph buffers and inert packed descriptors."""
    if n_blocks not in {0, 1, 2}:
        raise ValueError("test fixture supports zero, one, or two blocks")
    indptr = jnp.asarray(np.array([0, 1, 1, 1, 2, 2], dtype=np.int32))
    indices = jnp.asarray(np.array([1, 3], dtype=np.int32))
    data = jnp.asarray(np.array([1.0, 2.0], dtype=dtype))
    nonunique_indices = jnp.asarray(np.arange(4, dtype=np.int32))
    descriptor_index = {name: index for index, name in enumerate(ffi_cpu.PACKED_FFI_DESCRIPTOR_FIELDS)}
    descriptors = np.zeros(
        (descriptor_capacity, len(ffi_cpu.PACKED_FFI_DESCRIPTOR_FIELDS)),
        dtype=np.int32,
    )
    descriptors[:, descriptor_index["version"]] = ffi_cpu.PACKED_FFI_DESCRIPTOR_VERSION
    block_rows = (
        {
            "valid": 1,
            "node_start": 0,
            "node_length": 2,
            "indptr_start": 0,
            "indptr_length": 3,
            "edge_start": 0,
            "edge_length": 1,
            "compressed_start": 0,
            "compressed_length": 2,
            "min_index_to_keep": 0,
        },
        {
            "valid": 1,
            "node_start": 2,
            "node_length": 2,
            "indptr_start": 3,
            "indptr_length": 3,
            "edge_start": 1,
            "edge_length": 1,
            "compressed_start": 2,
            "compressed_length": 2,
            "min_index_to_keep": 2,
        },
    )
    for slot, values in enumerate(block_rows[:n_blocks]):
        for name, value in values.items():
            descriptors[slot, descriptor_index[name]] = value
    b = np.zeros((4, n_cols), dtype=dtype)
    if n_blocks:
        b[0] = np.arange(1, n_cols + 1)
    if n_blocks == 2:
        b[2] = np.arange(3, n_cols + 3)
    return (
        indptr,
        indices,
        data,
        nonunique_indices,
        jnp.asarray(descriptors),
        jnp.asarray(b),
    )


def _packed_reference(args: tuple, *, forward: bool) -> np.ndarray:
    indptr, indices, data, nonunique_indices, descriptors, b = args
    result = jnp.asarray(b)
    descriptor_index = {name: index for index, name in enumerate(ffi_cpu.PACKED_FFI_DESCRIPTOR_FIELDS)}
    solve = pure_jax_solve_forward_compressed if forward else pure_jax_solve_backward_compressed
    for row in np.asarray(descriptors):
        if not row[descriptor_index["valid"]]:
            continue
        node_start = int(row[descriptor_index["node_start"]])
        node_length = int(row[descriptor_index["node_length"]])
        indptr_start = int(row[descriptor_index["indptr_start"]])
        indptr_length = int(row[descriptor_index["indptr_length"]])
        edge_start = int(row[descriptor_index["edge_start"]])
        edge_length = int(row[descriptor_index["edge_length"]])
        compressed_start = int(row[descriptor_index["compressed_start"]])
        compressed_length = int(row[descriptor_index["compressed_length"]])
        min_index_to_keep = int(row[descriptor_index["min_index_to_keep"]])
        local_result = solve(
            indptr[indptr_start : indptr_start + indptr_length] - edge_start,
            indices[edge_start : edge_start + edge_length] - node_start,
            data[edge_start : edge_start + edge_length],
            nonunique_indices[node_start : node_start + node_length] - compressed_start,
            min_index_to_keep - node_start,
            result[compressed_start : compressed_start + compressed_length],
        )
        result = result.at[compressed_start : compressed_start + compressed_length].set(local_result)
    return np.asarray(result)


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


@pytest.mark.parametrize(
    ("wrapper_name", "expected_target"),
    [
        ("ffi_cpu_packed_solve_forward", "linear_dag_jaxlinarg_packed_solve_forward_f32"),
        ("ffi_cpu_packed_solve_backward", "linear_dag_jaxlinarg_packed_solve_backward_f32"),
    ],
)
def test_packed_ffi_solve_wrapper_uses_distinct_target_and_aliases_only_state(
    monkeypatch, wrapper_name, expected_target
):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_packed_available", lambda: True)
    captured = {}

    def fake_ffi_call(target_name, result_shape_dtypes, **kwargs):
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        captured["kwargs"] = kwargs

        def call(*args):
            captured["args"] = args
            return args[-1] + 1

        return call

    monkeypatch.setattr(ffi_cpu.jax.ffi, "ffi_call", fake_ffi_call)
    args = _packed_solve_args(n_blocks=1)

    result = getattr(ffi_cpu, wrapper_name)(*args)

    assert captured["target_name"] == expected_target
    assert captured["result_shape_dtypes"] == jax.ShapeDtypeStruct((4, 2), jnp.float32)
    assert captured["kwargs"]["vmap_method"] == "sequential"
    assert captured["kwargs"]["input_output_aliases"] == {5: 0}
    assert captured["args"] == args
    np.testing.assert_allclose(np.asarray(result), np.asarray(args[-1]) + 1)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="JAX x64 is disabled")
def test_packed_ffi_solve_wrapper_uses_float64_target_when_enabled(monkeypatch):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_packed_available", lambda: True)
    captured = {}

    def fake_ffi_call(target_name, result_shape_dtypes, **kwargs):
        del kwargs
        captured["target_name"] = target_name
        captured["result_shape_dtypes"] = result_shape_dtypes
        return lambda *args: args[-1]

    monkeypatch.setattr(ffi_cpu.jax.ffi, "ffi_call", fake_ffi_call)

    result = ffi_cpu.ffi_cpu_packed_solve_forward(*_packed_solve_args(np.float64))

    assert captured["target_name"] == "linear_dag_jaxlinarg_packed_solve_forward_f64"
    assert captured["result_shape_dtypes"] == jax.ShapeDtypeStruct((4, 2), result.dtype)


def test_packed_ffi_solve_wrapper_does_not_call_ffi_when_unavailable(monkeypatch):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_packed_available", lambda: False)
    monkeypatch.setattr(ffi_cpu, "last_ffi_cpu_packed_error", lambda: ImportError("missing packed targets"))

    def fail_ffi_call(*args, **kwargs):
        raise AssertionError("ffi_call should not be invoked when the handler is absent")

    monkeypatch.setattr(ffi_cpu.jax.ffi, "ffi_call", fail_ffi_call)

    with pytest.raises(RuntimeError, match="packed CPU FFI backend is unavailable.*missing packed targets"):
        ffi_cpu.ffi_cpu_packed_solve_forward(*_packed_solve_args(n_blocks=1))


def test_native_ffi_cpu_handler_is_available_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert ffi_cpu.is_ffi_cpu_available()


def test_native_packed_ffi_cpu_handler_is_available_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    ffi_cpu.is_ffi_cpu_packed_available.cache_clear()

    assert ffi_cpu.is_ffi_cpu_packed_available()


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


def test_native_ffi_cpu_registers_exact_and_packed_targets_on_cpu():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    registrations = ffi_cpu._import_ffi_cpu_impl().registrations()

    assert set(registrations) == {
        ffi_cpu.FFI_CPU_SOLVE_FORWARD_F32,
        ffi_cpu.FFI_CPU_SOLVE_BACKWARD_F32,
        ffi_cpu.FFI_CPU_SOLVE_FORWARD_F64,
        ffi_cpu.FFI_CPU_SOLVE_BACKWARD_F64,
        ffi_cpu.FFI_CPU_PACKED_SOLVE_FORWARD_F32,
        ffi_cpu.FFI_CPU_PACKED_SOLVE_BACKWARD_F32,
        ffi_cpu.FFI_CPU_PACKED_SOLVE_FORWARD_F64,
        ffi_cpu.FFI_CPU_PACKED_SOLVE_BACKWARD_F64,
    }


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_blocks", [0, 1, 2])
@pytest.mark.parametrize("forward", [True, False])
def test_native_packed_ffi_cpu_matches_repeated_pure_jax_solves(dtype, n_blocks, forward):
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    if dtype is np.float64 and not jax.config.jax_enable_x64:
        pytest.skip("JAX x64 is disabled")
    args = _packed_solve_args(dtype, n_blocks=n_blocks, descriptor_capacity=3)
    wrapper = ffi_cpu.ffi_cpu_packed_solve_forward if forward else ffi_cpu.ffi_cpu_packed_solve_backward

    result = wrapper(*args)

    np.testing.assert_allclose(
        np.asarray(result),
        _packed_reference(args, forward=forward),
        rtol=1e-12 if dtype is np.float64 else 1e-6,
        atol=1e-12 if dtype is np.float64 else 1e-6,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 999, "unsupported packed descriptor version"),
        ("valid", 2, "valid flag"),
        ("indptr_length", 2, "indptr length"),
        ("edge_start", -1, "edge span"),
        ("compressed_length", 99, "compressed span"),
        ("min_index_to_keep", 3, "min_index_to_keep"),
    ],
)
def test_native_packed_ffi_cpu_rejects_malformed_descriptor(field, value, message):
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    args = list(_packed_solve_args(n_blocks=1, descriptor_capacity=1))
    descriptors = np.asarray(args[4]).copy()
    descriptors[0, ffi_cpu.PACKED_FFI_DESCRIPTOR_FIELDS.index(field)] = value
    args[4] = jnp.asarray(descriptors)

    with pytest.raises(Exception, match=message):
        ffi_cpu.ffi_cpu_packed_solve_forward(*args).block_until_ready()


def test_native_packed_ffi_cpu_rejects_nonmonotonic_indptr():
    if jax.default_backend() != "cpu":
        pytest.skip("native CPU FFI handler is only required on CPU platforms")
    args = list(_packed_solve_args(n_blocks=1, descriptor_capacity=1))
    args[0] = jnp.asarray(np.array([0, 2, 1, 1, 2, 2], dtype=np.int32))

    with pytest.raises(Exception, match="indptr.*monotonic"):
        ffi_cpu.ffi_cpu_packed_solve_forward(*args).block_until_ready()


def test_packed_ffi_cpu_rejects_invalid_descriptor_shape_before_call():
    args = list(_packed_solve_args(n_blocks=1, descriptor_capacity=1))
    args[4] = jnp.zeros((1, 2), dtype=jnp.int32)

    with pytest.raises(ValueError, match="descriptor.*columns"):
        ffi_cpu.ffi_cpu_packed_solve_forward(*args)
