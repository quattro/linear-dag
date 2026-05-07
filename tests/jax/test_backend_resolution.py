# pattern: Functional Core

from __future__ import annotations

import numpy as np
import pytest

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG


def _minimal_operator_kwargs() -> dict:
    return {
        "indptr": np.array([0, 1, 1], dtype=np.int32),
        "indices": np.array([1], dtype=np.int32),
        "data": np.ones(1, dtype=np.float32),
        "src_of_edge": np.array([0], dtype=np.int32),
        "variant_indices": np.array([0], dtype=np.int32),
        "flip": np.array([False]),
        "sample_indices": np.array([1], dtype=np.int32),
        "nonunique_indices": np.array([0, 1], dtype=np.int32),
        "n_variants": 1,
        "n_samples": 1,
    }


@pytest.mark.parametrize(
    ("platform", "available", "expected"),
    [
        ("cpu", True, Backend.FFI_CPU),
        ("cpu", False, Backend.PURE_JAX),
        ("gpu", True, Backend.PALLAS_GPU),
        ("tpu", True, Backend.PURE_JAX),
    ],
)
def test_auto_backend_resolves_by_platform_and_ffi_availability(monkeypatch, platform, available, expected):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: available)

    assert jaxlinarg_operator.resolve_backend(Backend.AUTO, platform=platform) is expected


def test_explicit_ffi_cpu_resolves_to_pure_jax_when_handler_is_absent(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)

    with pytest.warns(UserWarning, match="FFI_CPU backend is unavailable"):
        backend = jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu")

    assert backend is Backend.PURE_JAX


def test_explicit_ffi_cpu_resolves_to_ffi_cpu_when_handler_is_available(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: True)

    assert jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu") is Backend.FFI_CPU


def test_invalid_backend_value_fails_at_operator_construction():
    kwargs = _minimal_operator_kwargs()
    kwargs["backend"] = "not-a-backend"

    with pytest.raises(ValueError, match="not-a-backend"):
        JaxLinearARG.from_lineararg_arrays(**kwargs)
