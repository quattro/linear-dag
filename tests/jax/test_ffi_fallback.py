# pattern: Functional Core

from __future__ import annotations

import warnings

import numpy as np
import pytest

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG


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
    }


def test_explicit_ffi_cpu_operator_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)

    with pytest.warns(UserWarning, match="FFI_CPU backend is unavailable"):
        op = JaxLinearARG.from_lineararg_arrays(
            **_minimal_operator_kwargs(),
            backend=Backend.FFI_CPU,
        )

    assert op.backend is Backend.PURE_JAX


def test_auto_operator_uses_pure_jax_without_warning_when_ffi_cpu_is_absent(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)
    monkeypatch.setattr(jaxlinarg_operator.jax, "default_backend", lambda: "cpu")

    with warnings.catch_warnings(record=True) as recorded_warnings:
        op = JaxLinearARG.from_lineararg_arrays(
            **_minimal_operator_kwargs(),
            backend=Backend.AUTO,
        )

    assert len(recorded_warnings) == 0
    assert op.backend is Backend.PURE_JAX
