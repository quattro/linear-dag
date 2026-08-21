# pattern: Functional Core

from __future__ import annotations

import warnings

from collections.abc import Sequence
from typing import cast

import jax
import numpy as np
import pytest

from jax.sharding import Mesh

import linear_dag.core.jaxlinarg.operator as jaxlinarg_operator

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG, JaxParallelOperator
from linear_dag.core.jaxlinarg.ingress import (
    _packed_from_block_arrays,
    from_block_arrays,
    read_hdf5_block_arrays,
)
from linear_dag.core.jaxlinarg.packing import LinearARGBlockArrays


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


def test_explicit_ffi_cpu_operator_fails_before_array_conversion(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "last_ffi_cpu_error",
        lambda: RuntimeError("exact targets are missing"),
    )

    class FailOnArrayConversion:
        def __array__(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("backend failure must precede graph-array conversion")

    kwargs = _minimal_operator_kwargs()
    kwargs["indptr"] = FailOnArrayConversion()

    with pytest.raises(RuntimeError, match="exact.*exact targets are missing"):
        JaxLinearARG.from_lineararg_arrays(**kwargs, backend=Backend.FFI_CPU)


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


def test_packed_explicit_ffi_fails_before_consuming_source_blocks(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_packed_available", lambda: False)
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "last_ffi_cpu_packed_error",
        lambda: RuntimeError("packed targets are missing"),
    )

    class FailOnIteration:
        def __iter__(self):
            raise AssertionError("backend failure must precede packed source consumption")

    mesh = Mesh(np.asarray(jax.devices()[:1]), ("graph",))
    with pytest.raises(RuntimeError, match="packed.*packed targets are missing"):
        _packed_from_block_arrays(
            cast(Sequence[LinearARGBlockArrays], FailOnIteration()),
            mesh=mesh,
            backend=Backend.FFI_CPU,
        )


def test_exact_only_extension_keeps_exact_auto_ffi_and_silently_falls_back_for_packed_auto(
    monkeypatch,
    linarg_h5_path,
    first_block_name,
):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: True)
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_packed_available", lambda: False)
    monkeypatch.setattr(jaxlinarg_operator.jax, "default_backend", lambda: "cpu")
    arrays = read_hdf5_block_arrays(linarg_h5_path, first_block_name)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("graph",))

    with warnings.catch_warnings(record=True) as recorded_warnings:
        exact = from_block_arrays(arrays, backend=Backend.AUTO)
        packed = _packed_from_block_arrays(
            (arrays,),
            mesh=mesh,
            backend=Backend.AUTO,
        ).operator

    assert recorded_warnings == []
    assert exact.backend is Backend.FFI_CPU
    assert packed.backend is Backend.PURE_JAX


def test_parallel_auto_operator_records_resolved_exact_backend(
    monkeypatch,
    linarg_h5_path,
    linarg_block_metadata,
):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: True)
    monkeypatch.setattr(jaxlinarg_operator.jax, "default_backend", lambda: "cpu")
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))

    operator = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=mesh,
        block_metadata=linarg_block_metadata,
        backend=Backend.AUTO,
    )

    assert operator.backend is Backend.FFI_CPU
