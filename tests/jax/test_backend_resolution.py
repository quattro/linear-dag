# pattern: Functional Core

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import linear_dag.core.jaxlinarg.kernels as kernels_pkg
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
        "allele_counts": np.array([1], dtype=np.int32),
        "n_variants": 1,
        "n_samples": 1,
    }


@pytest.fixture(autouse=True)
def _isolate_ffi_cpu_availability_cache():
    jaxlinarg_operator.ffi_cpu._load_ffi_cpu_impl.cache_clear()
    jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available.cache_clear()
    yield
    jaxlinarg_operator.ffi_cpu._load_ffi_cpu_impl.cache_clear()
    jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available.cache_clear()


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
    monkeypatch.setattr(jaxlinarg_operator.jax, "default_backend", lambda: platform)
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: available)
    if expected is Backend.PALLAS_GPU:
        monkeypatch.setattr(jaxlinarg_operator.pallas_gpu, "pl", object())

    assert jaxlinarg_operator.resolve_backend(Backend.AUTO) is expected


def test_auto_backend_falls_back_to_pure_jax_when_pallas_import_is_unavailable(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(jaxlinarg_operator.pallas_gpu, "pl", None)

    assert jaxlinarg_operator.resolve_backend(Backend.AUTO) is Backend.PURE_JAX


def test_explicit_pallas_gpu_rejects_gpu_platform_when_pallas_import_is_unavailable(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.pallas_gpu, "pl", None)

    with pytest.raises(
        ValueError,
        match="PALLAS_GPU backend is unavailable.*platform 'gpu'.*Pallas import",
    ):
        jaxlinarg_operator.resolve_backend(Backend.PALLAS_GPU, platform="gpu")


def test_explicit_pallas_tpu_rejects_tpu_platform_when_pallas_import_is_unavailable(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.pallas_tpu, "pl", None)

    with pytest.raises(
        ValueError,
        match="PALLAS_TPU backend is unavailable.*platform 'tpu'.*Pallas import",
    ):
        jaxlinarg_operator.resolve_backend(Backend.PALLAS_TPU, platform="tpu")


def test_explicit_ffi_cpu_resolves_to_pure_jax_when_handler_is_absent(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)

    with pytest.warns(UserWarning, match="FFI_CPU backend is unavailable"):
        backend = jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu")

    assert backend is Backend.PURE_JAX


def test_explicit_ffi_cpu_resolves_to_ffi_cpu_when_handler_is_available(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: True)

    assert jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu") is Backend.FFI_CPU


@pytest.mark.parametrize("platform", ["cpu", "tpu"])
def test_explicit_pallas_gpu_rejects_non_gpu_platforms(platform):
    with pytest.raises(ValueError, match=rf"PALLAS_GPU backend is unavailable.*{platform}"):
        jaxlinarg_operator.resolve_backend(Backend.PALLAS_GPU, platform=platform)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_explicit_pallas_tpu_rejects_non_tpu_platforms(platform, monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.pallas_tpu, "pl", object())

    with pytest.raises(ValueError, match=rf"PALLAS_TPU backend is unavailable.*{platform}"):
        jaxlinarg_operator.resolve_backend(Backend.PALLAS_TPU, platform=platform)


def test_explicit_pallas_tpu_resolves_on_tpu_when_pallas_imports(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.pallas_tpu, "pl", object())

    assert jaxlinarg_operator.resolve_backend(Backend.PALLAS_TPU, platform="tpu") is Backend.PALLAS_TPU


def test_pallas_tpu_solve_dispatch_uses_tpu_kernel(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(jaxlinarg_operator.pallas_tpu, "pl", object())
    calls = []

    def fake_solve_forward(*args):
        calls.append(args)
        return args[-1] + 1

    monkeypatch.setitem(jaxlinarg_operator._SOLVERS, ("forward", Backend.PALLAS_TPU), fake_solve_forward)
    op = JaxLinearARG(
        **_minimal_operator_kwargs(),
        backend=Backend.PALLAS_TPU,
        n_nonunique_indices=2,
        min_index_to_keep=1,
    )

    actual = op.matmat(np.ones((1, 1), dtype=np.float32))

    assert calls
    np.testing.assert_allclose(np.asarray(actual), np.array([[1.0]], dtype=np.float32))


def test_ffi_availability_returns_false_when_registrations_raises(monkeypatch):
    module_name = "linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl"
    fake_impl = types.SimpleNamespace(registrations=lambda: (_ for _ in ()).throw(RuntimeError("registrations failed")))
    monkeypatch.setitem(sys.modules, module_name, fake_impl)
    monkeypatch.setattr(kernels_pkg, "_ffi_cpu_impl", fake_impl, raising=False)
    jaxlinarg_operator.ffi_cpu._load_ffi_cpu_impl.cache_clear()
    jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert not jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available()
    assert "registrations failed" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_error())


def test_ffi_availability_returns_false_when_target_registration_raises(monkeypatch):
    module_name = "linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl"
    fake_impl = types.SimpleNamespace(registrations=lambda: {"target": object()})
    monkeypatch.setitem(sys.modules, module_name, fake_impl)
    monkeypatch.setattr(kernels_pkg, "_ffi_cpu_impl", fake_impl, raising=False)
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu.jax.ffi,
        "register_ffi_target",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )
    jaxlinarg_operator.ffi_cpu._load_ffi_cpu_impl.cache_clear()
    jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert not jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available()
    assert "registration failed" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_error())


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("registrations failed"),
        RuntimeError("registration failed"),
    ],
)
def test_auto_backend_treats_ffi_registration_failure_as_unavailable(monkeypatch, failure):
    jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available.cache_clear()
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "_load_ffi_cpu_impl",
        lambda: (_ for _ in ()).throw(failure),
    )

    assert jaxlinarg_operator.resolve_backend(Backend.AUTO, platform="cpu") is Backend.PURE_JAX


def test_explicit_ffi_cpu_warning_reports_registration_failure(monkeypatch):
    jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available.cache_clear()
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "_load_ffi_cpu_impl",
        lambda: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )

    with pytest.warns(UserWarning, match="registration failed"):
        backend = jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu")

    assert backend is Backend.PURE_JAX


def test_invalid_backend_value_fails_at_operator_construction():
    kwargs = _minimal_operator_kwargs()
    kwargs["backend"] = "not-a-backend"

    with pytest.raises(ValueError, match="not-a-backend"):
        JaxLinearARG.from_lineararg_arrays(**kwargs)
