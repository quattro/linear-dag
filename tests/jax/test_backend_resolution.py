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
    _clear_ffi_cpu_caches()
    yield
    _clear_ffi_cpu_caches()


def _clear_ffi_cpu_caches() -> None:
    ffi = jaxlinarg_operator.ffi_cpu
    for function in (
        ffi._import_ffi_cpu_impl,
        ffi._load_ffi_cpu_impl,
        ffi.is_ffi_cpu_available,
        ffi.is_ffi_cpu_packed_available,
    ):
        cache_clear = getattr(function, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@pytest.mark.parametrize(
    ("platform", "exact_available", "packed_available", "require_packed_targets", "expected"),
    [
        ("cpu", True, False, False, Backend.FFI_CPU),
        ("cpu", True, False, True, Backend.PURE_JAX),
        ("cpu", False, True, False, Backend.PURE_JAX),
        ("cpu", False, True, True, Backend.FFI_CPU),
        ("gpu", True, True, False, Backend.PURE_JAX),
        ("tpu", True, True, True, Backend.PURE_JAX),
    ],
)
def test_auto_backend_resolves_by_platform_and_representation_availability(
    monkeypatch,
    platform,
    exact_available,
    packed_available,
    require_packed_targets,
    expected,
):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: exact_available)
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_packed_available", lambda: packed_available)

    assert (
        jaxlinarg_operator.resolve_backend(
            Backend.AUTO,
            platform=platform,
            require_packed_targets=require_packed_targets,
        )
        is expected
    )


def test_explicit_ffi_cpu_fails_with_exact_reason_and_portable_guidance(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: False)
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "last_ffi_cpu_error",
        lambda: RuntimeError("missing exact target"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"exact.*missing exact target.*Backend\.AUTO.*Backend\.PURE_JAX",
    ):
        jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu")


def test_explicit_ffi_cpu_resolves_to_ffi_cpu_when_handler_is_available(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", lambda: True)

    assert jaxlinarg_operator.resolve_backend(Backend.FFI_CPU, platform="cpu") is Backend.FFI_CPU


@pytest.mark.parametrize("platform", ["gpu", "tpu"])
@pytest.mark.parametrize("require_packed_targets", [False, True])
def test_explicit_ffi_cpu_fails_on_non_cpu_before_capability_probe(
    monkeypatch,
    platform,
    require_packed_targets,
):
    def fail_probe() -> bool:
        raise AssertionError("non-CPU requests must fail before probing CPU targets")

    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_available", fail_probe)
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_packed_available", fail_probe)

    representation = "packed" if require_packed_targets else "exact"
    with pytest.raises(
        RuntimeError,
        match=rf"{representation}.*{platform}.*Backend\.AUTO.*Backend\.PURE_JAX",
    ):
        jaxlinarg_operator.resolve_backend(
            Backend.FFI_CPU,
            platform=platform,
            require_packed_targets=require_packed_targets,
        )


def test_ffi_target_capabilities_report_source_only_install_independently(monkeypatch):
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "_import_ffi_cpu_impl",
        lambda: (_ for _ in ()).throw(ImportError("native extension is not installed")),
    )
    _clear_ffi_cpu_caches()

    assert not jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available()
    assert not jaxlinarg_operator.ffi_cpu.is_ffi_cpu_packed_available()
    assert "exact" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_error())
    assert "native extension is not installed" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_error())
    assert "packed" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_packed_error())


def test_ffi_target_capabilities_return_false_when_registrations_raises(monkeypatch):
    module_name = "linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl"
    fake_impl = types.SimpleNamespace(registrations=lambda: (_ for _ in ()).throw(RuntimeError("registrations failed")))
    monkeypatch.setitem(sys.modules, module_name, fake_impl)
    monkeypatch.setattr(kernels_pkg, "_ffi_cpu_impl", fake_impl, raising=False)
    _clear_ffi_cpu_caches()

    assert not jaxlinarg_operator.ffi_cpu.is_ffi_cpu_available()
    assert not jaxlinarg_operator.ffi_cpu.is_ffi_cpu_packed_available()
    assert "registrations failed" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_error())
    assert "registrations failed" in str(jaxlinarg_operator.ffi_cpu.last_ffi_cpu_packed_error())


@pytest.mark.parametrize(
    ("missing_target", "exact_available", "packed_available"),
    [
        (None, True, True),
        (jaxlinarg_operator.ffi_cpu.FFI_CPU_PACKED_SOLVE_BACKWARD_F64, True, False),
        (jaxlinarg_operator.ffi_cpu.FFI_CPU_SOLVE_BACKWARD_F64, False, True),
    ],
)
def test_ffi_target_sets_are_complete_and_independent(
    monkeypatch,
    missing_target,
    exact_available,
    packed_available,
):
    ffi = jaxlinarg_operator.ffi_cpu
    registrations = {
        name: object() for name in (*ffi.FFI_CPU_EXACT_TARGETS, *ffi.FFI_CPU_PACKED_TARGETS) if name != missing_target
    }
    fake_impl = types.SimpleNamespace(registrations=lambda: registrations)
    monkeypatch.setattr(ffi, "_import_ffi_cpu_impl", lambda: fake_impl)
    monkeypatch.setattr(ffi.jax.ffi, "register_ffi_target", lambda *args, **kwargs: None)
    _clear_ffi_cpu_caches()

    assert ffi.is_ffi_cpu_available() is exact_available
    assert ffi.is_ffi_cpu_packed_available() is packed_available
    if missing_target in ffi.FFI_CPU_EXACT_TARGETS:
        assert missing_target in str(ffi.last_ffi_cpu_error())
        assert ffi.last_ffi_cpu_packed_error() is None
    elif missing_target in ffi.FFI_CPU_PACKED_TARGETS:
        assert ffi.last_ffi_cpu_error() is None
        assert missing_target in str(ffi.last_ffi_cpu_packed_error())


@pytest.mark.parametrize("failing_representation", ["exact", "packed"])
def test_registration_failure_is_atomic_per_target_set(monkeypatch, failing_representation):
    ffi = jaxlinarg_operator.ffi_cpu
    registrations = {name: object() for name in (*ffi.FFI_CPU_EXACT_TARGETS, *ffi.FFI_CPU_PACKED_TARGETS)}
    failing_target = (
        ffi.FFI_CPU_SOLVE_FORWARD_F32 if failing_representation == "exact" else ffi.FFI_CPU_PACKED_SOLVE_FORWARD_F32
    )

    def register(name, *args, **kwargs):
        del args, kwargs
        if name == failing_target:
            raise RuntimeError(f"cannot register {failing_target}")

    monkeypatch.setattr(ffi, "_import_ffi_cpu_impl", lambda: types.SimpleNamespace(registrations=lambda: registrations))
    monkeypatch.setattr(ffi.jax.ffi, "register_ffi_target", register)
    _clear_ffi_cpu_caches()

    assert ffi.is_ffi_cpu_available() is (failing_representation != "exact")
    assert ffi.is_ffi_cpu_packed_available() is (failing_representation != "packed")


def test_explicit_packed_ffi_failure_reports_packed_registration_reason(monkeypatch):
    monkeypatch.setattr(jaxlinarg_operator.ffi_cpu, "is_ffi_cpu_packed_available", lambda: False)
    monkeypatch.setattr(
        jaxlinarg_operator.ffi_cpu,
        "last_ffi_cpu_packed_error",
        lambda: RuntimeError("packed registration failed"),
    )

    with pytest.raises(RuntimeError, match="packed.*packed registration failed"):
        jaxlinarg_operator.resolve_backend(
            Backend.FFI_CPU,
            platform="cpu",
            require_packed_targets=True,
        )


def test_invalid_backend_value_fails_at_operator_construction():
    kwargs = _minimal_operator_kwargs()
    kwargs["backend"] = "not-a-backend"

    with pytest.raises(ValueError, match="not-a-backend"):
        JaxLinearARG.from_lineararg_arrays(**kwargs)
