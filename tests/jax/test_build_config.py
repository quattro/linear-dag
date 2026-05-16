# pattern: Imperative Shell

from __future__ import annotations

import types

import linear_dag

from linear_dag.core.jaxlinarg import build_config


def test_show_build_config_reports_runtime_and_ffi_metadata(monkeypatch):
    error = RuntimeError("registration failed")

    monkeypatch.setattr(build_config.jax, "__version__", "0.test")
    monkeypatch.setattr(build_config.jaxlib, "__version__", "0.testlib")
    monkeypatch.setattr(build_config.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(build_config.ffi_cpu, "is_ffi_cpu_built", lambda: True)
    monkeypatch.setattr(build_config.ffi_cpu, "is_ffi_cpu_available", lambda: False)
    monkeypatch.setattr(build_config.ffi_cpu, "is_ffi_cpu_blas_enabled", lambda: True)
    monkeypatch.setattr(build_config.ffi_cpu, "ffi_cpu_blas_backend", lambda: "openblas")
    monkeypatch.setattr(build_config.ffi_cpu, "is_ffi_cpu_native_tuning_enabled", lambda: True)
    monkeypatch.setattr(build_config.ffi_cpu, "last_ffi_cpu_error", lambda: error)

    assert build_config.show_build_config() == {
        "jax": "0.test",
        "jaxlib": "0.testlib",
        "backend": "cpu",
        "ffi_cpu_built": True,
        "ffi_cpu_available": False,
        "ffi_cpu_blas_enabled": True,
        "ffi_cpu_blas_backend": "openblas",
        "ffi_cpu_native_tuning": True,
        "ffi_cpu_error": "registration failed",
    }


def test_show_build_config_is_exported_from_public_package_api():
    assert linear_dag.show_build_config is build_config.show_build_config


def test_ffi_cpu_built_is_distinct_from_available(monkeypatch):
    fake_impl = types.SimpleNamespace(
        registrations=lambda: (_ for _ in ()).throw(RuntimeError("registration failed")),
        blas_enabled=lambda: False,
        blas_backend=lambda: "none",
        native_tuning_enabled=lambda: False,
    )
    monkeypatch.setattr(build_config.ffi_cpu, "_import_ffi_cpu_impl", lambda: fake_impl)
    build_config.ffi_cpu._load_ffi_cpu_impl.cache_clear()
    build_config.ffi_cpu.is_ffi_cpu_available.cache_clear()

    assert build_config.ffi_cpu.is_ffi_cpu_built()
    assert not build_config.ffi_cpu.is_ffi_cpu_available()
    assert "registration failed" in str(build_config.ffi_cpu.last_ffi_cpu_error())
