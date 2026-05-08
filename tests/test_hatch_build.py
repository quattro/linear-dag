# pattern: Functional Core

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hatch_build


def test_custom_build_hook_warns_and_continues_when_optional_native_build_fails(
    monkeypatch,
    tmp_path,
):
    def fail_build(root):
        raise RuntimeError(f"cannot compile extension under {root}")

    monkeypatch.delenv("LINEAR_DAG_REQUIRE_FFI_CPU", raising=False)
    monkeypatch.setattr(hatch_build, "build_ffi_cpu_extension", fail_build)
    monkeypatch.setattr(hatch_build, "ffi_cpu_extension_artifacts", lambda root: [])
    build_data = {"artifacts": [], "force_include": {}}

    with pytest.warns(RuntimeWarning, match="Could not build optional CPU FFI extension"):
        hatch_build.CustomBuildHook.initialize(
            SimpleNamespace(target_name="wheel", root=str(tmp_path)),
            "0.0.0",
            build_data,
        )

    assert build_data["artifacts"] == []
    assert build_data["force_include"] == {}


def test_custom_build_hook_does_not_include_stale_artifact_when_optional_build_fails(
    monkeypatch,
    tmp_path,
):
    def fail_build(root):
        raise RuntimeError(f"cannot compile extension under {root}")

    artifact_dir = tmp_path / "src" / "linear_dag" / "core" / "jaxlinarg" / "kernels"
    artifact_dir.mkdir(parents=True)
    stale_artifact = artifact_dir / "_ffi_cpu_impl.so"
    stale_artifact.touch()

    monkeypatch.delenv("LINEAR_DAG_REQUIRE_FFI_CPU", raising=False)
    monkeypatch.setattr(hatch_build, "build_ffi_cpu_extension", fail_build)
    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda name: ".so")
    monkeypatch.setattr(hatch_build.importlib.machinery, "EXTENSION_SUFFIXES", [".so"])
    build_data = {"artifacts": [], "force_include": {}}

    with pytest.warns(RuntimeWarning, match="Could not build optional CPU FFI extension"):
        hatch_build.CustomBuildHook.initialize(
            SimpleNamespace(target_name="wheel", root=str(tmp_path)),
            "0.0.0",
            build_data,
        )

    assert build_data["artifacts"] == []
    assert build_data["force_include"] == {}
    assert not stale_artifact.exists()


def test_custom_build_hook_reraises_native_build_failure_when_required(
    monkeypatch,
    tmp_path,
):
    def fail_build(root):
        raise RuntimeError(f"cannot compile extension under {root}")

    monkeypatch.setenv("LINEAR_DAG_REQUIRE_FFI_CPU", "1")
    monkeypatch.setattr(hatch_build, "build_ffi_cpu_extension", fail_build)
    build_data = {"artifacts": [], "force_include": {}}

    with pytest.raises(RuntimeError, match="cannot compile extension"):
        hatch_build.CustomBuildHook.initialize(
            SimpleNamespace(target_name="wheel", root=str(tmp_path)),
            "0.0.0",
            build_data,
        )


def test_ffi_cpu_extension_artifacts_use_platform_extension_suffix(
    monkeypatch,
    tmp_path,
):
    artifact_dir = tmp_path / "src" / "linear_dag" / "core" / "jaxlinarg" / "kernels"
    artifact_dir.mkdir(parents=True)
    expected = artifact_dir / "_ffi_cpu_impl.cp312-win_amd64.pyd"
    expected.touch()
    (artifact_dir / "_ffi_cpu_impl.txt").touch()

    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda name: ".pyd")
    monkeypatch.setattr(hatch_build.importlib.machinery, "EXTENSION_SUFFIXES", [".pyd"])

    assert hatch_build.ffi_cpu_extension_artifacts(tmp_path) == [str(expected)]
