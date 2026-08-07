# pattern: Functional Core

from __future__ import annotations

import csv
import io
import zipfile

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
    source_artifact_dir = tmp_path / "src" / "linear_dag" / "core" / "jaxlinarg" / "kernels"
    build_artifact_dir = tmp_path / "build" / "linear_dag" / "core" / "jaxlinarg" / "kernels"
    source_artifact_dir.mkdir(parents=True)
    build_artifact_dir.mkdir(parents=True)
    source_expected = source_artifact_dir / "_ffi_cpu_impl.cp312-win_amd64.pyd"
    build_expected = build_artifact_dir / "_ffi_cpu_impl.cp312-win_amd64.pyd"
    source_expected.touch()
    build_expected.touch()
    (source_artifact_dir / "_ffi_cpu_impl.txt").touch()

    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda name: ".pyd")
    monkeypatch.setattr(hatch_build.importlib.machinery, "EXTENSION_SUFFIXES", [".pyd"])

    assert hatch_build.ffi_cpu_extension_artifacts(tmp_path) == [
        str(build_expected),
        str(source_expected),
    ]


def test_ffi_cpu_source_extension_artifacts_exclude_build_cache(
    monkeypatch,
    tmp_path,
):
    source_artifact_dir = tmp_path / "src" / "linear_dag" / "core" / "jaxlinarg" / "kernels"
    build_artifact_dir = tmp_path / "build" / "linear_dag" / "core" / "jaxlinarg" / "kernels"
    source_artifact_dir.mkdir(parents=True)
    build_artifact_dir.mkdir(parents=True)
    source_expected = source_artifact_dir / "_ffi_cpu_impl.cp312-win_amd64.pyd"
    build_expected = build_artifact_dir / "_ffi_cpu_impl.cp312-win_amd64.pyd"
    source_expected.touch()
    build_expected.touch()

    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda name: ".pyd")
    monkeypatch.setattr(hatch_build.importlib.machinery, "EXTENSION_SUFFIXES", [".pyd"])

    assert hatch_build.ffi_cpu_source_extension_artifacts(tmp_path) == [str(source_expected)]


def test_macos_sdk_cxx_include_dirs_uses_sdkroot(monkeypatch, tmp_path):
    sdk_root = tmp_path / "MacOSX.sdk"
    include_dir = sdk_root / "usr" / "include" / "c++" / "v1"
    include_dir.mkdir(parents=True)

    monkeypatch.setattr(hatch_build.sys, "platform", "darwin")
    monkeypatch.setenv("SDKROOT", str(sdk_root))
    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda name: None)

    assert hatch_build._macos_sdk_cxx_include_dirs() == [str(include_dir)]


def test_sanitize_macos_linker_flags_deduplicates_sysconfig_rpaths(monkeypatch):
    config = {
        "LDSHARED": "clang -bundle -Wl,-rpath,/env/lib -L/env/lib -Wl,-rpath,/env/lib",
        "BLDSHARED": "",
        "LDCXXSHARED": "",
        "LDFLAGS": "-Wl,-rpath,/env/lib -L/env/lib -Wl,-rpath,/env/lib",
        "PY_LDFLAGS": "",
    }

    monkeypatch.setattr(hatch_build.sys, "platform", "darwin")
    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda key: config.get(key))
    monkeypatch.setattr(hatch_build.sysconfig, "get_config_vars", lambda: config)
    monkeypatch.setenv("LDFLAGS", "-Wl,-pie -Wl,-rpath,/env/lib -L/env/lib")

    hatch_build.sanitize_macos_linker_flags()

    assert config["LDSHARED"].count("-Wl,-rpath,/env/lib") == 1
    assert config["LDFLAGS"].count("-Wl,-rpath,/env/lib") == 1
    assert hatch_build.os.environ["LDSHARED"].count("-Wl,-rpath,/env/lib") == 1
    assert "-Wl,-rpath,/env/lib" not in hatch_build.os.environ["LDFLAGS"]
    assert "-Wl,-pie" in hatch_build.os.environ["LDFLAGS"]


def test_sanitize_macos_linker_flags_preserves_distinct_env_rpaths(monkeypatch):
    config = {
        "LDSHARED": "clang -bundle -Wl,-rpath,/env/lib",
        "BLDSHARED": "",
        "LDCXXSHARED": "",
        "LDFLAGS": "",
        "PY_LDFLAGS": "",
    }

    monkeypatch.setattr(hatch_build.sys, "platform", "darwin")
    monkeypatch.setattr(hatch_build.sysconfig, "get_config_var", lambda key: config.get(key))
    monkeypatch.setattr(hatch_build.sysconfig, "get_config_vars", lambda: config)
    monkeypatch.setenv("LDFLAGS", "-Wl,-rpath,/other/lib -L/other/lib")

    hatch_build.sanitize_macos_linker_flags()

    assert "-Wl,-rpath,/other/lib" in hatch_build.os.environ["LDFLAGS"]


def test_delete_duplicate_macos_rpaths_removes_only_repeated_entries(monkeypatch, tmp_path):
    artifact = tmp_path / "extension.so"
    artifact.touch()
    calls = []

    monkeypatch.setattr(
        hatch_build,
        "_macos_rpaths",
        lambda path: ["/env/lib", "/other/lib", "/env/lib"],
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(hatch_build.subprocess, "run", fake_run)

    hatch_build._delete_duplicate_macos_rpaths(artifact)

    assert calls == [
        (
            ["install_name_tool", "-delete_rpath", "/env/lib", str(artifact)],
            {"check": True},
        )
    ]


def test_repair_macos_wheel_rpaths_updates_wheel_payload_and_record(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    package_dir = source_root / "linear_dag" / "core"
    dist_info = source_root / "linear_dag-0.0.dist-info"
    package_dir.mkdir(parents=True)
    dist_info.mkdir()
    extension = package_dir / "brick_graph.so"
    extension.write_bytes(b"duplicate-rpath-payload")
    (dist_info / "METADATA").write_text("Name: linear-dag\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")

    wheel_path = tmp_path / "linear_dag-0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
            wheel.write(path, path.relative_to(source_root).as_posix())

    def fake_delete_duplicate_rpaths(path):
        if path.name == "brick_graph.so":
            path.write_bytes(b"repaired")

    monkeypatch.setattr(hatch_build.sys, "platform", "darwin")
    monkeypatch.setattr(hatch_build, "_delete_duplicate_macos_rpaths", fake_delete_duplicate_rpaths)

    hatch_build.repair_macos_wheel_rpaths(wheel_path)

    with zipfile.ZipFile(wheel_path) as wheel:
        assert wheel.read("linear_dag/core/brick_graph.so") == b"repaired"
        record = wheel.read("linear_dag-0.0.dist-info/RECORD").decode("utf-8")

    rows = {row[0]: row for row in csv.reader(io.StringIO(record))}
    extension_row = rows["linear_dag/core/brick_graph.so"]
    record_row = rows["linear_dag-0.0.dist-info/RECORD"]
    assert extension_row[1].startswith("sha256=")
    assert extension_row[2] == str(len(b"repaired"))
    assert record_row[1:] == ["", ""]
