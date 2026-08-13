# pattern: Imperative Shell

import base64
import csv
import glob
import hashlib
import importlib.machinery
import io
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import warnings
import zipfile

from pathlib import Path

import jax
import numpy as np
import scipy

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

FFI_CPU_BUILD_REQUIRED_ENV = "LINEAR_DAG_REQUIRE_FFI_CPU"
FFI_CPU_BLAS_DISABLED_ENV = "LINEAR_DAG_DISABLE_FFI_CPU_BLAS"
FFI_CPU_BLAS_REQUIRED_ENV = "LINEAR_DAG_REQUIRE_FFI_CPU_BLAS"
FFI_CPU_NATIVE_ENV = "LINEAR_DAG_FFI_CPU_NATIVE"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        sanitize_macos_linker_flags()
        build_data["include-dirs"] = [os.path.dirname(scipy.__file__)]
        if self.target_name != "sdist":
            remove_incompatible_cython_extension_artifacts(self.root)
            if build_ffi_cpu_extension_or_warn(self.root):
                for artifact in ffi_cpu_source_extension_artifacts(self.root):
                    relative_artifact = os.path.relpath(artifact, self.root)
                    build_data.setdefault("artifacts", []).append(f"/{relative_artifact}")
                    build_data.setdefault("force_include", {})[relative_artifact] = relative_artifact

    def finalize(self, version, build_data, artifact_path):
        del version, build_data
        repair_macos_extension_rpaths(self.root)
        repair_macos_wheel_rpaths(artifact_path)


def get_include_dirs():
    return [os.path.dirname(scipy.__file__)]


def build_hook(config):
    sanitize_macos_linker_flags()
    config["include-dirs"] = get_include_dirs()


def build_ffi_cpu_extension_or_warn(root):
    remove_ffi_cpu_extension_artifacts(root)
    try:
        build_ffi_cpu_extension_with_optional_blas_fallback(root)
    except Exception as exc:
        if is_ffi_cpu_build_required():
            raise
        warnings.warn(
            "Could not build optional CPU FFI extension; continuing with PURE_JAX "
            f"fallback. Set {FFI_CPU_BUILD_REQUIRED_ENV}=1 to make this fatal. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def build_ffi_cpu_extension_with_optional_blas_fallback(root):
    try:
        build_ffi_cpu_extension(root)
    except Exception:
        if _truthy_env(os.environ.get(FFI_CPU_BLAS_DISABLED_ENV)) or _truthy_env(
            os.environ.get(FFI_CPU_BLAS_REQUIRED_ENV)
        ):
            raise
        with _temporary_env(FFI_CPU_BLAS_DISABLED_ENV, "1"):
            remove_ffi_cpu_extension_artifacts(root)
            build_ffi_cpu_extension(root)


def remove_ffi_cpu_extension_artifacts(root):
    for artifact in ffi_cpu_extension_artifacts(root):
        Path(artifact).unlink()


def remove_incompatible_cython_extension_artifacts(
    root,
    current_extension_suffix=None,
):
    """Remove in-source Cython binaries built for a different Python ABI."""
    if current_extension_suffix is None:
        current_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not current_extension_suffix:
        return

    source_root = Path(root) / "src" / "linear_dag"
    for pyx_path in source_root.rglob("*.pyx"):
        module_stem = pyx_path.stem
        for compiled_suffix in (".so", ".pyd", ".dll"):
            for artifact in pyx_path.parent.glob(f"{module_stem}*{compiled_suffix}"):
                # A module's extension begins with `<stem>.`; avoid touching a
                # neighboring module whose name merely shares this prefix.
                if not artifact.name.startswith(f"{module_stem}."):
                    continue
                if not artifact.name.endswith(current_extension_suffix):
                    artifact.unlink()


def is_ffi_cpu_build_required():
    return _truthy_env(os.environ.get(FFI_CPU_BUILD_REQUIRED_ENV))


def _ffi_cpu_blas_options():
    empty_options = {
        "define_macros": [],
        "include_dirs": [],
        "library_dirs": [],
        "libraries": [],
        "extra_link_args": [],
        "blas_backend": "none",
    }
    if _truthy_env(os.environ.get(FFI_CPU_BLAS_DISABLED_ENV)):
        return empty_options
    if sys.platform == "darwin":
        return {
            **empty_options,
            "define_macros": [("LINEAR_DAG_HAVE_CBLAS", "1")],
            "extra_link_args": ["-framework", "Accelerate"],
            "blas_backend": "accelerate",
        }

    include_dir = _find_header_dir("cblas.h", _blas_include_dir_candidates())
    library_name, library_dir = _find_blas_library(_blas_library_dir_candidates())
    if include_dir is not None and library_name is not None:
        return {
            **empty_options,
            "define_macros": [("LINEAR_DAG_HAVE_CBLAS", "1")],
            "include_dirs": [str(include_dir)],
            "library_dirs": [str(library_dir)] if library_dir is not None else [],
            "libraries": [library_name],
            "blas_backend": library_name,
        }

    if _truthy_env(os.environ.get(FFI_CPU_BLAS_REQUIRED_ENV)):
        raise RuntimeError(
            "Could not find cblas.h and a BLAS library for the CPU FFI extension. "
            f"Set {FFI_CPU_BLAS_DISABLED_ENV}=1 to build the scalar fallback instead."
        )
    return empty_options


def _macos_sdk_cxx_include_dirs():
    if sys.platform != "darwin":
        return []

    sdk_root = os.environ.get("SDKROOT") or sysconfig.get_config_var("SDKROOT")
    if not sdk_root:
        return []

    include_dir = Path(sdk_root) / "usr" / "include" / "c++" / "v1"
    return [str(include_dir)] if include_dir.is_dir() else []


def build_ffi_cpu_extension(root):
    sanitize_macos_linker_flags()
    root = Path(root)
    blas_options = _ffi_cpu_blas_options()
    extension = Extension(
        "linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl",
        sources=[str(root / "src/linear_dag/core/jaxlinarg/kernels/ffi_cpu_impl.cc")],
        include_dirs=[
            *_macos_sdk_cxx_include_dirs(),
            jax.ffi.include_dir(),
            np.get_include(),
            os.path.dirname(scipy.__file__),
            *blas_options["include_dirs"],
        ],
        define_macros=[
            *blas_options["define_macros"],
            ("LINEAR_DAG_FFI_CPU_BLAS_BACKEND", f'"{blas_options["blas_backend"]}"'),
            ("LINEAR_DAG_FFI_CPU_NATIVE_TUNING", "1" if _truthy_env(os.environ.get(FFI_CPU_NATIVE_ENV)) else "0"),
        ],
        library_dirs=blas_options["library_dirs"],
        libraries=blas_options["libraries"],
        language="c++",
        extra_compile_args=_cxx_compile_args(),
        extra_link_args=blas_options["extra_link_args"],
    )
    distribution = Distribution(
        {
            "name": "linear-dag",
            "ext_modules": [extension],
            "package_dir": {"": "src"},
        }
    )
    command = build_ext(distribution)
    command.inplace = True
    command.force = True
    command.build_lib = str(root / "build")
    command.build_temp = str(root / "build/temp")
    command.ensure_finalized()
    command.run()


def ffi_cpu_extension_artifacts(root):
    return sorted(
        {
            *ffi_cpu_source_extension_artifacts(root),
            *ffi_cpu_build_cache_extension_artifacts(root),
        }
    )


def ffi_cpu_source_extension_artifacts(root):
    root = Path(root)
    return _extension_artifacts_for_base(
        root / "src" / "linear_dag" / "core" / "jaxlinarg" / "kernels" / "_ffi_cpu_impl"
    )


def ffi_cpu_build_cache_extension_artifacts(root):
    root = Path(root)
    return _extension_artifacts_for_base(
        root / "build" / "linear_dag" / "core" / "jaxlinarg" / "kernels" / "_ffi_cpu_impl"
    )


def _extension_artifacts_for_base(base):
    artifacts = []
    for suffix in _extension_suffixes():
        artifacts.extend(glob.glob(f"{base}*{suffix}"))
    return sorted(set(artifacts))


def repair_macos_extension_rpaths(root):
    if sys.platform != "darwin":
        return

    root = Path(root)
    for artifact in sorted(root.glob("src/linear_dag/**/*.so")):
        _delete_duplicate_macos_rpaths(artifact)


def repair_macos_wheel_rpaths(artifact_path):
    if sys.platform != "darwin":
        return

    artifact_path = Path(artifact_path)
    if artifact_path.suffix != ".whl" or not artifact_path.exists():
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        extracted_root = Path(temp_dir) / "wheel"
        with zipfile.ZipFile(artifact_path) as wheel:
            wheel.extractall(extracted_root)

        for artifact in sorted(extracted_root.glob("linear_dag/**/*.so")):
            _delete_duplicate_macos_rpaths(artifact)

        _rewrite_wheel_record(extracted_root)
        repaired_wheel = artifact_path.with_name(f"{artifact_path.name}.repaired")
        _write_wheel_archive(extracted_root, repaired_wheel)
        os.replace(repaired_wheel, artifact_path)


def _rewrite_wheel_record(root):
    record_paths = sorted(root.glob("*.dist-info/RECORD"))
    if not record_paths:
        return

    record_path = record_paths[0]
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        if path == record_path:
            rows.append([relative_path, "", ""])
            continue
        payload = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
        rows.append([relative_path, f"sha256={digest}", str(len(payload))])

    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    record_path.write_text(output.getvalue(), encoding="utf-8")


def _write_wheel_archive(root, artifact_path):
    with zipfile.ZipFile(artifact_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            wheel.write(path, path.relative_to(root).as_posix())


def _delete_duplicate_macos_rpaths(artifact):
    rpaths = _macos_rpaths(artifact)
    seen = set()
    for rpath in rpaths:
        if rpath not in seen:
            seen.add(rpath)
            continue
        subprocess.run(
            ["install_name_tool", "-delete_rpath", rpath, str(artifact)],
            check=True,
        )


def _macos_rpaths(artifact):
    output = subprocess.run(
        ["otool", "-l", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rpaths = []
    in_rpath_command = False
    for line in output:
        stripped = line.strip()
        if stripped == "cmd LC_RPATH":
            in_rpath_command = True
            continue
        if in_rpath_command and stripped.startswith("path "):
            rpaths.append(stripped.split()[1])
            in_rpath_command = False
    return rpaths


def _extension_suffixes():
    suffixes = [
        sysconfig.get_config_var("EXT_SUFFIX"),
        *importlib.machinery.EXTENSION_SUFFIXES,
    ]
    return tuple(dict.fromkeys(suffix for suffix in suffixes if suffix))


def _truthy_env(value):
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


class _temporary_env:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.previous = None

    def __enter__(self):
        self.previous = os.environ.get(self.key)
        os.environ[self.key] = self.value

    def __exit__(self, exc_type, exc, tb):
        if self.previous is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.previous
        return False


def _blas_include_dir_candidates():
    env_dirs = [os.environ.get("BLAS_INCLUDE_DIR"), os.environ.get("OPENBLAS_INCLUDE_DIR")]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    candidates = [Path(path) for path in env_dirs if path]
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "include")
    candidates.extend([Path("/usr/include"), Path("/usr/local/include"), Path("/opt/homebrew/include")])
    return _existing_dirs(candidates)


def _blas_library_dir_candidates():
    env_dirs = [os.environ.get("BLAS_LIBRARY_DIR"), os.environ.get("OPENBLAS_LIBRARY_DIR")]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    candidates = [Path(path) for path in env_dirs if path]
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib")
    candidates.extend([Path("/usr/lib"), Path("/usr/local/lib"), Path("/opt/homebrew/lib")])
    return _existing_dirs(candidates)


def _existing_dirs(paths):
    return [path for path in dict.fromkeys(paths) if path.is_dir()]


def _find_header_dir(header, candidates):
    for directory in candidates:
        if (directory / header).exists():
            return directory
    return None


def _find_blas_library(candidates):
    for name in ("openblas", "blas"):
        for directory in candidates:
            patterns = (
                f"lib{name}.so",
                f"lib{name}.dylib",
                f"lib{name}.a",
            )
            if any(directory.glob(pattern) for pattern in patterns):
                return name, directory
    return None, None


def _cxx_compile_args():
    args = ["-std=c++17", "-O3"]
    if sys.platform == "darwin":
        args.append("-stdlib=libc++")
    if _truthy_env(os.environ.get(FFI_CPU_NATIVE_ENV)):
        # Native CPU tuning is intentionally opt-in: it is useful for local
        # benchmark builds, but the resulting extension may not be portable.
        args.append("-mcpu=native" if sys.platform == "darwin" else "-march=native")
    return args


def sanitize_macos_linker_flags():
    """Remove duplicate Darwin rpath flags inherited from conda-style envs.

    Some macOS Python distributions expose the same `-Wl,-rpath,<prefix>/lib`
    value through both `sysconfig` and environment `LDFLAGS`. If extension
    builders concatenate those sources, the resulting Mach-O can contain
    duplicate `LC_RPATH` load commands, which modern `dlopen` rejects.
    """
    if sys.platform != "darwin":
        return

    config_keys = ("LDSHARED", "BLDSHARED", "LDCXXSHARED", "LDFLAGS", "PY_LDFLAGS")
    config_rpaths = set()
    for key in config_keys:
        value = sysconfig.get_config_var(key)
        if not isinstance(value, str) or not value:
            continue
        deduped, rpaths = _dedupe_rpath_flags(value)
        _set_sysconfig_var(key, deduped)
        if key in {"LDSHARED", "BLDSHARED", "LDCXXSHARED"}:
            os.environ[key] = deduped
        config_rpaths.update(rpaths)

    env_value = os.environ.get("LDFLAGS")
    if env_value:
        os.environ["LDFLAGS"] = _drop_known_rpath_flags(env_value, config_rpaths)


def _dedupe_rpath_flags(flags):
    tokens = shlex.split(flags)
    result = []
    seen_rpaths = set()
    i = 0
    while i < len(tokens):
        token = tokens[i]
        parsed = _parse_rpath_token(tokens, i)
        if parsed is None:
            result.append(token)
            i += 1
            continue

        rpath, consumed = parsed
        if rpath not in seen_rpaths:
            result.extend(tokens[i : i + consumed])
            seen_rpaths.add(rpath)
        i += consumed
    return shlex.join(result), seen_rpaths


def _drop_known_rpath_flags(flags, rpaths_to_drop):
    if not rpaths_to_drop:
        return flags

    tokens = shlex.split(flags)
    result = []
    i = 0
    while i < len(tokens):
        parsed = _parse_rpath_token(tokens, i)
        if parsed is None:
            result.append(tokens[i])
            i += 1
            continue

        rpath, consumed = parsed
        if rpath not in rpaths_to_drop:
            result.extend(tokens[i : i + consumed])
        i += consumed
    return shlex.join(result)


def _parse_rpath_token(tokens, index):
    token = tokens[index]
    if token.startswith("-Wl,-rpath,"):
        return token.removeprefix("-Wl,-rpath,"), 1
    if token == "-Wl,-rpath" and index + 1 < len(tokens):
        return tokens[index + 1], 2
    if token == "-rpath" and index + 1 < len(tokens):
        return tokens[index + 1], 2
    return None


def _set_sysconfig_var(key, value):
    config_vars = sysconfig.get_config_vars()
    if key in config_vars:
        config_vars[key] = value
