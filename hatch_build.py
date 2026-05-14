# pattern: Imperative Shell

import glob
import importlib.machinery
import os
import sys
import sysconfig
import warnings

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
        build_data["include-dirs"] = [os.path.dirname(scipy.__file__)]
        if self.target_name != "sdist":
            if build_ffi_cpu_extension_or_warn(self.root):
                for artifact in ffi_cpu_extension_artifacts(self.root):
                    relative_artifact = os.path.relpath(artifact, self.root)
                    build_data.setdefault("artifacts", []).append(f"/{relative_artifact}")
                    build_data.setdefault("force_include", {})[relative_artifact] = relative_artifact


def get_include_dirs():
    return [os.path.dirname(scipy.__file__)]


def build_hook(config):
    config["include-dirs"] = get_include_dirs()


def build_ffi_cpu_extension_or_warn(root):
    remove_ffi_cpu_extension_artifacts(root)
    try:
        build_ffi_cpu_extension(root)
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


def remove_ffi_cpu_extension_artifacts(root):
    for artifact in ffi_cpu_extension_artifacts(root):
        Path(artifact).unlink()


def is_ffi_cpu_build_required():
    return _truthy_env(os.environ.get(FFI_CPU_BUILD_REQUIRED_ENV))


def _ffi_cpu_blas_options():
    empty_options = {
        "define_macros": [],
        "include_dirs": [],
        "library_dirs": [],
        "libraries": [],
        "extra_link_args": [],
    }
    if _truthy_env(os.environ.get(FFI_CPU_BLAS_DISABLED_ENV)):
        return empty_options
    if sys.platform == "darwin":
        return {
            **empty_options,
            "define_macros": [("LINEAR_DAG_HAVE_CBLAS", "1")],
            "extra_link_args": ["-framework", "Accelerate"],
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
        }

    if _truthy_env(os.environ.get(FFI_CPU_BLAS_REQUIRED_ENV)):
        raise RuntimeError(
            "Could not find cblas.h and a BLAS library for the CPU FFI extension. "
            f"Set {FFI_CPU_BLAS_DISABLED_ENV}=1 to build the scalar fallback instead."
        )
    return empty_options


def build_ffi_cpu_extension(root):
    root = Path(root)
    blas_options = _ffi_cpu_blas_options()
    extension = Extension(
        "linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl",
        sources=[str(root / "src/linear_dag/core/jaxlinarg/kernels/ffi_cpu_impl.cc")],
        include_dirs=[
            jax.ffi.include_dir(),
            np.get_include(),
            os.path.dirname(scipy.__file__),
            *_macos_cxx_include_dirs(),
            *blas_options["include_dirs"],
        ],
        define_macros=blas_options["define_macros"],
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
    command.build_lib = str(root / "build")
    command.build_temp = str(root / "build/temp")
    command.ensure_finalized()
    command.run()


def ffi_cpu_extension_artifacts(root):
    base = Path(root) / "src" / "linear_dag" / "core" / "jaxlinarg" / "kernels" / "_ffi_cpu_impl"
    artifacts = []
    for suffix in _extension_suffixes():
        artifacts.extend(glob.glob(f"{base}*{suffix}"))
    return sorted(set(artifacts))


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


def _macos_cxx_include_dirs():
    if sys.platform != "darwin":
        return []
    candidates = sorted(glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk/usr/include/c++/v1"))
    return candidates[-1:] if candidates else []
