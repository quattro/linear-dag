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


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["include-dirs"] = [os.path.dirname(scipy.__file__)]
        if self.target_name != "sdist":
            build_ffi_cpu_extension_or_warn(self.root)
            for artifact in ffi_cpu_extension_artifacts(self.root):
                relative_artifact = os.path.relpath(artifact, self.root)
                build_data.setdefault("artifacts", []).append(f"/{relative_artifact}")
                build_data.setdefault("force_include", {})[relative_artifact] = relative_artifact


def get_include_dirs():
    return [os.path.dirname(scipy.__file__)]


def build_hook(config):
    config["include-dirs"] = get_include_dirs()


def build_ffi_cpu_extension_or_warn(root):
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


def is_ffi_cpu_build_required():
    return _truthy_env(os.environ.get(FFI_CPU_BUILD_REQUIRED_ENV))


def build_ffi_cpu_extension(root):
    root = Path(root)
    extension = Extension(
        "linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl",
        sources=[str(root / "src/linear_dag/core/jaxlinarg/kernels/ffi_cpu_impl.cc")],
        include_dirs=[
            jax.ffi.include_dir(),
            np.get_include(),
            os.path.dirname(scipy.__file__),
            *_macos_cxx_include_dirs(),
        ],
        language="c++",
        extra_compile_args=_cxx_compile_args(),
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


def _cxx_compile_args():
    args = ["-std=c++17"]
    if sys.platform == "darwin":
        args.append("-stdlib=libc++")
    return args


def _macos_cxx_include_dirs():
    if sys.platform != "darwin":
        return []
    candidates = sorted(glob.glob("/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk/usr/include/c++/v1"))
    return candidates[-1:] if candidates else []
