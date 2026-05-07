# pattern: Imperative Shell

import glob
import os
import sys

from pathlib import Path

import jax
import numpy as np
import scipy

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["include-dirs"] = [os.path.dirname(scipy.__file__)]
        if self.target_name != "sdist":
            build_ffi_cpu_extension(self.root)
            for artifact in ffi_cpu_extension_artifacts(self.root):
                relative_artifact = os.path.relpath(artifact, self.root)
                build_data["artifacts"].append(f"/{relative_artifact}")
                build_data["force_include"][relative_artifact] = relative_artifact


def get_include_dirs():
    return [os.path.dirname(scipy.__file__)]


def build_hook(config):
    config["include-dirs"] = get_include_dirs()


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
    pattern = os.path.join(
        root,
        "src",
        "linear_dag",
        "core",
        "jaxlinarg",
        "kernels",
        "_ffi_cpu_impl*.so",
    )
    return glob.glob(pattern)


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
