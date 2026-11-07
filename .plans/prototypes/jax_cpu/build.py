"""Build the isolated prototype with the invoking Python/JAX headers."""

from pathlib import Path

import jax

from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext

root = Path(__file__).resolve().parent
extension = Extension(
    "_packed_cpu",
    [str(root / "kernel.cc")],
    include_dirs=[jax.ffi.include_dir()],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
)
command = build_ext(Distribution({"ext_modules": [extension]}))
command.build_lib = str(root)
command.build_temp = str(root / "build")
command.ensure_finalized()
command.run()
