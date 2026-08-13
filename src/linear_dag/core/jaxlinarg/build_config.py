# pattern: Imperative Shell

"""Runtime diagnostics for the optional JAX native extension."""

from __future__ import annotations

from typing import Any

import jax
import jaxlib

from .kernels import ffi_cpu


def show_build_config() -> dict[str, Any]:
    """Return runtime and native-extension build configuration.

    **Returns:**

    - A dictionary describing the active JAX runtime and optional native CPU FFI
      extension. `ffi_cpu_built` reports whether the extension imports, while
      `ffi_cpu_available` reports whether JAX accepted its registered CPU FFI
      targets in the current runtime.
    """
    ffi_cpu_built = ffi_cpu.is_ffi_cpu_built()
    ffi_cpu_available = ffi_cpu.is_ffi_cpu_available()
    error = ffi_cpu.last_ffi_cpu_error()
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "ffi_cpu_built": ffi_cpu_built,
        "ffi_cpu_available": ffi_cpu_available,
        "ffi_cpu_blas_enabled": ffi_cpu.is_ffi_cpu_blas_enabled(),
        "ffi_cpu_blas_backend": ffi_cpu.ffi_cpu_blas_backend(),
        "ffi_cpu_native_tuning": ffi_cpu.is_ffi_cpu_native_tuning_enabled(),
        "ffi_cpu_error": None if error is None else str(error),
    }
