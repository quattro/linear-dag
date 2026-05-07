# pattern: Imperative Shell

from __future__ import annotations

from functools import cache


@cache
def is_ffi_cpu_available() -> bool:
    """Return whether the native CPU FFI handler can be imported."""
    try:
        from linear_dag.core.jaxlinarg.kernels import _ffi_cpu_impl  # noqa: F401
    except ImportError:
        return False
    return True
