# pattern: Imperative Shell

from typing import Any

from .operator import Backend


def from_lineararg(
    linarg: Any,
    *,
    backend: Backend = Backend.AUTO,
    bucket: tuple[int, int] | None = None,
    dtype: Any = None,
) -> Any:
    raise NotImplementedError


def from_hdf5_block(
    path: Any,
    block: Any,
    *,
    backend: Backend = Backend.AUTO,
    bucket: tuple[int, int] | None = None,
    load_metadata: bool = False,
    dtype: Any = None,
) -> Any:
    raise NotImplementedError
