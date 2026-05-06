# pattern: Functional Core

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .operator import Backend

_KERNELS_MESSAGE = "JaxLinearARG numerical kernels are implemented in Phase 2"


def _shape_tuple(shape: Any) -> tuple[int, int]:
    return tuple(shape)


class JaxParallelOperator(eqx.Module):
    r"""JAX-compatible parallel operator scaffold.

    !!! info
        Numerical products are intentionally unavailable until Phase 2.

    **Arguments:**

    - `blocks`: Block operators composed by this wrapper.
    - `shape`: Matrix shape as `(n_rows, n_cols)`.
    - `backend`: Requested numerical backend.
    - `dtype`: Computation dtype.
    """

    blocks: tuple[Any, ...] = eqx.field(converter=tuple)
    shape: tuple[int, int] = eqx.field(converter=_shape_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)
    dtype: Any = eqx.field(default=jnp.float32, converter=jnp.dtype, static=True)

    def __check_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("shape must contain exactly two dimensions")
        if self.shape[0] < 0 or self.shape[1] < 0:
            raise ValueError("shape dimensions must be nonnegative")

    def matmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def rmatmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)
