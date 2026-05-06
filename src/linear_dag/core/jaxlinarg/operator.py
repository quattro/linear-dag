# pattern: Functional Core

from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility.

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


_KERNELS_MESSAGE = "JaxLinearARG numerical kernels are implemented in Phase 2"


class Backend(StrEnum):
    AUTO = "auto"
    PURE_JAX = "pure_jax"
    FFI_CPU = "ffi_cpu"
    PALLAS_GPU = "pallas_gpu"


class JaxLinearARG(eqx.Module):
    r"""JAX-compatible LinearARG operator scaffold.

    !!! info
        Numerical products are intentionally unavailable until Phase 2.

    **Arguments:**

    - `indptr`: CSC index pointer array.
    - `indices`: CSC row index array.
    - `data`: CSC edge value array.
    - `src_of_edge`: Source node index for each edge.
    - `variant_indices`: Variant node indices.
    - `flip`: Allele flip flags aligned to `variant_indices`.
    - `sample_indices`: Sample node indices.
    - `nonunique_indices`: Nonunique sample index mapping.
    - `n_variants`: Number of variants in genotype space.
    - `n_samples`: Number of samples in genotype space.
    - `backend`: Requested numerical backend.
    - `dtype`: Computation dtype.
    - `transpose`: Whether this view is transposed.
    """

    indptr: Any = eqx.field(converter=jnp.asarray)
    indices: Any = eqx.field(converter=jnp.asarray)
    data: Any = eqx.field(converter=jnp.asarray)
    src_of_edge: Any = eqx.field(converter=jnp.asarray)
    variant_indices: Any = eqx.field(converter=jnp.asarray)
    flip: Any = eqx.field(converter=jnp.asarray)
    sample_indices: Any = eqx.field(converter=jnp.asarray)
    nonunique_indices: Any = eqx.field(converter=jnp.asarray)
    n_variants: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)
    dtype: Any = eqx.field(default=jnp.float32, converter=jnp.dtype, static=True)
    transpose: bool = eqx.field(default=False, converter=bool, static=True)

    def __check_init__(self) -> None:
        arrays = {
            "indptr": self.indptr,
            "indices": self.indices,
            "data": self.data,
            "src_of_edge": self.src_of_edge,
            "variant_indices": self.variant_indices,
            "flip": self.flip,
            "sample_indices": self.sample_indices,
            "nonunique_indices": self.nonunique_indices,
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be rank 1")

        n_edges = self.indices.shape[0]
        if self.data.shape[0] != n_edges:
            raise ValueError("data must have the same length as indices")
        if self.src_of_edge.shape[0] != n_edges:
            raise ValueError("src_of_edge must have the same length as indices")
        if self.indptr.shape[0] == 0:
            raise ValueError("indptr must contain at least one entry")
        if self.nonunique_indices.shape[0] != self.indptr.shape[0] - 1:
            raise ValueError("nonunique_indices length must match the node count from indptr")
        if self.variant_indices.shape[0] != self.flip.shape[0]:
            raise ValueError("variant_indices and flip must have the same length")
        if self.variant_indices.shape[0] != self.n_variants:
            raise ValueError("variant_indices length must match n_variants")
        if self.sample_indices.shape[0] != self.n_samples:
            raise ValueError("sample_indices length must match n_samples")
        if self.n_variants < 0:
            raise ValueError("n_variants must be nonnegative")
        if self.n_samples < 0:
            raise ValueError("n_samples must be nonnegative")

    def matmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def rmatmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def _matmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def _rmatmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def matvec(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def rmatvec(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    @property
    def T(self) -> "_TransposeView":
        raise NotImplementedError(_KERNELS_MESSAGE)

    def transpose_view(self) -> "_TransposeView":
        raise NotImplementedError(_KERNELS_MESSAGE)


class _TransposeView(eqx.Module):
    parent: JaxLinearARG

    def matmat(self, x: Any) -> Any:
        return self.parent.rmatmat(x)

    def rmatmat(self, x: Any) -> Any:
        return self.parent.matmat(x)
