# pattern: Mixed (unavoidable)
# Reason: Backend dispatch is pure, but module construction must normalize the
# requested runtime backend and emit the required user-facing fallback warning.

import warnings

from enum import Enum
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .kernels import ffi_cpu
from .kernels.pure_jax import (
    pure_jax_solve_backward_compressed,
    pure_jax_solve_forward_compressed,
)

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility.

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


class Backend(StrEnum):
    """Numerical backend choices for JAX LinearARG solves."""

    AUTO = "auto"
    PURE_JAX = "pure_jax"
    FFI_CPU = "ffi_cpu"


_FORWARD_SOLVERS = {
    Backend.PURE_JAX: pure_jax_solve_forward_compressed,
    Backend.FFI_CPU: ffi_cpu.ffi_cpu_solve_forward,
}

_BACKWARD_SOLVERS = {
    Backend.PURE_JAX: pure_jax_solve_backward_compressed,
    Backend.FFI_CPU: ffi_cpu.ffi_cpu_solve_backward,
}


def resolve_backend(requested: Backend, *, platform: str | None = None) -> Backend:
    """Resolve a requested backend to a concrete executable backend."""
    requested = Backend(requested)
    platform = (jax.default_backend() if platform is None else platform).lower()

    if requested is Backend.PURE_JAX:
        return Backend.PURE_JAX
    if requested is Backend.AUTO:
        if platform == "cpu" and ffi_cpu.is_ffi_cpu_available():
            return Backend.FFI_CPU
        return Backend.PURE_JAX
    if requested is Backend.FFI_CPU:
        if ffi_cpu.is_ffi_cpu_available():
            return Backend.FFI_CPU
        warnings.warn(
            _ffi_cpu_unavailable_message(),
            UserWarning,
            stacklevel=2,
        )
        return Backend.PURE_JAX
    raise ValueError(f"unknown backend: {requested}")


class JaxLinearARG(eqx.Module):
    r"""JAX-compatible LinearARG operator.

    !!! info
        `Backend.PURE_JAX` is always available and supports JAX transforms.
        `Backend.FFI_CPU` uses the native CPU FFI handler when it is installed;
        if the handler is unavailable, explicit FFI CPU requests warn and fall
        back to `Backend.PURE_JAX`. Accelerator platforms currently use
        `Backend.PURE_JAX`. Reverse-mode autodiff is supported through custom
        VJP rules; forward mode is not supported for the solve primitive.

    **Arguments:**

    - `indptr`: CSC index pointer array.
    - `indices`: CSC row index array.
    - `data`: CSC edge value array.
    - `variant_indices`: Variant node indices.
    - `flip`: Allele flip flags aligned to `variant_indices`.
    - `sample_indices`: Sample node indices.
    - `nonunique_indices`: Nonunique sample index mapping.
    - `allele_counts`: Cached allele counts aligned to variants, or `-1` for
      missing entries.
    - `n_variants`: Number of variants in genotype space.
    - `n_samples`: Number of samples in genotype space.
    - `backend`: Requested numerical backend.
    - `dtype`: Computation dtype.
    """

    indptr: Any = eqx.field(converter=jnp.asarray)
    indices: Any = eqx.field(converter=jnp.asarray)
    data: Any = eqx.field(converter=jnp.asarray)
    variant_indices: Any = eqx.field(converter=jnp.asarray)
    flip: Any = eqx.field(converter=jnp.asarray)
    sample_indices: Any = eqx.field(converter=jnp.asarray)
    nonunique_indices: Any = eqx.field(converter=jnp.asarray)
    allele_counts: Any = eqx.field(converter=jnp.asarray)
    n_variants: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    n_nonunique_indices: int = eqx.field(default=-1, static=True)
    min_index_to_keep: int = eqx.field(default=0, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=resolve_backend, static=True)
    dtype: Any = eqx.field(default=jnp.float32, converter=jnp.dtype, static=True)
    _arrays_validated: bool = eqx.field(default=False, converter=bool, static=True)
    _flipped_variant_indices: Any = eqx.field(
        default_factory=lambda: jnp.asarray([], dtype=jnp.int32),
        converter=jnp.asarray,
    )

    @classmethod
    def from_lineararg_arrays(
        cls,
        *,
        indptr: Any,
        indices: Any,
        data: Any,
        variant_indices: Any,
        flip: Any,
        sample_indices: Any,
        nonunique_indices: Any | None,
        n_variants: int,
        n_samples: int,
        n_nonunique_indices: int | None = None,
        allele_counts: Any | None = None,
        backend: Backend = Backend.AUTO,
        dtype: Any = jnp.float32,
    ) -> "JaxLinearARG":
        """Construct a JAX operator from LinearARG array components.

        **Arguments:**

        - `indptr`: CSC index pointer array.
        - `indices`: CSC row index array.
        - `data`: CSC edge value array.
        - `variant_indices`: Variant node indices.
        - `flip`: Allele flip flags aligned to `variant_indices`.
        - `sample_indices`: Sample node indices.
        - `nonunique_indices`: Optional compressed node mapping.
        - `n_variants`: Number of variants in genotype space.
        - `n_samples`: Number of samples in genotype space.
        - `n_nonunique_indices`: Optional compressed-node count.
        - `allele_counts`: Optional allele counts aligned to variants.
        - `backend`: Requested numerical backend.
        - `dtype`: Computation dtype.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].

        **Raises:**

        - `ValueError`: If array shapes, indices, or backend settings are invalid.
        """
        dtype = jnp.dtype(dtype)
        n_variants = int(n_variants)
        n_samples = int(n_samples)

        indptr = np.asarray(indptr, dtype=np.int32)
        indices = np.asarray(indices, dtype=np.int32)
        data = np.asarray(data, dtype=np.dtype(dtype))
        variant_indices = np.asarray(variant_indices, dtype=np.int32)
        flip = np.asarray(flip, dtype=np.bool_)
        sample_indices = np.asarray(sample_indices, dtype=np.int32)
        allele_counts = _canonical_allele_counts_array(allele_counts, n_variants=n_variants)

        node_count = int(indptr.shape[0]) - 1
        if nonunique_indices is None:
            nonunique_indices = np.arange(node_count, dtype=np.int32)
        nonunique_indices = np.asarray(nonunique_indices, dtype=np.int32)
        real_n_nonunique_indices = int(np.max(nonunique_indices)) + 1 if nonunique_indices.size else 0
        if n_nonunique_indices is None:
            n_nonunique_indices = real_n_nonunique_indices
        else:
            n_nonunique_indices = int(n_nonunique_indices)
            if n_nonunique_indices < real_n_nonunique_indices:
                raise ValueError("n_nonunique_indices cannot be smaller than the maximum nonunique index")
        min_index_to_keep = int(sample_indices[-1]) if sample_indices.size else 0
        _validate_array_contract(
            {
                "indptr": indptr,
                "indices": indices,
                "data": data,
                "variant_indices": variant_indices,
                "flip": flip,
                "sample_indices": sample_indices,
                "nonunique_indices": nonunique_indices,
                "allele_counts": allele_counts,
            },
            n_variants=n_variants,
            n_samples=n_samples,
            n_nonunique_indices=n_nonunique_indices,
            min_index_to_keep=min_index_to_keep,
        )

        return cls(
            indptr=jnp.asarray(indptr, dtype=jnp.int32),
            indices=jnp.asarray(indices, dtype=jnp.int32),
            data=jnp.asarray(data, dtype=dtype),
            variant_indices=jnp.asarray(variant_indices, dtype=jnp.int32),
            flip=jnp.asarray(flip, dtype=jnp.bool_),
            _flipped_variant_indices=jnp.asarray(np.flatnonzero(flip), dtype=jnp.int32),
            sample_indices=sample_indices,
            nonunique_indices=nonunique_indices,
            allele_counts=jnp.asarray(allele_counts, dtype=jnp.int32),
            n_variants=n_variants,
            n_samples=n_samples,
            n_nonunique_indices=n_nonunique_indices,
            min_index_to_keep=min_index_to_keep,
            backend=backend,
            dtype=dtype,
            _arrays_validated=True,
        )

    @classmethod
    def from_lineararg(
        cls,
        linarg: Any,
        *,
        backend: Backend = Backend.AUTO,
        dtype: Any = None,
    ) -> "JaxLinearARG":
        """Construct a JAX operator from a [`linear_dag.core.lineararg.LinearARG`][].

        !!! info
            `Backend.AUTO` resolves from the active JAX platform. CPU uses
            `Backend.FFI_CPU` when the native handler is registered and
            otherwise falls back to `Backend.PURE_JAX`.

        **Arguments:**

        - `linarg`: Source LinearARG object.
        - `backend`: Requested numerical backend.
        - `dtype`: Optional computation dtype.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
        """
        from .ingress import from_lineararg

        return from_lineararg(
            linarg,
            backend=backend,
            dtype=dtype,
        )

    @classmethod
    def from_hdf5_block(
        cls,
        path: Any,
        block: Any,
        *,
        backend: Backend = Backend.AUTO,
        load_metadata: bool = False,
        dtype: Any = None,
    ) -> "JaxLinearARG":
        """Construct a JAX operator from one HDF5 LinearARG block.

        **Arguments:**

        - `path`: HDF5 file path.
        - `block`: Block name inside the HDF5 file.
        - `backend`: Requested numerical backend.
        - `load_metadata`: Whether to load optional LinearARG metadata.
        - `dtype`: Optional computation dtype.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
        """
        from .ingress import from_hdf5_block

        return from_hdf5_block(
            path,
            block,
            backend=backend,
            load_metadata=load_metadata,
            dtype=dtype,
        )

    def __check_init__(self) -> None:
        if self._arrays_validated:
            _validate_array_shapes(
                {
                    "indptr": self.indptr,
                    "indices": self.indices,
                    "data": self.data,
                    "variant_indices": self.variant_indices,
                    "flip": self.flip,
                    "sample_indices": self.sample_indices,
                    "nonunique_indices": self.nonunique_indices,
                    "allele_counts": self.allele_counts,
                },
                n_variants=self.n_variants,
                n_samples=self.n_samples,
                n_nonunique_indices=self.n_nonunique_indices,
                min_index_to_keep=self.min_index_to_keep,
            )
            return

        # Direct constructor calls have not passed through the NumPy ingress
        # validation above. Validate them on the host once; these checks produce
        # Python exceptions and are intentionally outside traced numerical code.
        _validate_array_contract(
            {
                "indptr": np.asarray(self.indptr),
                "indices": np.asarray(self.indices),
                "data": np.asarray(self.data),
                "variant_indices": np.asarray(self.variant_indices),
                "flip": np.asarray(self.flip),
                "sample_indices": np.asarray(self.sample_indices),
                "nonunique_indices": np.asarray(self.nonunique_indices),
                "allele_counts": np.asarray(self.allele_counts),
            },
            n_variants=self.n_variants,
            n_samples=self.n_samples,
            n_nonunique_indices=self.n_nonunique_indices,
            min_index_to_keep=self.min_index_to_keep,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the operator shape `(n_samples, n_variants)`."""
        return (self.n_samples, self.n_variants)

    def matmat(self, x: Any) -> Any:
        """Multiply by the represented genotype matrix.

        **Arguments:**

        - `x`: Rank-1 or rank-2 array with leading dimension `n_variants`.

        **Returns:**

        - Product with leading dimension `n_samples`.

        **Raises:**

        - `ValueError`: If `x` has an incompatible rank or leading dimension.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.n_variants, dtype=self.dtype)
        flip_sign = jnp.where(self.flip, -1, 1).astype(x.dtype)
        b = jnp.zeros((self.n_nonunique_indices, x.shape[1]), dtype=x.dtype)
        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        b = b.at[variant_nonunique_indices, :].add(x * flip_sign[:, None])
        solved = _solve_forward(
            self.backend,
            self.indptr,
            self.indices,
            self.data,
            self.nonunique_indices,
            self.min_index_to_keep,
            b,
        )
        # Match the NumPy/Cython path: only flipped rows contribute to this
        # correction. A dense mask multiply materializes work for every variant.
        flip_sum = jnp.sum(x[self._flipped_variant_indices, :], axis=0)
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        result = solved[sample_nonunique_indices, :] + flip_sum
        return result[:, 0] if was_vector else result

    def rmatmat(self, x: Any) -> Any:
        """Multiply by the transpose of the represented genotype matrix.

        **Arguments:**

        - `x`: Rank-1 or rank-2 array with leading dimension `n_samples`.

        **Returns:**

        - Product with leading dimension `n_variants`.

        **Raises:**

        - `ValueError`: If `x` has an incompatible rank or leading dimension.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.n_samples, dtype=self.dtype)
        b = jnp.zeros((self.n_nonunique_indices, x.shape[1]), dtype=x.dtype)
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        b = b.at[sample_nonunique_indices, :].set(x)
        solved = _solve_backward(
            self.backend,
            self.indptr,
            self.indices,
            self.data,
            self.nonunique_indices,
            self.min_index_to_keep,
            b,
        )
        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        values = solved[variant_nonunique_indices, :]
        total = jnp.sum(x, axis=0)
        result = jnp.where(self.flip[:, None], total[None, :] - values, values)
        return result[:, 0] if was_vector else result

    def matvec(self, x: Any) -> Any:
        """Multiply a vector by the represented genotype matrix."""
        return self.matmat(x)

    def rmatvec(self, x: Any) -> Any:
        """Multiply a vector by the transpose of the represented matrix."""
        return self.rmatmat(x)

    @property
    def T(self) -> "_TransposeView":
        """Return a lightweight transpose view."""
        return self.transpose_view()

    def transpose_view(self) -> "_TransposeView":
        """Return a lightweight transpose view of this operator."""
        return _TransposeView(self)

    def __matmul__(self, x: Any) -> Any:
        return self.matmat(x)


class _TransposeView(eqx.Module):
    """Lightweight transpose view for [`linear_dag.core.jaxlinarg.JaxLinearARG`][]."""

    parent: JaxLinearARG

    @property
    def shape(self) -> tuple[int, int]:
        """Return the transposed operator shape."""
        rows, cols = self.parent.shape
        return (cols, rows)

    def matmat(self, x: Any) -> Any:
        """Multiply by the transposed matrix."""
        return self.parent.rmatmat(x)

    def rmatmat(self, x: Any) -> Any:
        """Multiply by the original matrix."""
        return self.parent.matmat(x)

    def matvec(self, x: Any) -> Any:
        """Multiply a vector by the transposed matrix."""
        return self.matmat(x)

    def rmatvec(self, x: Any) -> Any:
        """Multiply a vector by the original matrix."""
        return self.rmatmat(x)

    @property
    def T(self) -> JaxLinearARG:
        """Return the original non-transposed operator."""
        return self.transpose_view()

    def transpose_view(self) -> JaxLinearARG:
        """Return the original non-transposed operator."""
        return self.parent

    def __matmul__(self, x: Any) -> Any:
        return self.matmat(x)


def _canonical_allele_counts_array(allele_counts: Any | None, *, n_variants: int) -> np.ndarray:
    if allele_counts is None:
        return np.full((n_variants,), -1, dtype=np.int32)
    return np.asarray(allele_counts, dtype=np.int32)


def _validate_array_shapes(
    arrays: dict[str, Any],
    *,
    n_variants: int,
    n_samples: int,
    n_nonunique_indices: int,
    min_index_to_keep: int,
) -> None:
    for name, array in arrays.items():
        if array.ndim != 1:
            raise ValueError(f"{name} must be rank 1")

    n_edges = arrays["indices"].shape[0]
    if arrays["data"].shape[0] != n_edges:
        raise ValueError("data must have the same length as indices")
    if arrays["indptr"].shape[0] == 0:
        raise ValueError("indptr must contain at least one entry")
    if arrays["nonunique_indices"].shape[0] < arrays["indptr"].shape[0] - 1:
        raise ValueError("nonunique_indices length must cover the node count from indptr")
    if arrays["variant_indices"].shape[0] != arrays["flip"].shape[0]:
        raise ValueError("variant_indices and flip must have the same length")
    if arrays["variant_indices"].shape[0] != n_variants:
        raise ValueError("variant_indices length must match n_variants")
    if arrays["allele_counts"].shape[0] != n_variants:
        raise ValueError("allele_counts length must match n_variants")
    if arrays["sample_indices"].shape[0] != n_samples:
        raise ValueError("sample_indices length must match n_samples")
    if n_variants < 0:
        raise ValueError("n_variants must be nonnegative")
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative")
    if n_nonunique_indices < 0:
        raise ValueError("n_nonunique_indices must be nonnegative")


def _validate_array_contract(
    arrays: dict[str, np.ndarray],
    *,
    n_variants: int,
    n_samples: int,
    n_nonunique_indices: int,
    min_index_to_keep: int,
) -> None:
    _validate_array_shapes(
        arrays,
        n_variants=n_variants,
        n_samples=n_samples,
        n_nonunique_indices=n_nonunique_indices,
        min_index_to_keep=min_index_to_keep,
    )
    indptr = arrays["indptr"]
    indices = arrays["indices"]
    data = arrays["data"]
    node_count = indptr.shape[0] - 1
    n_edges = indices.shape[0]

    if int(indptr[0]) != 0:
        raise ValueError("indptr must start at 0")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be monotonic")
    if int(indptr[-1]) != n_edges:
        raise ValueError("final indptr entry must match the edge count")

    for name in (
        "indices",
        "variant_indices",
        "sample_indices",
        "nonunique_indices",
    ):
        array = arrays[name]
        if array.shape[0] and int(np.min(array)) < 0:
            raise ValueError(f"{name} contains a negative index")

    if min_index_to_keep < 0 or min_index_to_keep > node_count:
        raise ValueError("min_index_to_keep must be within the node range")
    if indices.shape[0] and int(np.max(indices)) >= node_count:
        raise ValueError("indices contains an out-of-range node index")
    source_indices = np.repeat(np.arange(node_count, dtype=np.int32), np.diff(indptr))
    invalid_edge_order = (indices < source_indices) | ((indices == source_indices) & (data != 0))
    if indices.shape[0] and np.any(invalid_edge_order):
        raise ValueError("indices must be greater than their source nodes")
    if arrays["variant_indices"].shape[0] and int(np.max(arrays["variant_indices"])) >= node_count:
        raise ValueError("variant_indices contains an out-of-range node index")
    if arrays["sample_indices"].shape[0] and int(np.max(arrays["sample_indices"])) >= node_count:
        raise ValueError("sample_indices contains an out-of-range node index")
    if arrays["nonunique_indices"].shape[0] and int(np.max(arrays["nonunique_indices"])) >= n_nonunique_indices:
        raise ValueError("nonunique_indices contains an out-of-range compressed index")


def _ffi_cpu_unavailable_message() -> str:
    error = ffi_cpu.last_ffi_cpu_error()
    if error is None:
        return "FFI_CPU backend is unavailable; falling back to PURE_JAX."
    return f"FFI_CPU backend is unavailable ({error}); falling back to PURE_JAX."


def _as_rank2_matrix(x: Any, *, expected_rows: int, dtype: Any) -> tuple[jax.Array, bool]:
    array = jnp.asarray(x, dtype=dtype)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
        was_vector = True
    elif array.ndim == 2:
        was_vector = False
    else:
        raise ValueError(f"expected rank 1 or 2 input, got rank {array.ndim}")
    if array.shape[0] != expected_rows:
        raise ValueError(f"expected leading dimension {expected_rows}, got {array.shape[0]}")
    return array, was_vector


# The solve kernels are split by direction at the Python boundary so `matmat`
# and `rmatmat` read as forward/backward operations instead of string-dispatching
# through one generic helper.
#
# Algebraically, each helper is a linear map from the right-hand side `b` to the
# solved state. If the forward solve applies some linear operator `S`, then for a
# scalar loss L and cotangent g = dL/d(S b), the gradient with respect to `b` is
# S.T @ g. In this LinearARG representation, applying S.T is exactly the
# opposite-direction triangular solve over the same compressed graph. That is why
# `_solve_forward_bwd` calls `_solve_backward`, and `_solve_backward_bwd` calls
# `_solve_forward`.
#
# `jax.custom_vjp` disables forward-mode AD for these wrapped functions, so the
# comments below focus only on reverse mode.
@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def _solve_forward(
    backend: Backend,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    # Primal forward solve used by `matmat`.
    backend = _resolve_solve_backend(backend)
    return _FORWARD_SOLVERS[backend](
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def _solve_forward_fwd(
    backend: Backend,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> tuple[jax.Array, tuple[Any, Any, Any, Any]]:
    # This is the "fwd" half of JAX's custom VJP protocol, not another forward
    # triangular solve. It computes the primal output and returns residual data
    # that the "bwd" half will need later.
    #
    # We run the primal solve via `.fun` to call the raw wrapped function. Calling
    # `_solve_forward(...)` here would recursively enter this custom VJP rule.
    result = _solve_forward.fun(
        backend,
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )
    # The backward pass needs the graph arrays to apply S.T to the incoming
    # cotangent. It does not need the solved primal output because this solve is
    # linear in `b`; the derivative does not depend on the value of `b`.
    #
    # `min_index_to_keep` is a nondiff/static argument, so JAX passes it directly
    # to `_solve_forward_bwd` instead of storing it in the residual.
    return result, (indptr, indices, data, nonunique_indices)


def _solve_forward_bwd(
    backend: Backend,
    min_index_to_keep: int,
    residual: tuple[Any, Any, Any, Any],
    grad: Any,
) -> tuple[None, None, None, None, jax.Array]:
    indptr, indices, data, nonunique_indices = residual
    # `grad` is the cotangent for the solved state, dL/d(S b). To get the
    # cotangent for the right-hand side, we apply the adjoint map S.T to `grad`.
    # The graph is triangular/topologically ordered, so S.T is implemented by the
    # backward solve over the same compressed graph arrays.
    #
    # `.fun` keeps this as a plain primal call and avoids nesting custom VJP
    # traces. The transpose solve is already the complete gradient calculation.
    grad_b = _solve_backward.fun(
        backend,
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        grad,
    )
    # Return cotangents only for differentiable arguments. Backend selection,
    # graph structure, and the static cutoff are treated as fixed operator state;
    # only the right-hand side `b` receives dL/db.
    return None, None, None, None, grad_b


_solve_forward.defvjp(_solve_forward_fwd, _solve_forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 5))
def _solve_backward(
    backend: Backend,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    # Primal backward solve used by `rmatmat`.
    backend = _resolve_solve_backend(backend)
    return _BACKWARD_SOLVERS[backend](
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def _solve_backward_fwd(
    backend: Backend,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> tuple[jax.Array, tuple[Any, Any, Any, Any]]:
    # This mirrors `_solve_forward_fwd`: compute the primal backward solve and
    # save only the fixed graph arrays needed to apply the adjoint in reverse
    # mode. The primal output itself is unnecessary because the solve is linear.
    result = _solve_backward.fun(
        backend,
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )
    return result, (indptr, indices, data, nonunique_indices)


def _solve_backward_bwd(
    backend: Backend,
    min_index_to_keep: int,
    residual: tuple[Any, Any, Any, Any],
    grad: Any,
) -> tuple[None, None, None, None, jax.Array]:
    indptr, indices, data, nonunique_indices = residual
    # Here the primal map is the backward solve, call it T. The incoming `grad`
    # is dL/d(T b), so dL/db = T.T @ grad. T.T is the forward solve over the same
    # graph, which is exactly the pairing used by `matmat` and `rmatmat`.
    grad_b = _solve_forward.fun(
        backend,
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        grad,
    )
    # Only the right-hand side `b` is differentiable; graph arrays remain fixed
    # operator data.
    return None, None, None, None, grad_b


_solve_backward.defvjp(_solve_backward_fwd, _solve_backward_bwd)


def _resolve_solve_backend(backend: Backend) -> Backend:
    # Backend availability is resolved once at the solve boundary so the primal
    # helpers and VJP helpers can share identical fallback behavior.
    if backend is Backend.AUTO:
        backend = resolve_backend(backend)
    if backend is Backend.FFI_CPU and not ffi_cpu.is_ffi_cpu_available():
        warnings.warn(
            _ffi_cpu_unavailable_message(),
            UserWarning,
            stacklevel=2,
        )
        return Backend.PURE_JAX
    if backend not in _FORWARD_SOLVERS:
        raise ValueError(f"unknown backend: {backend.value}")
    return backend
