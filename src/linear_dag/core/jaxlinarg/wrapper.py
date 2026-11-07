# pattern: Functional Core

import operator

from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jaxtyping import Array, ArrayLike

from linear_dag.core.lineararg import LinearARG, list_blocks

from ._packed import _ExecutionConfig, _pack
from .operator import Backend, JaxLinearARG, resolve_backend


def variant_offsets_from_metadata(metadata: pl.DataFrame) -> np.ndarray:
    """Return leading-zero cumulative variant offsets in metadata row order.

    **Arguments:**

    - `metadata`: Block metadata containing an `n_variants` column.

    **Returns:**

    - An int64 array of length `metadata.height + 1`, starting at zero.

    **Raises:**

    - `ValueError`: If the `n_variants` column is missing.
    """
    _require_metadata_columns(metadata, "n_variants")
    n_variants = metadata.get_column("n_variants").to_numpy()
    return np.insert(np.cumsum(n_variants), 0, 0).astype(np.int64)


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


class JaxParallelOperator(eqx.Module):
    r"""JAX-compatible multi-block LinearARG operator.

    !!! info
        CPU FFI packs ragged blocks without padding and runs native workers
        on one CPU device. `num_processes` controls worker threads, not OS
        processes. Scratch space is bounded by `workspace_bytes` (256 MiB by
        default); worker count and RHS tile width are reduced to fit.
        The pure-JAX reference backend runs on one device without native workers.

        All blocks must represent the same sample rows in the same order.
        Construction checks row counts, but does not join or validate IID
        identities; callers establish sample alignment before conversion.

    !!! Example
        ```python
        from linear_dag.core.jaxlinarg import JaxParallelOperator

        operator = JaxParallelOperator.from_hdf5("lineararg.h5", num_processes=4)
        ```

    **Arguments:**

    - `blocks`: Block operators composed by this wrapper.
    - `variant_offsets`: Leading-zero cumulative variant offsets.
    - `backend`: Requested numerical backend.
    - `num_processes`: Maximum native CPU worker threads.
    - `max_num_traits`: Maximum RHS tile width (default 8).
    - `workspace_bytes`: Native scratch budget (default 256 MiB), excluding
      graph/input/output storage and transformation intermediates.
    """

    _blocks: tuple[JaxLinearARG, ...]
    variant_offsets: tuple[int, ...] = eqx.field(converter=_int_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)

    _graph: tuple[Array, ...] | None
    _config: _ExecutionConfig | None = eqx.field(static=True)
    _specs: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    _dtype: Any = eqx.field(static=True)

    def __init__(
        self,
        blocks: Iterable[JaxLinearARG],
        variant_offsets: Iterable[int],
        backend: Backend = Backend.AUTO,
        *,
        num_processes: int | None = None,
        max_num_traits: int = 8,
        workspace_bytes: int = 256 * 1024**2,
    ) -> None:
        # Validate execution controls even when the selected backend ignores them.
        num_processes = None if num_processes is None else operator.index(num_processes)
        max_num_traits = operator.index(max_num_traits)
        workspace_bytes = operator.index(workspace_bytes)

        self._blocks = tuple(blocks)
        self.variant_offsets = _int_tuple(variant_offsets)
        self.backend = Backend(backend)
        self._graph = None
        self._config = None
        self._specs = ()
        self._dtype = self._blocks[0].dtype if self._blocks else jnp.dtype(jnp.float32)
        # Validate the original block sequence before replacing it with packed
        # state. Equinox's automatic post-init check then sees the packed form.
        self.__check_init__()
        self.backend = resolve_backend(_backend_for_lineargs(self._blocks, backend=self.backend))
        if num_processes is not None and num_processes < 1:
            raise ValueError("num_processes must be positive")
        if max_num_traits < 1 or workspace_bytes < 1:
            raise ValueError("max_num_traits and workspace_bytes must be positive")
        if self.backend is Backend.FFI_CPU:
            keys = (
                "indptr",
                "indices",
                "data",
                "nonunique_indices",
                "variant_indices",
                "sample_indices",
                "flip",
                "allele_counts",
                "min_index_to_keep",
            )
            # Native kernels use shared-memory threads on a single CPU device.
            with jax.default_device(jax.devices("cpu")[0]):
                self._graph, self._config = _pack(
                    ({key: getattr(block, key) for key in keys} for block in self._blocks),
                    workers=num_processes,
                    chunk=max_num_traits,
                    workspace_bytes=workspace_bytes,
                    dtype=self._dtype,
                )
            self._specs = tuple(tuple(map(int, row)) for row in np.asarray(self._graph[0]))
            # Keep one canonical graph payload. Inspection reconstructs per-block
            # views on demand instead of retaining a duplicate set of arrays.
            self._blocks = ()

    @property
    def blocks(self) -> tuple[JaxLinearARG, ...]:
        """Return individual blocks for inspection.

        Packed operators reconstruct block arrays on demand. These are not
        cached and may allocate device buffers; avoid repeated access in hot paths.
        """
        if self._graph is None:
            return self._blocks
        _, ptr, indices, data, mapping, variants, samples, flip, counts = self._graph
        result = []
        for i, (po, no, eo, vo, nodes, nv, ns, keep) in enumerate(self._specs):
            edge_end = self._specs[i + 1][2] if i + 1 < len(self._specs) else indices.shape[0]
            result.append(
                JaxLinearARG(
                    indptr=ptr[po : po + nodes + 1],
                    indices=indices[eo:edge_end],
                    data=data[eo:edge_end],
                    nonunique_indices=mapping[no : no + nodes],
                    variant_indices=variants[vo : vo + nv],
                    sample_indices=samples[i * self.shape[0] : (i + 1) * self.shape[0]],
                    flip=flip[vo : vo + nv],
                    allele_counts=counts[vo : vo + nv],
                    n_variants=nv,
                    n_samples=self.shape[0],
                    n_nonunique_indices=ns,
                    min_index_to_keep=keep,
                    backend=self.backend,
                    dtype=self._dtype,
                    _arrays_validated=True,
                )
            )
        return tuple(result)

    @property
    def dtype(self) -> np.dtype:
        """Return the computation dtype."""
        return self._dtype

    @property
    def allele_counts(self) -> Array:
        """Return variant-aligned allele counts, or -1 for missing counts."""
        return self._graph[-1] if self._graph is not None else jnp.concatenate([b.allele_counts for b in self._blocks])

    @property
    def num_processes(self) -> int:
        """Return the effective native worker count (zero for the pure-JAX path)."""
        return self._config.workers if self._config is not None else 0

    @classmethod
    def from_linearargs(
        cls,
        lineargs: Iterable[LinearARG | JaxLinearARG],
        *,
        backend: Backend = Backend.AUTO,
        num_processes: int | None = None,
        max_num_traits: int = 8,
        workspace_bytes: int = 256 * 1024**2,
        dtype: Any = jnp.float32,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from LinearARG objects.

        !!! info
            When all inputs are already
            [`linear_dag.core.jaxlinarg.JaxLinearARG`][] objects and `backend`
            is `Backend.AUTO`, the wrapper preserves their common concrete
            backend.

        **Arguments:**

        - `lineargs`: Iterable of LinearARG or JAX LinearARG blocks.
        - `backend`: Requested numerical backend.
        - `num_processes`: Maximum native CPU worker threads; defaults to CPU
          count capped by the number of blocks and scratch budget.
        - `max_num_traits`: Maximum RHS tile width (default 8).
        - `workspace_bytes`: Native scratch budget, excluding graph, inputs,
          outputs, and JAX transformation intermediates.
        - `dtype`: Computation dtype for blocks loaded from host data.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxParallelOperator`][].

        **Raises:**

        - `ValueError`: If block settings or shapes are invalid.
        """
        lineargs = tuple(lineargs)
        backend = _backend_for_lineargs(lineargs, backend=backend)
        blocks = tuple(
            linearg
            if isinstance(linearg, JaxLinearARG)
            else JaxLinearARG.from_lineararg(linearg, backend=backend, dtype=dtype)
            for linearg in lineargs
        )
        return cls(
            blocks=blocks,
            variant_offsets=np.insert(np.cumsum([block.n_variants for block in blocks]), 0, 0),
            backend=backend,
            num_processes=num_processes,
            max_num_traits=max_num_traits,
            workspace_bytes=workspace_bytes,
        )

    @classmethod
    def from_hdf5(
        cls,
        path: Any,
        *,
        block_metadata: pl.DataFrame | None = None,
        backend: Backend = Backend.AUTO,
        num_processes: int | None = None,
        max_num_traits: int = 8,
        workspace_bytes: int = 256 * 1024**2,
        dtype: Any = jnp.float32,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from an HDF5 LinearARG file.

        !!! info
            On CPU, `Backend.AUTO` selects native execution when available.
            Otherwise it selects the single-device pure-JAX fallback.

        **Arguments:**

        - `path`: HDF5 file path.
        - `block_metadata`: Optional preloaded block metadata.
        - `backend`: Requested numerical backend.
        - `num_processes`: Maximum native CPU worker threads; defaults to CPU
          count capped by the number of blocks and scratch budget.
        - `max_num_traits`: Maximum RHS tile width (default 8).
        - `workspace_bytes`: Native scratch budget, excluding graph, inputs,
          outputs, and JAX transformation intermediates.
        - `dtype`: Computation dtype for blocks loaded from host data.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxParallelOperator`][].

        **Raises:**

        - `ValueError`: If metadata, block settings, or shapes are invalid.
        """
        metadata = list_blocks(path) if block_metadata is None else block_metadata
        block_names = metadata.get_column("block_name").to_list()
        blocks = tuple(
            JaxLinearARG.from_hdf5_block(
                path,
                block_name,
                backend=backend,
                dtype=dtype,
            )
            for block_name in block_names
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            backend=backend,
            num_processes=num_processes,
            max_num_traits=max_num_traits,
            workspace_bytes=workspace_bytes,
        )

    def __check_init__(self) -> None:
        if self._graph is not None:
            return
        if not self.blocks:
            raise ValueError("blocks must contain at least one JaxLinearARG")
        n_samples = {block.n_samples for block in self.blocks}
        if len(n_samples) != 1:
            raise ValueError("all blocks must have the same n_samples")
        dtypes = {jnp.dtype(block.dtype) for block in self.blocks}
        if len(dtypes) != 1:
            raise ValueError("all blocks must have the same dtype")
        _validate_jax_block_settings(
            self.blocks,
            backend=self.backend,
        )

        if len(self.variant_offsets) != len(self.blocks) + 1:
            raise ValueError("variant_offsets length must be n_blocks + 1")
        if self.variant_offsets[0] != 0:
            raise ValueError("variant_offsets must start at zero")
        offsets = np.asarray(self.variant_offsets, dtype=np.int64)
        if np.any(np.diff(offsets) < 0):
            raise ValueError("variant_offsets must be monotone")
        block_n_variants = np.asarray([block.n_variants for block in self.blocks], dtype=np.int64)
        if not np.array_equal(np.diff(offsets), block_n_variants):
            raise ValueError("variant_offsets increments must match each block n_variants")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the composed operator shape `(n_samples, total_variants)`."""
        return (
            self._config.samples if self._config is not None else self._blocks[0].n_samples,
            self.variant_offsets[-1],
        )

    def matmat(self, x: ArrayLike) -> Array:
        """Multiply by the concatenated multi-block genotype matrix.

        **Arguments:**

        - `x`: Rank-1 or rank-2 array with leading dimension equal to the total
          variant count.

        **Returns:**

        - Product with leading dimension equal to the sample count.

        **Raises:**

        - `ValueError`: If `x` has an incompatible rank or leading dimension.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[1], dtype=self.dtype)
        if self._graph is not None:
            from ._packed import _product

            result = _product(self._graph, x, self._config)
        elif len(self.blocks) == 1:
            result = self.blocks[0].matmat(x)
        else:
            result = jnp.zeros((self.shape[0], x.shape[1]), dtype=x.dtype)
            for block, start, end in zip(
                self.blocks,
                self.variant_offsets[:-1],
                self.variant_offsets[1:],
                strict=True,
            ):
                result = result + block.matmat(x[start:end])
        return result[:, 0] if was_vector else result

    def rmatmat(self, x: ArrayLike) -> Array:
        """Multiply by the transpose of the multi-block genotype matrix.

        **Arguments:**

        - `x`: Rank-1 or rank-2 array with leading dimension equal to the
          sample count.

        **Returns:**

        - Product with leading dimension equal to the total variant count.

        **Raises:**

        - `ValueError`: If `x` has an incompatible rank or leading dimension.
        """
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.dtype)
        if self._graph is not None:
            from ._packed import _product

            result = _product(self._graph, x, self._config, transpose=True)
        elif len(self.blocks) == 1:
            result = self.blocks[0].rmatmat(x)
        else:
            result = jnp.concatenate([block.rmatmat(x) for block in self._blocks], axis=0)
        return result[:, 0] if was_vector else result

    def matvec(self, x: ArrayLike) -> Array:
        """Multiply a vector by the composed genotype matrix."""
        return self.matmat(x)

    def rmatvec(self, x: ArrayLike) -> Array:
        """Multiply a vector by the transpose of the composed matrix."""
        return self.rmatmat(x)

    def __matmul__(self, x: ArrayLike) -> Array:
        return self.matmat(x)


def _backend_for_lineargs(lineargs: tuple[Any, ...], *, backend: Backend) -> Backend:
    requested_backend = Backend(backend)
    if requested_backend is not Backend.AUTO:
        return requested_backend
    if not lineargs or not all(isinstance(linearg, JaxLinearARG) for linearg in lineargs):
        return requested_backend

    prebuilt_backends = {linearg.backend for linearg in lineargs}
    if len(prebuilt_backends) != 1:
        observed = ", ".join(sorted(block_backend.value for block_backend in prebuilt_backends))
        raise ValueError(
            f"prebuilt JaxLinearARG block backends must be consistent when wrapper backend is AUTO; observed {observed}"
        )
    return next(iter(prebuilt_backends))


def _validate_jax_block_settings(
    blocks: tuple[JaxLinearARG, ...],
    *,
    backend: Backend,
) -> None:
    expected_backend = _backend_for_lineargs(blocks, backend=Backend(backend))
    expected_backend = resolve_backend(expected_backend)
    for block in blocks:
        if block.backend is not expected_backend:
            raise ValueError(
                "prebuilt JaxLinearARG block backend must match requested wrapper backend; "
                f"expected {expected_backend.value}, observed {block.backend.value}"
            )


def _require_metadata_columns(metadata: pl.DataFrame, *columns: str) -> None:
    missing = tuple(column for column in columns if column not in metadata.columns)
    if missing:
        expected = ", ".join(f'"{column}"' for column in columns)
        observed = ", ".join(f'"{column}"' for column in metadata.columns)
        raise ValueError(f"metadata must contain columns {expected}; observed columns: {observed}")


def _as_rank2_matrix(x: ArrayLike, *, expected_rows: int, dtype: Any) -> tuple[Array, bool]:
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
