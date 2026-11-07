# pattern: Functional Core

import operator

from collections.abc import Callable, Iterable
from functools import cached_property
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P
from jaxtyping import Array, ArrayLike

from linear_dag.core.lineararg import list_blocks

from ._packed import _ExecutionConfig, _pack
from .operator import Backend, JaxLinearARG, resolve_backend


def _shard_map(
    function: Callable[..., Any],
    *,
    mesh: Mesh | AbstractMesh,
    in_specs: Any,
    out_specs: Any,
    axis_names: set[str],
) -> Callable[..., Any]:
    return jax.shard_map(
        function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        axis_names=axis_names,
    )


def split_blocks_by_n_entries(metadata: pl.DataFrame, num_devices: int) -> tuple[tuple[int, int], ...]:
    """Split contiguous block ranges by cumulative `n_entries` weight."""
    if num_devices < 1:
        raise ValueError(f"num_devices must be positive. Observed {num_devices}.")
    _require_metadata_columns(metadata, "n_entries")

    size_array = metadata.get_column("n_entries").to_numpy()
    size_cumsum = np.insert(np.cumsum(size_array), 0, 0)
    chunk_size = size_cumsum[-1] / num_devices

    block_indices = []
    for i in range(1, num_devices):
        target_sum = i * chunk_size
        idx = np.searchsorted(size_cumsum, target_sum)
        block_indices.append(idx)
    block_indices.append(len(size_cumsum) - 1)

    block_indices = np.array([0] + block_indices)
    return tuple((int(start), int(end)) for start, end in zip(block_indices[:-1], block_indices[1:], strict=False))


def variant_offsets_from_metadata(metadata: pl.DataFrame) -> np.ndarray:
    """Return leading-zero cumulative variant offsets from block metadata."""
    _require_metadata_columns(metadata, "n_variants")
    n_variants = metadata.get_column("n_variants").to_numpy()
    return np.insert(np.cumsum(n_variants), 0, 0).astype(np.int64)


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _range_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(start), int(end)) for start, end in values)


class JaxParallelOperator(eqx.Module):
    r"""JAX-compatible multi-block LinearARG operator.

    !!! info
        CPU FFI packs ragged blocks without padding and runs native workers
        on one CPU device. `num_processes` controls worker threads, not OS
        processes. Scratch space is bounded by `workspace_bytes` (256 MiB by
        default); worker count and RHS tile width are reduced to fit.
        The pure-JAX backend retains mesh-based execution. A supplied `mesh`
        must contain a `"blocks"` axis.

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
    - `mesh`: Optional JAX mesh containing a `"blocks"` axis.
    - `backend`: Requested numerical backend.
    - `block_ranges`: Optional contiguous block ranges assigned to mesh devices.
    - `num_processes`: Maximum native CPU worker threads.
    - `max_num_traits`: Maximum RHS tile width (default 8).
    - `workspace_bytes`: Native scratch budget (default 256 MiB), excluding
      graph/input/output storage and transformation intermediates.
    """

    _blocks: tuple[JaxLinearARG, ...]
    variant_offsets: tuple[int, ...] = eqx.field(converter=_int_tuple, static=True)
    mesh: Mesh | AbstractMesh = eqx.field(static=True)
    block_ranges: tuple[tuple[int, int], ...] = eqx.field(converter=_range_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)

    _graph: tuple[Array, ...] | None
    _config: _ExecutionConfig | None = eqx.field(static=True)
    _specs: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    _dtype: Any = eqx.field(static=True)

    def __init__(
        self,
        blocks: Iterable[JaxLinearARG],
        variant_offsets: Iterable[int],
        mesh: Mesh | AbstractMesh | None = None,
        block_ranges: Iterable[tuple[int, int]] | None = None,
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
        self.mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",)) if mesh is None else mesh
        self.block_ranges = (
            _range_tuple(block_ranges)
            if block_ranges is not None
            else split_blocks_by_n_entries(_metadata_from_lineargs(self._blocks), _mesh_blocks_axis_size(self.mesh))
        )
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
            if isinstance(self.mesh, AbstractMesh) or self.mesh.devices.size != 1:
                raise ValueError("Packed CPU execution uses one device; use num_processes for worker parallelism")
            if self.mesh.devices.flat[0].platform != "cpu":
                raise ValueError("Packed CPU execution requires a CPU device")
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
            with jax.default_device(self.mesh.devices.flat[0]):
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
        lineargs: Any,
        *,
        mesh: Mesh | AbstractMesh | None = None,
        backend: Backend = Backend.AUTO,
        num_processes: int | None = None,
        max_num_traits: int = 8,
        workspace_bytes: int = 256 * 1024**2,
        dtype: Any = jnp.float32,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from LinearARG objects.

        !!! info
            `mesh` must have a `"blocks"` axis. When all inputs are already
            [`linear_dag.core.jaxlinarg.JaxLinearARG`][] objects and `backend`
            is `Backend.AUTO`, the wrapper preserves their common concrete
            backend.

        **Arguments:**

        - `lineargs`: Iterable of LinearARG or JAX LinearARG blocks.
        - `mesh`: Optional JAX mesh with a `"blocks"` axis; defaults to one device.
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

        - `ValueError`: If block settings, shapes, or mesh ranges are invalid.
        """
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",)) if mesh is None else mesh
        lineargs = tuple(lineargs)
        backend = _backend_for_lineargs(lineargs, backend=backend)
        metadata = _metadata_from_lineargs(lineargs)
        blocks = tuple(
            linearg
            if isinstance(linearg, JaxLinearARG)
            else JaxLinearARG.from_lineararg(linearg, backend=backend, dtype=dtype)
            for linearg in lineargs
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            block_ranges=split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh)),
            num_processes=num_processes,
            max_num_traits=max_num_traits,
            workspace_bytes=workspace_bytes,
        )

    @classmethod
    def from_hdf5(
        cls,
        path: Any,
        *,
        mesh: Mesh | AbstractMesh | None = None,
        block_metadata: pl.DataFrame | None = None,
        backend: Backend = Backend.AUTO,
        num_processes: int | None = None,
        max_num_traits: int = 8,
        workspace_bytes: int = 256 * 1024**2,
        dtype: Any = jnp.float32,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from an HDF5 LinearARG file.

        !!! info
            Blocks are assigned to contiguous mesh ranges using HDF5 block
            metadata. `Backend.AUTO` keeps CPU-only environments usable through
            FFI or pure-JAX fallback.

        **Arguments:**

        - `path`: HDF5 file path.
        - `mesh`: Optional JAX mesh with a `"blocks"` axis; defaults to one device.
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

        - `ValueError`: If metadata, block settings, shapes, or mesh ranges are invalid.
        """
        mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",)) if mesh is None else mesh
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
            mesh=mesh,
            backend=backend,
            block_ranges=split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh)),
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
        _validate_block_ranges(
            self.block_ranges,
            n_blocks=len(self.blocks),
            n_mesh_blocks=_mesh_blocks_axis_size(self.mesh),
        )

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
        elif len(self.block_ranges) > 1:
            result = self._sharded_matmat(x)
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
            result = self._cached_rmatmat(x)
        return result[:, 0] if was_vector else result

    def _sharded_matmat(self, x: Array) -> Array:
        @jax.custom_vjp
        def product(values: Array) -> Array:
            return self._sharded_matmat_primal(values)

        def product_fwd(values: Array) -> tuple[Array, None]:
            return self._sharded_matmat_primal(values), None

        def product_bwd(_residual: None, cotangent: Array) -> tuple[Array]:
            return (self._cached_rmatmat(cotangent),)

        product.defvjp(product_fwd, product_bwd)
        return product(x)

    def _sharded_matmat_primal(self, x: Array) -> Array:
        branches = tuple(self._matmat_branch(start, end) for start, end in self.block_ranges)

        def mapped(values: Array) -> Array:
            axis_index = jax.lax.axis_index("blocks")
            local = jax.lax.switch(axis_index, branches, values)
            # Each device owns a contiguous subset of variant blocks and returns
            # a sample-sized partial product. The full matmat is their sum.
            return jax.lax.psum(local, "blocks")

        product = _shard_map(
            mapped,
            mesh=self.mesh,
            in_specs=P(),
            # `psum` leaves each device with the full sample-sized product.
            # Returning a replicated value avoids forcing XLA to slice and
            # exchange an already-replicated result into a sharded layout.
            out_specs=P(),
            axis_names={"blocks"},
        )
        return product(x)

    def _matmat_branch(self, start: int, end: int) -> Callable[[Array], Array]:
        def branch(values: Array) -> Array:
            local = jnp.zeros((self.shape[0], values.shape[1]), dtype=values.dtype)
            for block_index in range(start, end):
                block_start = self.variant_offsets[block_index]
                block_end = self.variant_offsets[block_index + 1]
                local = local + self.blocks[block_index].matmat(values[block_start:block_end])
            return local

        return branch

    def _cached_rmatmat(self, x: Array) -> Array:
        # Reverse products naturally return variant rows, and each block range
        # owns a different number of variants. Do not route this through
        # shard_map: shard_map would force equal per-device output shapes and
        # reintroduce padding. Instead, keep one cached JIT entrypoint per
        # non-empty range and concatenate the exact-size range outputs.
        assembly_device = _mesh_assembly_device(self.mesh)
        segments = [
            _device_put_if_needed(product(_device_put_if_needed(x, device)), assembly_device)
            for device, product in self._rmatmat_products
        ]
        return jnp.concatenate(segments, axis=0)

    @cached_property
    def _rmatmat_products(self) -> tuple[tuple[jax.Device | None, Callable[[Array], Array]], ...]:
        devices = _mesh_block_devices(self.mesh)
        products = []
        for range_index, (start, end) in enumerate(self.block_ranges):
            if start == end:
                continue
            device = devices[range_index] if devices is not None else None
            products.append((device, jax.jit(_rmatmat_range_product(self.blocks[start:end]))))
        return tuple(products)

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


def _metadata_from_lineargs(lineargs: tuple[Any, ...]) -> pl.DataFrame:
    # Accept both already-converted JAX blocks and source LinearARG blocks so
    # callers can build wrappers before or after per-block conversion.
    n_entries = []
    n_variants = []
    n_samples = []
    n_nodes = []
    for linearg in lineargs:
        if isinstance(linearg, JaxLinearARG):
            n_entries.append(int(linearg.indices.shape[0]))
            n_variants.append(int(linearg.n_variants))
            n_samples.append(int(linearg.n_samples))
            n_nodes.append(int(linearg.indptr.shape[0] - 1))
        else:
            n_entries.append(int(linearg.A.nnz))
            n_variants.append(int(linearg.shape[1]))
            n_samples.append(int(linearg.shape[0]))
            n_nodes.append(int(linearg.A.shape[0]))

    return pl.DataFrame(
        {
            "n_entries": n_entries,
            "n_variants": n_variants,
            "n_samples": n_samples,
            "n": n_nodes,
        }
    )


def _require_metadata_columns(metadata: pl.DataFrame, *columns: str) -> None:
    missing = tuple(column for column in columns if column not in metadata.columns)
    if missing:
        expected = ", ".join(f'"{column}"' for column in columns)
        observed = ", ".join(f'"{column}"' for column in metadata.columns)
        raise ValueError(f"metadata must contain columns {expected}; observed columns: {observed}")


def _validate_mesh(mesh: Mesh | AbstractMesh) -> None:
    if not isinstance(mesh, (Mesh, AbstractMesh)):
        raise TypeError("mesh must be a jax.sharding.Mesh or jax.sharding.AbstractMesh")
    if isinstance(mesh, AbstractMesh):
        device_count = int(np.prod(tuple(mesh.shape.values()), dtype=np.int64))
    else:
        device_count = int(np.asarray(getattr(mesh, "devices", ())).size)
    if device_count < 1:
        raise ValueError("mesh must contain at least one device")
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    if axis_names.count("blocks") != 1:
        raise ValueError('mesh axis names must include exactly one "blocks" axis')


def _validate_block_ranges(
    block_ranges: tuple[tuple[int, int], ...],
    *,
    n_blocks: int,
    n_mesh_blocks: int,
) -> None:
    if len(block_ranges) != n_mesh_blocks:
        raise ValueError(
            "block_ranges length must match the mesh blocks axis size; "
            f"observed {len(block_ranges)} ranges for axis size {n_mesh_blocks}"
        )

    expected_start = 0
    for start, end in block_ranges:
        if start < 0 or end < start or end > n_blocks:
            raise ValueError("block_ranges must be valid block index ranges")
        if start != expected_start:
            raise ValueError("block_ranges must be contiguous and cover every block")
        expected_start = end
    if expected_start != n_blocks:
        raise ValueError("block_ranges must be contiguous and cover every block")


def _mesh_blocks_axis_size(mesh: Mesh | AbstractMesh) -> int:
    _validate_mesh(mesh)
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    if isinstance(mesh, AbstractMesh):
        return int(mesh.shape["blocks"])
    devices = np.asarray(mesh.devices)
    axis_index = axis_names.index("blocks")
    if devices.ndim <= axis_index:
        return int(devices.size)
    return int(devices.shape[axis_index])


def _mesh_block_devices(mesh: Mesh | AbstractMesh) -> tuple[jax.Device, ...] | None:
    if isinstance(mesh, AbstractMesh):
        return None
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    devices = np.asarray(mesh.devices)
    axis_index = axis_names.index("blocks")
    if devices.ndim <= axis_index:
        return tuple(devices.reshape(-1).tolist())
    moved = np.moveaxis(devices, axis_index, 0)
    return tuple(moved.reshape((moved.shape[0], -1))[:, 0].tolist())


def _mesh_assembly_device(mesh: Mesh | AbstractMesh) -> jax.Device | None:
    if isinstance(mesh, AbstractMesh):
        return None
    return np.asarray(mesh.devices).reshape(-1).tolist()[0]


def _rmatmat_range_product(blocks: tuple[JaxLinearARG, ...]) -> Callable[[Array], Array]:
    # A range product is deliberately exact-size. This mirrors the NumPy
    # ParallelOperator model: each worker/range computes its own variant rows,
    # and the wrapper assembles the public `(sum p_i, k)` result by
    # concatenation. Keeping the function object cached lets JAX reuse compiled
    # executables for repeated calls with the same input shape and dtype.
    def product(values: Array) -> Array:
        return jnp.concatenate([block.rmatmat(values) for block in blocks], axis=0)

    return product


def _device_put_if_needed(value: Array, device: jax.Device | None) -> Array:
    if device is None:
        return value
    return jax.device_put(value, device)


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
