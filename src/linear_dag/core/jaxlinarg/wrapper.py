# pattern: Mixed (unavoidable)
# Reason: Public compatibility constructors coordinate ingress/device placement
# while the operator methods implement pure array algebra.

"""Device-aware composition of ragged JAX LinearARG blocks."""

from contextlib import nullcontext
from functools import cached_property
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jax.sharding import AbstractMesh, Mesh
from jaxtyping import Array, ArrayLike

from linear_dag.core.lineararg import list_blocks

from .operator import _as_rank2_matrix, Backend, JaxLinearARG, resolve_backend


def split_blocks_by_n_entries(metadata: pl.DataFrame, num_devices: int) -> tuple[tuple[int, int], ...]:
    """Balance contiguous block ranges by cumulative graph edge count.

    The result always contains one half-open block-index range per device.
    Empty ranges are possible when there are more devices than blocks.

    **Arguments:**

    - `metadata`: Block metadata containing an `n_entries` column.
    - `num_devices`: Number of device ranges to produce.

    **Returns:**

    - Contiguous `(start, end)` ranges covering every metadata row once.

    **Raises:**

    - `ValueError`: If `num_devices` is not positive or `metadata` lacks the
      required column.
    """
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
    return tuple((int(start), int(end)) for start, end in zip(block_indices[:-1], block_indices[1:], strict=True))


def variant_offsets_from_metadata(metadata: pl.DataFrame) -> np.ndarray:
    """Build leading-zero cumulative variant offsets from block metadata.

    **Arguments:**

    - `metadata`: Block metadata containing an `n_variants` column.

    **Returns:**

    - An `int64` array of length `len(metadata) + 1`.

    **Raises:**

    - `ValueError`: If `metadata` lacks the required column.
    """
    _require_metadata_columns(metadata, "n_variants")
    n_variants = metadata.get_column("n_variants").to_numpy()
    return np.insert(np.cumsum(n_variants), 0, 0).astype(np.int64)


# Equinox field converters keep metadata hashable and normalize NumPy scalar
# inputs before the values become static PyTree leaves.
def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _range_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(start), int(end)) for start, end in values)


class JaxParallelOperator(eqx.Module):
    r"""JAX-compatible multi-block LinearARG operator.

    !!! info
        `mesh` must contain a `"blocks"` axis. A single-device mesh is valid
        and uses the same public API as multi-device execution.

        Concrete meshes place each ragged block on its assigned device and use
        cached exact-shape range programs. Do not wrap a bound multi-block
        `matmat` or `rmatmat` method in an additional `jax.jit`; doing so captures
        the operator arrays as constants and bypasses device-local ownership.

    !!! Example
        ```python
        import jax
        import numpy as np

        from jax.sharding import Mesh
        from linear_dag.core.jaxlinarg import JaxParallelOperator

        mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))
        operator = JaxParallelOperator.from_hdf5("lineararg.h5", mesh=mesh)
        ```

    """

    blocks: tuple[JaxLinearARG, ...] = eqx.field(converter=tuple)
    variant_offsets: tuple[int, ...] = eqx.field(converter=_int_tuple, static=True)
    mesh: Mesh | AbstractMesh = eqx.field(static=True)
    block_ranges: tuple[tuple[int, int], ...] = eqx.field(converter=_range_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=resolve_backend, static=True)

    @classmethod
    def from_linearargs(
        cls,
        lineargs: Any,
        *,
        mesh: Mesh | AbstractMesh,
        backend: Backend = Backend.AUTO,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from LinearARG objects.

        !!! info
            `mesh` must have a `"blocks"` axis. When all inputs are already
            [`linear_dag.core.jaxlinarg.JaxLinearARG`][] objects and `backend`
            is `Backend.AUTO`, the wrapper preserves their common concrete
            backend.

        **Arguments:**

        - `lineargs`: Iterable of LinearARG or JAX LinearARG blocks.
        - `mesh`: JAX mesh with a `"blocks"` axis.
        - `backend`: Requested numerical backend.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxParallelOperator`][].

        **Raises:**

        - `ValueError`: If block settings, shapes, or mesh ranges are invalid.
        - `RuntimeError`: If `Backend.FFI_CPU` is explicitly requested but its
          exact single-block targets are unavailable on the active platform.
        """
        lineargs = tuple(lineargs)
        backend = resolve_backend(_backend_for_lineargs(lineargs, backend=backend))
        metadata = _metadata_from_lineargs(lineargs)
        block_ranges = split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh))
        block_devices = _devices_for_blocks(mesh, block_ranges, n_blocks=len(lineargs))
        blocks = tuple(
            _jax_block_on_device(linearg, device=device, backend=backend)
            for linearg, device in zip(lineargs, block_devices, strict=True)
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            block_ranges=block_ranges,
        )

    @classmethod
    def from_hdf5(
        cls,
        path: Any,
        *,
        mesh: Mesh | AbstractMesh,
        block_metadata: pl.DataFrame | None = None,
        backend: Backend = Backend.AUTO,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from an HDF5 LinearARG file.

        !!! info
            Blocks are assigned to contiguous mesh ranges using HDF5 block
            metadata and created directly on their assigned devices.
            `Backend.AUTO` keeps CPU-only environments usable through FFI or
            pure-JAX fallback.

        **Arguments:**

        - `path`: HDF5 file path.
        - `mesh`: JAX mesh with a `"blocks"` axis.
        - `block_metadata`: Optional preloaded block metadata.
        - `backend`: Requested numerical backend.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxParallelOperator`][].

        **Raises:**

        - `ValueError`: If metadata, block settings, shapes, or mesh ranges are invalid.
        - `RuntimeError`: If `Backend.FFI_CPU` is explicitly requested but its
          exact single-block targets are unavailable on the active platform.
        """
        backend = resolve_backend(backend)
        metadata = list_blocks(path) if block_metadata is None else block_metadata
        block_names = metadata.get_column("block_name").to_list()
        block_ranges = split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh))
        block_devices = _devices_for_blocks(mesh, block_ranges, n_blocks=len(block_names))
        blocks = tuple(
            _hdf5_block_on_device(path, block_name, device=device, backend=backend)
            for block_name, device in zip(block_names, block_devices, strict=True)
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            block_ranges=block_ranges,
        )

    @classmethod
    def from_zarr(
        cls,
        reader: Any,
        *,
        mesh: Mesh | AbstractMesh,
        block_metadata: pl.DataFrame | None = None,
        backend: Backend = Backend.AUTO,
        dtype: Any = None,
    ) -> "JaxParallelOperator":
        """Construct a multi-block JAX operator from a LinearARG Zarr reader.

        !!! info
            Blocks are created directly on their assigned devices so ingress
            does not transiently duplicate the full graph on the default
            device.

        **Arguments:**

        - `reader`: Open [`linear_dag.core.zarr_io.LinearARGZarrReader`][].
        - `mesh`: JAX mesh with a `"blocks"` axis.
        - `block_metadata`: Optional preloaded block metadata.
        - `backend`: Requested numerical backend.
        - `dtype`: Optional computation dtype.

        **Returns:**

        - A [`linear_dag.core.jaxlinarg.JaxParallelOperator`][].

        **Raises:**

        - `ValueError`: If metadata, block settings, shapes, or mesh ranges are invalid.
        - `RuntimeError`: If `Backend.FFI_CPU` is explicitly requested but its
          exact single-block targets are unavailable on the active platform.
        """
        backend = resolve_backend(backend)
        metadata = reader.list_blocks() if block_metadata is None else block_metadata
        block_names = metadata.get_column("block_name").to_list()
        block_ranges = split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh))
        block_devices = _devices_for_blocks(mesh, block_ranges, n_blocks=len(block_names))
        blocks_group = reader.root["blocks"]
        blocks = tuple(
            _zarr_block_on_device(
                blocks_group[block_name],
                device=device,
                backend=backend,
                dtype=dtype,
            )
            for block_name, device in zip(block_names, block_devices, strict=True)
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            block_ranges=block_ranges,
        )

    def __check_init__(self) -> None:
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
        _validate_concrete_block_placement(
            self.blocks,
            mesh=self.mesh,
            block_ranges=self.block_ranges,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the composed operator shape `(n_samples, total_variants)`."""
        return (self.blocks[0].n_samples, self.variant_offsets[-1])

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
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[1], dtype=self.blocks[0].dtype)
        if len(self.blocks) == 1:
            result = self.blocks[0].matmat(x)
        else:
            result = self._device_local_matmat(x)
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
        x, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.blocks[0].dtype)
        if len(self.blocks) == 1:
            result = self.blocks[0].rmatmat(x)
        else:
            result = self._cached_rmatmat(x)
        return result[:, 0] if was_vector else result

    def _device_local_matmat(self, x: Array) -> Array:
        @jax.custom_vjp
        def product(values: Array) -> Array:
            return self._cached_matmat(values)

        def product_fwd(values: Array) -> tuple[Array, None]:
            return self._cached_matmat(values), None

        def product_bwd(_residual: None, cotangent: Array) -> tuple[Array]:
            return (self._cached_rmatmat(cotangent),)

        product.defvjp(product_fwd, product_bwd)
        return product(x)

    def _cached_matmat(self, x: Array) -> Array:
        contributions = [
            _matmat_range_product_jit(
                blocks,
                _device_put_if_needed(x[variant_start:variant_end], device),
            )
            for device, blocks, variant_start, variant_end in self._device_block_ranges
        ]
        assembly_device = _mesh_assembly_device(self.mesh)
        result = _device_put_if_needed(jnp.zeros((self.shape[0], x.shape[1]), dtype=x.dtype), assembly_device)
        for contribution in contributions:
            result = result + _device_put_if_needed(contribution, assembly_device)
        return result

    def _cached_rmatmat(self, x: Array) -> Array:
        # Reverse products naturally return variant rows, and each block range
        # owns a different number of variants. Do not route this through
        # shard_map: shard_map would force equal per-device output shapes and
        # reintroduce padding. Instead, keep one cached JIT entrypoint per
        # non-empty range and concatenate the exact-size range outputs.
        assembly_device = _mesh_assembly_device(self.mesh)
        device_segments = [
            _rmatmat_range_product_jit(blocks, _device_put_if_needed(x, device))
            for device, blocks, _variant_start, _variant_end in self._device_block_ranges
        ]
        segments = [_device_put_if_needed(segment, assembly_device) for segment in device_segments]
        return jnp.concatenate(segments, axis=0)

    @cached_property
    def _device_block_ranges(
        self,
    ) -> tuple[tuple[jax.Device | None, tuple[JaxLinearARG, ...], int, int], ...]:
        devices = _mesh_block_devices(self.mesh)
        ranges = []
        for range_index, (start, end) in enumerate(self.block_ranges):
            if start == end:
                continue
            device = devices[range_index] if devices is not None else None
            ranges.append(
                (
                    device,
                    self.blocks[start:end],
                    self.variant_offsets[start],
                    self.variant_offsets[end],
                )
            )
        return tuple(ranges)

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


def _devices_for_blocks(
    mesh: Mesh | AbstractMesh,
    block_ranges: tuple[tuple[int, int], ...],
    *,
    n_blocks: int,
) -> tuple[jax.Device | None, ...]:
    range_devices = _mesh_block_devices(mesh)
    if range_devices is None:
        return tuple(None for _ in range(n_blocks))

    block_devices: list[jax.Device | None] = [None] * n_blocks
    for device, (start, end) in zip(range_devices, block_ranges, strict=True):
        for block_index in range(start, end):
            block_devices[block_index] = device
    return tuple(block_devices)


def _validate_concrete_block_placement(
    blocks: tuple[JaxLinearARG, ...],
    *,
    mesh: Mesh | AbstractMesh,
    block_ranges: tuple[tuple[int, int], ...],
) -> None:
    if isinstance(mesh, AbstractMesh):
        return

    expected_devices = _devices_for_blocks(mesh, block_ranges, n_blocks=len(blocks))
    for block_index, (block, expected_device) in enumerate(zip(blocks, expected_devices, strict=True)):
        resident_devices = {
            device
            for leaf in jax.tree_util.tree_leaves(block)
            if isinstance(leaf, jax.Array)
            for device in leaf.devices()
        }
        if resident_devices != {expected_device}:
            observed = ", ".join(sorted(str(device) for device in resident_devices)) or "no resident device"
            raise ValueError(
                f"block {block_index} must reside only on assigned device {expected_device}; observed {observed}. "
                "Use JaxParallelOperator.from_linearargs, from_hdf5, or from_zarr to place blocks."
            )


def _put_block_on_device(block: JaxLinearARG, device: jax.Device | None) -> JaxLinearARG:
    if device is None:
        return block
    return eqx.filter_shard(block, device)


# These ingress-specific helpers are deliberately separate: construction must
# happen inside the assigned default-device context. Constructing first and
# moving afterward would transiently duplicate every graph on the default device.
def _jax_block_on_device(
    linearg: Any,
    *,
    device: jax.Device | None,
    backend: Backend,
) -> JaxLinearARG:
    if isinstance(linearg, JaxLinearARG):
        return _put_block_on_device(linearg, device)
    with jax.default_device(device) if device is not None else nullcontext():
        block = JaxLinearARG.from_lineararg(linearg, backend=backend)
    return _put_block_on_device(block, device)


def _hdf5_block_on_device(
    path: Any,
    block_name: Any,
    *,
    device: jax.Device | None,
    backend: Backend,
) -> JaxLinearARG:
    with jax.default_device(device) if device is not None else nullcontext():
        block = JaxLinearARG.from_hdf5_block(path, block_name, backend=backend)
    return _put_block_on_device(block, device)


def _zarr_block_on_device(
    group: Any,
    *,
    device: jax.Device | None,
    backend: Backend,
    dtype: Any,
) -> JaxLinearARG:
    from .ingress import from_zarr_group

    with jax.default_device(device) if device is not None else nullcontext():
        block = from_zarr_group(group, backend=backend, dtype=dtype)
    return _put_block_on_device(block, device)


@eqx.filter_jit
def _matmat_range_product_jit(blocks: tuple[JaxLinearARG, ...], values: Array) -> Array:
    result = jnp.zeros((blocks[0].n_samples, values.shape[1]), dtype=values.dtype)
    variant_start = 0
    for block in blocks:
        variant_end = variant_start + block.n_variants
        result = result + block.matmat(values[variant_start:variant_end])
        variant_start = variant_end
    return result


@eqx.filter_jit
def _rmatmat_range_product_jit(blocks: tuple[JaxLinearARG, ...], values: Array) -> Array:
    return jnp.concatenate([block.rmatmat(values) for block in blocks], axis=0)


def _device_put_if_needed(value: Array, device: jax.Device | None) -> Array:
    if device is None:
        return value
    return jax.device_put(value, device)
