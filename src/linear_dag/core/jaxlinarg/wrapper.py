# pattern: Functional Core

from typing import Any

import equinox as eqx
import numpy as np
import polars as pl

from linear_dag.core.lineararg import list_blocks

from .operator import Backend, JaxLinearARG

_KERNELS_MESSAGE = "JaxLinearARG numerical kernels are implemented in Phase 2"


def split_blocks_by_n_entries(metadata: pl.DataFrame, num_devices: int) -> tuple[tuple[int, int], ...]:
    """Split contiguous block ranges by cumulative `n_entries` weight."""
    if num_devices < 1:
        raise ValueError(f"num_devices must be positive. Observed {num_devices}.")

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
    n_variants = metadata.get_column("n_variants").to_numpy()
    return np.insert(np.cumsum(n_variants), 0, 0).astype(np.int64)


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _range_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(start), int(end)) for start, end in values)


class JaxParallelOperator(eqx.Module):
    r"""JAX-compatible multi-block LinearARG operator.

    !!! info
        `mesh` must contain a `"blocks"` axis. A single-device mesh is valid
        and uses the same public API as multi-device execution.

    !!! Example
        ```python
        import jax
        import numpy as np

        from jax.sharding import Mesh
        from linear_dag.core.jaxlinarg import JaxParallelOperator

        mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))
        operator = JaxParallelOperator.from_hdf5("lineararg.h5", mesh=mesh)
        ```

    **Arguments:**

    - `blocks`: Block operators composed by this wrapper.
    - `variant_offsets`: Leading-zero cumulative variant offsets.
    - `mesh`: JAX mesh containing a `"blocks"` axis.
    - `backend`: Requested numerical backend.
    - `block_ranges`: Contiguous block ranges assigned to mesh devices.
    """

    blocks: tuple[JaxLinearARG, ...] = eqx.field(converter=tuple)
    variant_offsets: tuple[int, ...] = eqx.field(converter=_int_tuple, static=True)
    mesh: Any = eqx.field(static=True)
    block_ranges: tuple[tuple[int, int], ...] = eqx.field(converter=_range_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)

    @classmethod
    def from_linearargs(
        cls,
        lineargs: Any,
        *,
        mesh: Any,
        backend: Backend = Backend.AUTO,
        buckets: Any = "auto",
    ) -> "JaxParallelOperator":
        blocks = tuple(
            _as_jax_block(linearg, backend=backend, bucket=bucket)
            for linearg, bucket in _zip_buckets(lineargs, buckets)
        )
        metadata = _metadata_from_blocks(blocks)
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            block_ranges=split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh)),
        )

    @classmethod
    def from_hdf5(
        cls,
        path: Any,
        *,
        mesh: Any,
        block_metadata: pl.DataFrame | None = None,
        backend: Backend = Backend.AUTO,
        buckets: Any = "auto",
    ) -> "JaxParallelOperator":
        metadata = list_blocks(path) if block_metadata is None else block_metadata
        block_names = metadata.get_column("block_name").to_list()
        blocks = tuple(
            JaxLinearARG.from_hdf5_block(path, block_name, backend=backend, bucket=bucket)
            for block_name, bucket in _zip_buckets(block_names, buckets)
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            block_ranges=split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh)),
        )

    def __check_init__(self) -> None:
        if not self.blocks:
            raise ValueError("blocks must contain at least one JaxLinearARG")
        _validate_mesh(self.mesh)
        n_samples = {block.n_samples for block in self.blocks}
        if len(n_samples) != 1:
            raise ValueError("all blocks must have the same n_samples")

        if len(self.variant_offsets) != len(self.blocks) + 1:
            raise ValueError("variant_offsets length must be n_blocks + 1")
        if self.variant_offsets[0] != 0:
            raise ValueError("variant_offsets must start at zero")
        offsets = np.asarray(self.variant_offsets, dtype=np.int64)
        if np.any(np.diff(offsets) < 0):
            raise ValueError("variant_offsets must be monotone")
        expected_variants = sum(block.n_variants for block in self.blocks)
        if self.variant_offsets[-1] != expected_variants:
            raise ValueError("final variant_offsets entry must match total block variants")
        for start, end in self.block_ranges:
            if start < 0 or end < start or end > len(self.blocks):
                raise ValueError("block_ranges must be valid block index ranges")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.blocks[0].n_samples, self.variant_offsets[-1])

    def matmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def rmatmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)


def _zip_buckets(values: Any, buckets: Any) -> tuple[tuple[Any, Any], ...]:
    values = tuple(values)
    if buckets == "auto" or buckets is None:
        return tuple((value, None) for value in values)
    if isinstance(buckets, (str, bytes)):
        raise ValueError("buckets must be 'auto', None, a bucket spec, or one bucket per block")
    try:
        bucket_values = tuple(buckets)
    except TypeError:
        return tuple((value, buckets) for value in values)
    if len(bucket_values) != len(values):
        raise ValueError("buckets must contain one entry per block")
    return tuple(zip(values, bucket_values, strict=True))


def _as_jax_block(linearg: Any, *, backend: Backend, bucket: Any) -> JaxLinearARG:
    if isinstance(linearg, JaxLinearARG):
        return linearg
    return JaxLinearARG.from_lineararg(linearg, backend=backend, bucket=bucket)


def _metadata_from_blocks(blocks: tuple[JaxLinearARG, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "n_entries": [int(block.data.shape[0]) for block in blocks],
            "n_variants": [block.n_variants for block in blocks],
            "n_samples": [block.n_samples for block in blocks],
        }
    )


def _validate_mesh(mesh: Any) -> None:
    if _mesh_device_count(mesh) < 1:
        raise ValueError("mesh must contain at least one device")
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    if axis_names.count("blocks") != 1:
        raise ValueError('mesh axis names must include exactly one "blocks" axis')


def _mesh_device_count(mesh: Any) -> int:
    return int(np.asarray(getattr(mesh, "devices", ())).size)


def _mesh_blocks_axis_size(mesh: Any) -> int:
    _validate_mesh(mesh)
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    devices = np.asarray(mesh.devices)
    axis_index = axis_names.index("blocks")
    if devices.ndim <= axis_index:
        return int(devices.size)
    return int(devices.shape[axis_index])
