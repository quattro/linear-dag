# pattern: Functional Core

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P

from linear_dag.core.lineararg import list_blocks

from .operator import Backend, JaxLinearARG, resolve_backend
from .padding import BucketSpec, choose_bucket, choose_buckets

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
    - `level_schedule`: Whether block-level Pallas GPU dispatch should use
      precomputed level schedules.
    - `block_ranges`: Contiguous block ranges assigned to mesh devices.
    """

    blocks: tuple[JaxLinearARG, ...] = eqx.field(converter=tuple)
    variant_offsets: tuple[int, ...] = eqx.field(converter=_int_tuple, static=True)
    mesh: Any = eqx.field(static=True)
    block_ranges: tuple[tuple[int, int], ...] = eqx.field(converter=_range_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)
    level_schedule: bool = eqx.field(default=False, converter=bool, static=True)

    @classmethod
    def from_linearargs(
        cls,
        lineargs: Any,
        *,
        mesh: Any,
        backend: Backend = Backend.AUTO,
        buckets: Any = "auto",
        level_schedule: bool = False,
    ) -> "JaxParallelOperator":
        lineargs = tuple(lineargs)
        backend = _backend_for_lineargs(lineargs, backend=backend)
        metadata = _metadata_from_lineargs(lineargs)
        blocks = tuple(
            _as_jax_block(
                linearg,
                backend=backend,
                bucket=bucket,
                level_schedule=level_schedule,
            )
            for linearg, bucket in _zip_buckets(
                lineargs,
                buckets,
                shapes=_bucket_shapes_from_metadata(metadata),
            )
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            level_schedule=level_schedule,
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
        level_schedule: bool = False,
    ) -> "JaxParallelOperator":
        metadata = list_blocks(path) if block_metadata is None else block_metadata
        block_names = metadata.get_column("block_name").to_list()
        blocks = tuple(
            JaxLinearARG.from_hdf5_block(
                path,
                block_name,
                backend=backend,
                bucket=bucket,
                level_schedule=level_schedule,
            )
            for block_name, bucket in _zip_buckets(
                block_names,
                buckets,
                shapes=_bucket_shapes_from_metadata(metadata),
            )
        )
        return cls(
            blocks=blocks,
            variant_offsets=variant_offsets_from_metadata(metadata),
            mesh=mesh,
            backend=backend,
            level_schedule=level_schedule,
            block_ranges=split_blocks_by_n_entries(metadata, _mesh_blocks_axis_size(mesh)),
        )

    def __check_init__(self) -> None:
        if not self.blocks:
            raise ValueError("blocks must contain at least one JaxLinearARG")
        _validate_mesh(self.mesh)
        n_samples = {block.n_samples for block in self.blocks}
        if len(n_samples) != 1:
            raise ValueError("all blocks must have the same n_samples")
        dtypes = {jnp.dtype(block.dtype) for block in self.blocks}
        if len(dtypes) != 1:
            raise ValueError("all blocks must have the same dtype")
        _validate_jax_block_settings(
            self.blocks,
            backend=self.backend,
            level_schedule=self.level_schedule,
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
        expected_variants = sum(block.n_variants for block in self.blocks)
        if self.variant_offsets[-1] != expected_variants:
            raise ValueError("final variant_offsets entry must match total block variants")
        _validate_block_ranges(
            self.block_ranges,
            n_blocks=len(self.blocks),
            n_mesh_blocks=_mesh_blocks_axis_size(self.mesh),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.blocks[0].n_samples, self.variant_offsets[-1])

    def matmat(self, x: Any) -> Any:
        matrix, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[1], dtype=self.blocks[0].dtype)
        result = self._matmat(matrix)
        return result[:, 0] if was_vector else result

    def rmatmat(self, x: Any) -> Any:
        matrix, was_vector = _as_rank2_matrix(x, expected_rows=self.shape[0], dtype=self.blocks[0].dtype)
        result = self._rmatmat(matrix)
        return result[:, 0] if was_vector else result

    def _matmat(self, x: Any) -> Any:
        if _mesh_blocks_axis_size(self.mesh) > 1:
            return self._sharded_matmat(x)
        contributions = [
            block.matmat(x[start:end])
            for block, start, end in zip(
                self.blocks,
                self.variant_offsets[:-1],
                self.variant_offsets[1:],
                strict=True,
            )
        ]
        return jnp.sum(jnp.stack(contributions, axis=0), axis=0)

    def _rmatmat(self, x: Any) -> Any:
        if _mesh_blocks_axis_size(self.mesh) > 1:
            return self._sharded_rmatmat(x)
        return jnp.concatenate([block.rmatmat(x) for block in self.blocks], axis=0)

    def _sharded_matmat(self, x: Any) -> Any:
        @jax.custom_vjp
        def product(values: Any) -> Any:
            return self._sharded_matmat_primal(values)

        def product_fwd(values: Any) -> tuple[Any, None]:
            return self._sharded_matmat_primal(values), None

        def product_bwd(_residual: None, cotangent: Any) -> tuple[Any]:
            return (self._sharded_rmatmat(cotangent),)

        product.defvjp(product_fwd, product_bwd)
        return product(x)

    def _sharded_matmat_primal(self, x: Any) -> Any:
        branches = tuple(self._matmat_branch(start, end) for start, end in self.block_ranges)

        def mapped(values: Any) -> Any:
            axis_index = jax.lax.axis_index("blocks")
            local = jax.lax.switch(axis_index, branches, values)
            return jax.lax.psum(local, "blocks")

        product = jax.shard_map(
            mapped,
            mesh=self.mesh,
            in_specs=P(),
            out_specs=P("blocks"),
            axis_names={"blocks"},
        )
        stacked_total = product(x)
        return stacked_total[: self.shape[0]]

    def _matmat_branch(self, start: int, end: int) -> Any:
        def branch(values: Any) -> Any:
            local = jnp.zeros((self.shape[0], values.shape[1]), dtype=values.dtype)
            for block_index in range(start, end):
                block_start = self.variant_offsets[block_index]
                block_end = self.variant_offsets[block_index + 1]
                local = local + self.blocks[block_index].matmat(values[block_start:block_end])
            return local

        return branch

    def _sharded_rmatmat(self, x: Any) -> Any:
        device_variant_counts = tuple(
            int(self.variant_offsets[end] - self.variant_offsets[start]) for start, end in self.block_ranges
        )
        max_device_variants = max(device_variant_counts)
        branches = tuple(self._rmatmat_branch(start, end, max_device_variants) for start, end in self.block_ranges)

        def mapped(values: Any) -> Any:
            axis_index = jax.lax.axis_index("blocks")
            return jax.lax.switch(axis_index, branches, values)

        product = jax.shard_map(
            mapped,
            mesh=self.mesh,
            in_specs=P(),
            out_specs=P("blocks"),
            axis_names={"blocks"},
        )
        padded_segments = product(x)
        segments = []
        for device_index, variant_count in enumerate(device_variant_counts):
            start = device_index * max_device_variants
            segments.append(padded_segments[start : start + variant_count])
        return jnp.concatenate(segments, axis=0)

    def _rmatmat_branch(self, start: int, end: int, max_device_variants: int) -> Any:
        def branch(values: Any) -> Any:
            local_blocks = [self.blocks[block_index].rmatmat(values) for block_index in range(start, end)]
            if local_blocks:
                local = jnp.concatenate(local_blocks, axis=0)
            else:
                local = jnp.zeros((0, values.shape[1]), dtype=values.dtype)
            padding = max_device_variants - local.shape[0]
            return jnp.pad(local, ((0, padding), (0, 0)))

        return branch

    def matvec(self, x: Any) -> Any:
        return self.matmat(x)

    def rmatvec(self, x: Any) -> Any:
        return self.rmatmat(x)

    def __matmul__(self, x: Any) -> Any:
        return self.matmat(x)


def _zip_buckets(
    values: Any,
    buckets: Any,
    *,
    shapes: tuple[BucketSpec, ...] | None = None,
) -> tuple[tuple[Any, Any], ...]:
    values = tuple(values)
    if buckets == "auto":
        if shapes is None:
            raise ValueError('buckets="auto" requires block shapes')
        chosen_buckets = choose_buckets(shapes)
        bucket_values = tuple(choose_bucket(shape, chosen_buckets) for shape in shapes)
        return tuple(zip(values, bucket_values, strict=True))
    if buckets is None:
        return tuple((value, None) for value in values)
    shared_bucket = _as_single_bucket_spec(buckets)
    if shared_bucket is not None:
        return tuple((value, shared_bucket) for value in values)
    if isinstance(buckets, (str, bytes)):
        raise ValueError("buckets must be 'auto', None, a bucket spec, or one bucket per block")
    try:
        bucket_values = tuple(buckets)
    except TypeError:
        return tuple((value, buckets) for value in values)
    if len(bucket_values) != len(values):
        raise ValueError("buckets must contain one entry per block")
    return tuple(zip(values, bucket_values, strict=True))


def _as_jax_block(
    linearg: Any,
    *,
    backend: Backend,
    bucket: Any,
    level_schedule: bool,
) -> JaxLinearARG:
    if isinstance(linearg, JaxLinearARG):
        _validate_jax_block_settings(
            (linearg,),
            backend=backend,
            level_schedule=level_schedule,
        )
        return linearg
    return JaxLinearARG.from_lineararg(
        linearg,
        backend=backend,
        bucket=bucket,
        level_schedule=level_schedule,
    )


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
            "prebuilt JaxLinearARG block backends must be consistent when wrapper backend is AUTO; "
            f"observed {observed}"
        )
    return next(iter(prebuilt_backends))


def _validate_jax_block_settings(
    blocks: tuple[JaxLinearARG, ...],
    *,
    backend: Backend,
    level_schedule: bool,
) -> None:
    requested_backend = Backend(backend)
    if requested_backend is Backend.AUTO:
        expected_backend = _backend_for_lineargs(blocks, backend=requested_backend)
        if expected_backend is Backend.AUTO:
            expected_backend = resolve_backend(expected_backend)
    else:
        expected_backend = resolve_backend(requested_backend)
    expected_level_schedule = bool(level_schedule)
    for block in blocks:
        if block.backend is not expected_backend:
            raise ValueError(
                "prebuilt JaxLinearARG block backend must match requested wrapper backend; "
                f"expected {expected_backend.value}, observed {block.backend.value}"
            )
        if block.level_schedule is not expected_level_schedule:
            raise ValueError(
                "prebuilt JaxLinearARG block level_schedule must match requested wrapper level_schedule; "
                f"expected {expected_level_schedule}, observed {block.level_schedule}"
            )


def _metadata_from_lineargs(lineargs: tuple[Any, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "n_entries": [_edge_count(linearg) for linearg in lineargs],
            "n_variants": [_n_variants(linearg) for linearg in lineargs],
            "n_samples": [_n_samples(linearg) for linearg in lineargs],
            "n": [_node_count(linearg) for linearg in lineargs],
        }
    )


def _bucket_shapes_from_metadata(metadata: pl.DataFrame) -> tuple[BucketSpec, ...]:
    if "n" not in metadata.columns:
        raise ValueError('metadata must contain an "n" column for bucket assignment')
    return tuple(
        BucketSpec(max_nodes=int(n_nodes), max_nnz=int(n_entries))
        for n_nodes, n_entries in zip(
            metadata.get_column("n").to_list(),
            metadata.get_column("n_entries").to_list(),
            strict=True,
        )
    )


def _node_count(linearg: Any) -> int:
    if isinstance(linearg, JaxLinearARG):
        return int(linearg.indptr.shape[0] - 1)
    return int(linearg.A.shape[0])


def _edge_count(linearg: Any) -> int:
    if isinstance(linearg, JaxLinearARG):
        return int(linearg.indices.shape[0])
    return int(linearg.A.nnz)


def _n_variants(linearg: Any) -> int:
    if isinstance(linearg, JaxLinearARG):
        return int(linearg.n_variants)
    return int(linearg.shape[1])


def _n_samples(linearg: Any) -> int:
    if isinstance(linearg, JaxLinearARG):
        return int(linearg.n_samples)
    return int(linearg.shape[0])


def _as_single_bucket_spec(bucket: Any) -> BucketSpec | None:
    if isinstance(bucket, BucketSpec):
        return bucket
    if isinstance(bucket, (str, bytes)):
        return None
    try:
        values = tuple(bucket)
    except TypeError:
        return None
    if len(values) != 2 or not all(_is_int_like(value) for value in values):
        return None
    return BucketSpec(max_nodes=int(values[0]), max_nnz=int(values[1]))


def _is_int_like(value: Any) -> bool:
    return isinstance(value, (int, np.integer))


def _validate_mesh(mesh: Any) -> None:
    if not isinstance(mesh, (Mesh, AbstractMesh)):
        raise TypeError("mesh must be a jax.sharding.Mesh or jax.sharding.AbstractMesh")
    if _mesh_device_count(mesh) < 1:
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


def _mesh_device_count(mesh: Any) -> int:
    if isinstance(mesh, AbstractMesh):
        return int(np.prod(tuple(mesh.shape.values()), dtype=np.int64))
    return int(np.asarray(getattr(mesh, "devices", ())).size)


def _mesh_blocks_axis_size(mesh: Any) -> int:
    _validate_mesh(mesh)
    axis_names = tuple(getattr(mesh, "axis_names", ()))
    if isinstance(mesh, AbstractMesh):
        return int(mesh.shape["blocks"])
    devices = np.asarray(mesh.devices)
    axis_index = axis_names.index("blocks")
    if devices.ndim <= axis_index:
        return int(devices.size)
    return int(devices.shape[axis_index])


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
