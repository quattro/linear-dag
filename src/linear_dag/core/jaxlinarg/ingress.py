# pattern: Imperative Shell

"""Host-to-device ingress for LinearARG arrays, HDF5 blocks, and Zarr groups."""

from __future__ import annotations

import warnings

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module
from importlib.util import find_spec
from os import PathLike
from typing import Any

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import Array
from scipy import sparse

from linear_dag.core.lineararg import LinearARG

from .operator import Backend, JaxLinearARG, resolve_backend
from .packing import (
    _allocate_host_buffers,
    _block_metrics,
    _block_packing_summary_from_arrays,
    _BlockPackingSummary,
    _finalize_staged_descriptor,
    _make_packed_graph_value,
    _pack_block_into_buffers,
    _packed_graph_component,
    _PackedGraphLogicalMetadata,
    _plan_packing_from_summaries,
    canonicalize_block_arrays,
    GRAPH_FIELD_NAMES,
    LinearARGBlockArrays,
    PACKED_COMPONENT_NAMES,
    PackedGraph,
    PackingPlan,
    VALID_LENGTH_FIELDS,
    validate_packed_graph,
)

_STAGING_ACCOUNTING_NOTE = "deterministic one-source-block ingress accounting; not a JAX allocator high-water mark"
_ROOT_GRAPH_DATASETS = frozenset(("indptr", "indices", "data", "variant_indices", "flip"))
_ROOT_GRAPH_ATTRIBUTES = frozenset(("n", "n_samples", "n_variants", "n_entries"))


def _resolve_packed_backend(requested: Backend) -> Backend:
    return resolve_backend(requested, require_packed_targets=True)


@dataclass(frozen=True)
class _PackedIngressDiagnostics:
    """Host-staging and final device-residency accounting for packed ingress.

    `staging_bytes` is a deterministic one-source-block ingress peak derived
    from canonical source arrays. It is not a JAX allocator high-water mark.
    """

    canonical_graph_bytes: int
    padded_graph_bytes: int
    descriptor_bytes: int
    staging_bytes: int
    staging_bytes_by_device: tuple[int, ...]
    staging_block_owners: tuple[int, ...]
    final_graph_bytes_by_device: tuple[int, ...]
    final_bytes_by_device: tuple[int, ...]
    padding_ratio: float
    max_padding_ratio: float | None
    component_count: int
    pytree_leaf_count: int
    staging_accounting: str = _STAGING_ACCOUNTING_NOTE


class _PackedJaxLinearARG(eqx.Module):
    """Private convenience carrier around one opaque packed graph value."""

    n_samples: int = eqx.field(static=True)
    n_variants: int = eqx.field(static=True)
    capacities: tuple[int, ...] = eqx.field(static=True)
    graph_mesh: Mesh = eqx.field(static=True)
    graph: Any
    backend: Backend = eqx.field(
        default=Backend.PURE_JAX,
        converter=partial(resolve_backend, require_packed_targets=True),
        static=True,
    )
    iids: tuple[str, ...] | None = eqx.field(default=None, converter=lambda value: _iid_tuple(value), static=True)

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
        nonunique_indices: Any,
        n_variants: int,
        n_samples: int,
        allele_counts: Any = None,
        iids: Any = None,
        mesh: Mesh | None = None,
        backend: Backend = Backend.AUTO,
        dtype: Any = None,
        max_padding_ratio: float | None = 1.25,
    ) -> _PackedJaxLinearARG:
        """Construct the private candidate from one block of host arrays."""
        block = LinearARGBlockArrays(
            indptr=np.asarray(indptr),
            indices=np.asarray(indices),
            data=np.asarray(data),
            variant_indices=np.asarray(variant_indices),
            flip=np.asarray(flip),
            sample_indices=np.asarray(sample_indices),
            nonunique_indices=None if nonunique_indices is None else np.asarray(nonunique_indices),
            allele_counts=None if allele_counts is None else np.asarray(allele_counts),
            n_variants=n_variants,
            n_samples=n_samples,
        )
        return _packed_from_block_arrays(
            (block,),
            mesh=_single_block_mesh(mesh),
            backend=backend,
            dtype=dtype,
            max_padding_ratio=max_padding_ratio,
            iids=iids,
        ).operator

    @classmethod
    def from_lineararg(
        cls,
        linarg: LinearARG,
        *,
        mesh: Mesh | None = None,
        backend: Backend = Backend.AUTO,
        dtype: Any = None,
        max_padding_ratio: float | None = 1.25,
    ) -> _PackedJaxLinearARG:
        """Construct the private candidate from one in-memory LinearARG."""
        return _packed_from_block_arrays(
            (_lineararg_block_arrays(linarg, dtype=dtype),),
            mesh=_single_block_mesh(mesh),
            backend=backend,
            dtype=dtype,
            max_padding_ratio=max_padding_ratio,
            iids=getattr(linarg, "iids", None),
        ).operator

    @classmethod
    def from_linearargs(
        cls,
        lineargs: Iterable[LinearARG],
        *,
        mesh: Mesh,
        backend: Backend = Backend.AUTO,
        dtype: Any = None,
        max_padding_ratio: float | None = 1.25,
    ) -> _PackedJaxLinearARG:
        """Construct the private candidate from in-memory LinearARG blocks."""
        sources = tuple(lineargs)
        iids = _shared_lineararg_iids(sources)
        blocks = (_lineararg_block_arrays(linarg, dtype=dtype) for linarg in sources)
        return _packed_from_block_arrays(
            blocks,
            mesh=mesh,
            backend=backend,
            dtype=dtype,
            max_padding_ratio=max_padding_ratio,
            iids=iids,
        ).operator

    @classmethod
    def from_hdf5_block(
        cls,
        path: str | PathLike[str],
        block: Any = None,
        *,
        mesh: Mesh | None = None,
        backend: Backend = Backend.AUTO,
        load_metadata: bool = False,
        dtype: Any = None,
        max_padding_ratio: float | None = 1.25,
    ) -> _PackedJaxLinearARG:
        """Construct the private candidate from one HDF5 block or root file."""
        del load_metadata
        return _packed_from_hdf5(
            path,
            None if block is None else (block,),
            mesh=_single_block_mesh(mesh),
            backend=backend,
            dtype=dtype,
            max_padding_ratio=max_padding_ratio,
            root_only=block is None,
        ).operator

    @classmethod
    def from_hdf5(
        cls,
        path: str | PathLike[str],
        *,
        mesh: Mesh,
        block_metadata: pl.DataFrame | None = None,
        backend: Backend = Backend.AUTO,
        dtype: Any = None,
        max_padding_ratio: float | None = 1.25,
    ) -> _PackedJaxLinearARG:
        """Construct the private candidate from an HDF5 LinearARG file."""
        return _packed_from_hdf5(
            path,
            None,
            mesh=mesh,
            block_metadata=block_metadata,
            backend=backend,
            dtype=dtype,
            max_padding_ratio=max_padding_ratio,
        ).operator

    def __check_init__(self) -> None:
        arrays = self.graph.components
        if len(self.capacities) != len(GRAPH_FIELD_NAMES):
            raise ValueError("capacities must contain one entry per packed graph field")
        if self.n_samples < 0 or self.n_variants < 0:
            raise ValueError("packed global shape must be nonnegative")
        expected_metadata = _PackedGraphLogicalMetadata(self.n_samples, self.n_variants, self.capacities)
        if self.graph.metadata != expected_metadata:
            raise ValueError("packed graph metadata must match the convenience carrier logical metadata")
        num_devices = arrays[0].shape[0]
        if num_devices < 1:
            raise ValueError("packed ingress requires at least one device")
        for name, array in zip(PACKED_COMPONENT_NAMES, arrays, strict=True):
            if not isinstance(array, jax.Array):
                raise ValueError(f"{name} must be a JAX array")
            if array.shape[0] != num_devices:
                raise ValueError(f"{name} leading dimension must equal the graph device count")
            if not isinstance(array.sharding, NamedSharding):
                raise ValueError(f"{name} must use NamedSharding")
            if array.sharding.mesh.axis_names != ("graph",) or array.sharding.spec[0] != "graph":
                raise ValueError(f"{name} must be sharded on the dedicated graph axis")

        from .packed_products import _validate_packed_carrier

        _validate_packed_carrier(self)

    @property
    def indptr(self) -> Array:
        return _packed_graph_component(self.graph, 0)

    @property
    def indices(self) -> Array:
        return _packed_graph_component(self.graph, 1)

    @property
    def data(self) -> Array:
        return _packed_graph_component(self.graph, 2)

    @property
    def variant_indices(self) -> Array:
        return _packed_graph_component(self.graph, 3)

    @property
    def flip(self) -> Array:
        return _packed_graph_component(self.graph, 4)

    @property
    def sample_indices(self) -> Array:
        return _packed_graph_component(self.graph, 5)

    @property
    def nonunique_indices(self) -> Array:
        return _packed_graph_component(self.graph, 6)

    @property
    def allele_counts(self) -> Array:
        return _packed_graph_component(self.graph, 7)

    @property
    def logical_variant_indices(self) -> Array:
        return _packed_graph_component(self.graph, 8)

    @property
    def block_descriptors(self) -> Array:
        return _packed_graph_component(self.graph, 9)

    @property
    def valid_lengths(self) -> Array:
        return _packed_graph_component(self.graph, 10)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the logical sample-by-variant shape."""
        return (self.n_samples, self.n_variants)

    @property
    def dtype(self) -> jnp.dtype:
        """Return the packed edge-data dtype."""
        return self.data.dtype

    def matmat(self, values: Any) -> Array:
        r"""Multiply by the packed LinearARG with graph state explicit.

        !!! info
            The memory guarantee excludes raw bound-method closure capture.
            Pass this carrier explicitly to `lineararg_matmat`, or
            use [`compile_matmat`][linear_dag.core.jaxlinarg.ingress._PackedJaxLinearARG.compile_matmat].

        **Arguments:**

        - `values`: Rank-1 or rank-2 logical variant values.

        **Returns:**

        - Logical sample-space product.
        """
        from .packed_products import lineararg_matmat

        return lineararg_matmat(self, values)

    def rmatmat(self, values: Any) -> Array:
        r"""Multiply by the transpose with graph state explicit.

        !!! info
            The memory guarantee excludes raw bound-method closure capture.
            Pass this carrier explicitly to `lineararg_rmatmat`, or
            use [`compile_rmatmat`][linear_dag.core.jaxlinarg.ingress._PackedJaxLinearARG.compile_rmatmat].

        **Arguments:**

        - `values`: Rank-1 or rank-2 logical sample values.

        **Returns:**

        - Logical variant-space product in exact variant order.
        """
        from .packed_products import lineararg_rmatmat

        return lineararg_rmatmat(self, values)

    def compile_matmat(self) -> Any:
        r"""Compile a forward product without raw bound-method closure capture.

        The returned Python wrapper retains this carrier for convenience but
        supplies it as an explicit argument to a module-level JIT on every
        invocation, preserving the graph-memory guarantee.

        **Returns:**

        - Callable compiled forward-product wrapper.
        """
        from .packed_products import compile_matmat

        return compile_matmat(self)

    def compile_rmatmat(self) -> Any:
        r"""Compile a transpose product without raw bound-method closure capture.

        The returned Python wrapper retains this carrier for convenience but
        supplies it as an explicit argument to a module-level JIT on every
        invocation, preserving the graph-memory guarantee.

        **Returns:**

        - Callable compiled transpose-product wrapper.
        """
        from .packed_products import compile_rmatmat

        return compile_rmatmat(self)


@dataclass(frozen=True)
class _PackedIngressResult:
    """Private packed operator plus host-only construction diagnostics."""

    operator: _PackedJaxLinearARG
    diagnostics: _PackedIngressDiagnostics


def _packed_from_block_arrays(
    blocks: Iterable[LinearARGBlockArrays],
    *,
    mesh: Mesh,
    backend: Backend = Backend.PURE_JAX,
    dtype: Any = None,
    max_padding_ratio: float | None = 1.25,
    allow_excess_padding: bool = False,
    iids: Any = None,
) -> _PackedIngressResult:
    """Construct the private packed carrier from canonical host blocks."""
    backend = _resolve_packed_backend(backend)
    loaded_blocks = list(blocks)
    normalized_dtype = _normalize_dtype(dtype)
    plan = _plan_packing_from_summaries(
        (_block_packing_summary_from_arrays(block, dtype=normalized_dtype) for block in loaded_blocks),
        num_devices=_graph_mesh_devices(mesh),
        dtype=normalized_dtype,
        max_padding_ratio=max_padding_ratio,
        allow_excess_padding=allow_excess_padding,
    )
    pending_blocks: list[LinearARGBlockArrays | None] = list(loaded_blocks)
    del loaded_blocks

    def load_block(logical_block_index: int) -> LinearARGBlockArrays:
        block = pending_blocks[logical_block_index]
        if block is None:
            raise RuntimeError("canonical block source was released before staging")
        pending_blocks[logical_block_index] = None
        return block

    return _packed_from_plan(load_block, plan=plan, mesh=mesh, backend=backend, iids=iids)


def _packed_from_hdf5(
    path: str | PathLike[str],
    block_names: Iterable[Any] | None = None,
    *,
    mesh: Mesh,
    block_metadata: pl.DataFrame | None = None,
    backend: Backend = Backend.PURE_JAX,
    dtype: Any = None,
    max_padding_ratio: float | None = 1.25,
    allow_excess_padding: bool = False,
    root_only: bool = False,
) -> _PackedIngressResult:
    """Construct the private packed carrier from a validated HDF5 layout."""
    backend = _resolve_packed_backend(backend)
    normalized_dtype = _normalize_dtype(dtype)
    _ensure_hdf5_plugins()

    with h5py.File(_hdf5_path(path), "r") as file:
        layout = _hdf5_layout(file)
        if root_only and layout != "root":
            raise ValueError("block=None is valid only for a root-level single-block HDF5 file")
        if layout == "root":
            if block_metadata is not None:
                raise ValueError("block_metadata is not valid for a root-level single-block HDF5 file")
            if block_names is not None and tuple(block_names):
                raise ValueError("block names are not valid for a root-level single-block HDF5 file")
            groups = (file,)
        else:
            names = _validated_hdf5_block_names(
                file,
                block_names=block_names,
                block_metadata=block_metadata,
            )
            groups = tuple(file[name] for name in names)

        summaries = tuple(_block_packing_summary_from_group(group, dtype=normalized_dtype) for group in groups)
        plan = _plan_packing_from_summaries(
            summaries,
            num_devices=_graph_mesh_devices(mesh),
            dtype=normalized_dtype,
            max_padding_ratio=max_padding_ratio,
            allow_excess_padding=allow_excess_padding,
        )

        def load_block(logical_block_index: int) -> LinearARGBlockArrays:
            return _read_block_arrays_from_group(groups[logical_block_index], dtype=normalized_dtype)

        return _packed_from_plan(
            load_block,
            plan=plan,
            mesh=mesh,
            backend=backend,
            iids=_read_hdf5_iids(file),
        )


def _packed_from_group_reader(
    reader: Any,
    block_names: Iterable[Any],
    *,
    mesh: Mesh,
    backend: Backend = Backend.PURE_JAX,
    dtype: Any = None,
    max_padding_ratio: float | None = 1.25,
    allow_excess_padding: bool = False,
) -> _PackedIngressResult:
    """Construct from the existing duck-typed group-reader test seam."""
    backend = _resolve_packed_backend(backend)
    names = tuple(block_names)
    normalized_dtype = _normalize_dtype(dtype)

    blocks_group = reader.root["blocks"]
    summaries = tuple(_block_packing_summary_from_group(blocks_group[name], dtype=normalized_dtype) for name in names)
    plan = _plan_packing_from_summaries(
        summaries,
        num_devices=_graph_mesh_devices(mesh),
        dtype=normalized_dtype,
        max_padding_ratio=max_padding_ratio,
        allow_excess_padding=allow_excess_padding,
    )

    def load_block(logical_block_index: int) -> LinearARGBlockArrays:
        return _read_block_arrays_from_group(blocks_group[names[logical_block_index]], dtype=normalized_dtype)

    return _packed_from_plan(load_block, plan=plan, mesh=mesh, backend=backend, iids=None)


def _packed_from_plan(
    load_block: Callable[[int], LinearARGBlockArrays],
    *,
    plan: PackingPlan,
    mesh: Mesh,
    backend: Backend,
    iids: Any,
) -> _PackedIngressResult:
    backend = _resolve_packed_backend(backend)
    packed, staging_bytes_by_block = _stage_blocks(load_block, plan=plan)
    arrays = {name: _assemble_host_shards(getattr(packed.buffers, name), mesh=mesh) for name in PACKED_COMPONENT_NAMES}
    for array in arrays.values():
        array.block_until_ready()

    first_array = next(iter(arrays.values()))
    first_sharding = first_array.sharding
    if not isinstance(first_sharding, NamedSharding):
        raise ValueError("packed arrays must use NamedSharding")
    device_order = _addressable_devices_in_shard_order(first_sharding, first_array.shape)
    final_graph_bytes = _resident_bytes_by_device(
        tuple(arrays[name] for name in GRAPH_FIELD_NAMES),
        device_order=device_order,
    )
    final_bytes = _resident_bytes_by_device(tuple(arrays.values()), device_order=device_order)
    staging_by_device = tuple(
        max(
            (size for size, owner in zip(staging_bytes_by_block, plan.assignment, strict=True) if owner == device),
            default=0,
        )
        for device in range(plan.config.num_devices)
    )
    diagnostics = _PackedIngressDiagnostics(
        canonical_graph_bytes=plan.diagnostics.canonical_graph_bytes,
        padded_graph_bytes=plan.diagnostics.padded_graph_bytes,
        descriptor_bytes=sum(final_bytes) - sum(final_graph_bytes),
        staging_bytes=max(staging_bytes_by_block, default=0),
        staging_bytes_by_device=staging_by_device,
        staging_block_owners=plan.assignment,
        final_graph_bytes_by_device=final_graph_bytes,
        final_bytes_by_device=final_bytes,
        padding_ratio=plan.diagnostics.padding_ratio,
        max_padding_ratio=plan.diagnostics.max_padding_ratio,
        component_count=len(PACKED_COMPONENT_NAMES),
        pytree_leaf_count=1,
    )
    return _PackedIngressResult(
        operator=_PackedJaxLinearARG(
            n_samples=plan.n_samples,
            n_variants=plan.n_variants,
            capacities=tuple(plan.capacities.values()),
            graph_mesh=mesh,
            backend=backend,
            iids=iids,
            graph=_make_packed_graph_value(
                tuple(arrays[name] for name in PACKED_COMPONENT_NAMES),
                metadata=_PackedGraphLogicalMetadata(
                    n_samples=plan.n_samples,
                    n_variants=plan.n_variants,
                    capacities=tuple(plan.capacities.values()),
                ),
            ),
        ),
        diagnostics=diagnostics,
    )


def _stage_blocks(
    load_block: Callable[[int], LinearARGBlockArrays],
    *,
    plan: PackingPlan,
) -> tuple[PackedGraph, tuple[int, ...]]:
    buffers = _allocate_host_buffers(plan)
    descriptor_row_by_block: dict[int, int] = {}
    rows_by_device = [0] * plan.config.num_devices
    for descriptor in plan.descriptors:
        descriptor_row_by_block[descriptor.logical_block_index] = rows_by_device[descriptor.device]
        rows_by_device[descriptor.device] += 1

    staging_bytes = [0] * len(plan.assignment)
    finalized_descriptors = []
    compressed_starts = [0] * plan.config.num_devices
    for planned_descriptor in plan.descriptors:
        logical_block_index = planned_descriptor.logical_block_index
        source_block = load_block(logical_block_index)
        canonical_block = canonicalize_block_arrays(source_block, dtype=plan.data_dtype)
        descriptor = _finalize_staged_descriptor(
            replace(planned_descriptor, compressed_start=compressed_starts[planned_descriptor.device]),
            canonical_block,
        )
        compressed_starts[descriptor.device] += descriptor.compressed_length
        _pack_block_into_buffers(buffers, canonical_block, descriptor)
        descriptor_row = descriptor_row_by_block[logical_block_index]
        buffers.block_descriptors[descriptor.device, descriptor_row] = descriptor.as_array_row()
        staging_bytes[logical_block_index] = _block_metrics(canonical_block).canonical_bytes
        finalized_descriptors.append(descriptor)
        del source_block, canonical_block

    for device in range(plan.config.num_devices):
        indptr_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index("indptr")])
        edge_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index("indices")])
        buffers.indptr[device, indptr_length:] = edge_length

    packed = PackedGraph(buffers=buffers, plan=replace(plan, descriptors=tuple(finalized_descriptors)))
    validate_packed_graph(packed)
    return packed, tuple(staging_bytes)


def _graph_mesh_devices(mesh: Mesh) -> int:
    if not isinstance(mesh, Mesh):
        raise ValueError("packed ingress requires a concrete single-host Mesh")
    if mesh.axis_names != ("graph",):
        raise ValueError('packed ingress mesh must use the dedicated axis name "graph"')
    sharding = NamedSharding(mesh, PartitionSpec("graph"))
    addressable_devices = tuple(sharding.addressable_devices)
    if len(addressable_devices) != mesh.size:
        raise ValueError("packed ingress requires every graph mesh device to be addressable on this host")
    return len(addressable_devices)


def _assemble_host_shards(host_values: np.ndarray, *, mesh: Mesh) -> Array:
    spec = PartitionSpec("graph", *([None] * (host_values.ndim - 1)))
    sharding = NamedSharding(mesh, spec)
    devices = _addressable_devices_in_shard_order(sharding, host_values.shape)
    local_arrays = tuple(
        jax.device_put(np.ascontiguousarray(host_values[index : index + 1]), device)
        for index, device in enumerate(devices)
    )
    return _assemble_single_device_arrays(host_values.shape, sharding, local_arrays)


def _assemble_single_device_arrays(
    global_shape: tuple[int, ...],
    sharding: NamedSharding,
    local_arrays: Iterable[Array],
) -> Array:
    """Validate committed local shards before global array assembly."""
    arrays = tuple(local_arrays)
    devices = _addressable_devices_in_shard_order(sharding, global_shape)
    if len(arrays) != len(devices):
        raise ValueError("one local array is required for each addressable sharding device")
    expected_shape = sharding.shard_shape(global_shape)
    for array, device in zip(arrays, devices, strict=True):
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError(f"local shard shape must be {expected_shape}; observed {array.shape}")
        if not array.committed or array.devices() != {device}:
            raise ValueError(f"local shard must be committed only to {device}")
    return jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)


def _addressable_devices_in_shard_order(
    sharding: NamedSharding,
    global_shape: tuple[int, ...],
) -> tuple[jax.Device, ...]:
    """Return addressable devices in the local-array order required by JAX."""
    return tuple(sharding.addressable_devices_indices_map(global_shape))


def _resident_bytes_by_device(
    arrays: tuple[Array, ...],
    *,
    device_order: tuple[jax.Device, ...],
) -> tuple[int, ...]:
    byte_counts = {device: 0 for device in device_order}
    for array in arrays:
        for shard in array.addressable_shards:
            shard.data.block_until_ready()
            if shard.device not in byte_counts:
                raise ValueError(f"packed array has an unexpected resident device {shard.device}")
            byte_counts[shard.device] += int(shard.data.on_device_size_in_bytes())
    return tuple(byte_counts[device] for device in device_order)


def _single_block_mesh(mesh: Mesh | None) -> Mesh:
    if mesh is None:
        mesh = Mesh(np.asarray((jax.devices()[0],)), ("graph",))
    if _graph_mesh_devices(mesh) != 1:
        raise ValueError("single-block packed construction requires a one-device graph mesh")
    return mesh


def _iid_tuple(values: Any) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, pl.Series):
        values = values.to_list()
    return tuple(str(value) for value in values)


def _shared_lineararg_iids(lineargs: tuple[LinearARG, ...]) -> tuple[str, ...] | None:
    if not lineargs:
        raise ValueError("at least one LinearARG block is required")
    iids = tuple(_iid_tuple(getattr(linarg, "iids", None)) for linarg in lineargs)
    first = iids[0]
    if any(current != first for current in iids[1:]):
        raise ValueError("all LinearARG blocks must have identical IID metadata")
    return first


def _lineararg_block_arrays(linarg: LinearARG, *, dtype: Any = None) -> LinearARGBlockArrays:
    """Convert one in-memory LinearARG into canonical host transfer arrays."""
    normalized_dtype = _normalize_dtype(dtype)
    graph = _as_csc(linarg.A)
    n_nodes = graph.shape[0]
    return LinearARGBlockArrays(
        indptr=np.asarray(graph.indptr, dtype=np.int32),
        indices=np.asarray(graph.indices, dtype=np.int32),
        data=np.asarray(graph.data, dtype=np.dtype(normalized_dtype)),
        variant_indices=np.asarray(linarg.variant_indices, dtype=np.int32),
        flip=np.asarray(linarg.flip, dtype=np.bool_),
        sample_indices=np.asarray(linarg.sample_indices, dtype=np.int32),
        nonunique_indices=_canonical_nonunique_indices(getattr(linarg, "nonunique_indices", None), n_nodes),
        allele_counts=_cached_allele_counts(linarg),
        n_variants=int(linarg.shape[1]),
        n_samples=int(linarg.shape[0]),
    )


def _hdf5_layout(file: h5py.File) -> str:
    groups = tuple(name for name in file if isinstance(file[name], h5py.Group))
    root_datasets = {name for name in _ROOT_GRAPH_DATASETS if name in file and isinstance(file[name], h5py.Dataset)}
    root_attributes = {name for name in _ROOT_GRAPH_ATTRIBUTES if name in file.attrs}
    if groups:
        if root_datasets or root_attributes:
            raise ValueError("mixed root-level graph and block-group HDF5 layouts are ambiguous")
        return "blocks"
    missing_datasets = _ROOT_GRAPH_DATASETS - root_datasets
    missing_attributes = _ROOT_GRAPH_ATTRIBUTES - root_attributes
    if missing_datasets or missing_attributes:
        missing = sorted((*missing_datasets, *missing_attributes))
        raise ValueError(f"empty or corrupt HDF5 source is missing root graph fields: {missing}")
    return "root"


def _hdf5_block_sort_key(block_name: str) -> tuple[int, int | str, float]:
    parts = block_name.split("_")
    if len(parts) == 3:
        chrom, start, _ = parts
    else:
        try:
            chrom, interval = block_name.split(":", maxsplit=1)
            start = interval.split("-", maxsplit=1)[0]
        except ValueError as error:
            raise ValueError(f"HDF5 block name {block_name!r} does not encode chromosome/start order") from error
    normalized_chrom = chrom[3:] if chrom.startswith("chr") else chrom
    try:
        chrom_key: tuple[int, int | str] = (0, int(normalized_chrom))
    except ValueError:
        chrom_key = (1, normalized_chrom)
    try:
        start_key = float(start)
    except ValueError:
        start_key = float("inf")
    return (*chrom_key, start_key)


def _ordered_hdf5_block_names(file: h5py.File) -> tuple[str, ...]:
    names = tuple(name for name in file if isinstance(file[name], h5py.Group))
    return tuple(sorted(names, key=_hdf5_block_sort_key))


def _validated_hdf5_block_names(
    file: h5py.File,
    *,
    block_names: Iterable[Any] | None,
    block_metadata: pl.DataFrame | None,
) -> tuple[str, ...]:
    canonical_names = _ordered_hdf5_block_names(file)
    if not canonical_names:
        raise ValueError("HDF5 source contains no LinearARG blocks")

    requested = None if block_names is None else tuple(str(name) for name in block_names)
    if requested is not None and len(set(requested)) != len(requested):
        raise ValueError("block names must be unique")

    if block_metadata is None:
        selected = canonical_names if requested is None else requested
    else:
        if not isinstance(block_metadata, pl.DataFrame):
            raise TypeError("block_metadata must be a Polars DataFrame")
        required_columns = {"block_name", "n_entries", "n_variants", "n_samples"}
        missing_columns = required_columns - set(block_metadata.columns)
        if missing_columns:
            raise ValueError(f"block_metadata is missing required columns: {sorted(missing_columns)}")
        metadata_names = tuple(str(name) for name in block_metadata.get_column("block_name").to_list())
        if len(set(metadata_names)) != len(metadata_names):
            raise ValueError("block_metadata block_name values must be unique")
        if requested is None:
            selected = metadata_names
        else:
            requested_set = set(requested)
            selected = tuple(name for name in metadata_names if name in requested_set)
            if set(selected) != requested_set:
                missing = sorted(requested_set - set(selected))
                raise ValueError(f"block_metadata does not contain requested blocks: {missing}")

        rows = {str(row["block_name"]): row for row in block_metadata.iter_rows(named=True)}
        for name in selected:
            if name not in file or not isinstance(file[name], h5py.Group):
                raise ValueError(f"block_metadata names an HDF5 block that does not exist: {name!r}")
            _validate_hdf5_metadata_row(rows[name], file[name])

    unknown = sorted(set(selected) - set(canonical_names))
    if unknown:
        raise ValueError(f"requested HDF5 blocks do not exist: {unknown}")
    expected_order = tuple(name for name in canonical_names if name in set(selected))
    if tuple(selected) != expected_order:
        raise ValueError("HDF5 block order must match canonical chromosome/start order")
    if not selected:
        raise ValueError("HDF5 block selection is empty")
    return tuple(selected)


def _validate_hdf5_metadata_row(row: dict[str, Any], group: h5py.Group) -> None:
    expected = {
        "n_entries": _group_array_length(group, "indices"),
        "n_variants": int(group.attrs["n_variants"]),
        "n_samples": int(group.attrs["n_samples"]),
    }
    if "n" in row:
        expected["n"] = int(group.attrs["n"])
    for name, expected_value in expected.items():
        try:
            observed = int(row[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"block_metadata {name} must be an integer") from error
        if observed != expected_value:
            raise ValueError(
                f"block_metadata {name} mismatch for {group.name!r}: expected {expected_value}, observed {observed}"
            )


def _read_hdf5_iids(file: h5py.File) -> tuple[str, ...]:
    if "iids" not in file or not isinstance(file["iids"], h5py.Dataset):
        raise ValueError("iids not found in HDF5 file")
    return tuple(str(value) for value in np.asarray(file["iids"][:]).astype(str).tolist())


def from_lineararg(
    linarg: LinearARG,
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> JaxLinearARG:
    """Convert a [`linear_dag.core.lineararg.LinearARG`][] to a JAX operator.

    **Arguments:**

    - `linarg`: Source LinearARG object.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    backend = resolve_backend(backend)
    dtype = _normalize_dtype(dtype)
    graph = _as_csc(linarg.A)
    n_nodes = graph.shape[0]
    nonunique_indices = _canonical_nonunique_indices(getattr(linarg, "nonunique_indices", None), n_nodes)

    indptr = np.asarray(graph.indptr, dtype=np.int32)
    indices = np.asarray(graph.indices, dtype=np.int32)
    data = np.asarray(graph.data, dtype=np.dtype(dtype))
    return JaxLinearARG.from_lineararg_arrays(
        **_block_arrays_kwargs(
            LinearARGBlockArrays(
                indptr=indptr,
                indices=indices,
                data=data,
                variant_indices=np.asarray(linarg.variant_indices, dtype=np.int32),
                flip=np.asarray(linarg.flip, dtype=np.bool_),
                sample_indices=np.asarray(linarg.sample_indices, dtype=np.int32),
                nonunique_indices=nonunique_indices,
                allele_counts=_cached_allele_counts(linarg),
                n_variants=int(linarg.shape[1]),
                n_samples=int(linarg.shape[0]),
            )
        ),
        backend=backend,
        dtype=dtype,
    )


def from_hdf5_block(
    path: Any,
    block: Any,
    *,
    backend: Backend = Backend.AUTO,
    load_metadata: bool = False,
    dtype: Any = None,
) -> JaxLinearARG:
    """Read one HDF5 LinearARG block as a JAX operator.

    **Arguments:**

    - `path`: HDF5 file path.
    - `block`: Block name inside the HDF5 file.
    - `backend`: Requested numerical backend.
    - `load_metadata`: Accepted for compatibility. JAX operators do not retain
      variant metadata, so this does not materialize metadata tables.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    del load_metadata
    backend = resolve_backend(backend)
    return from_block_arrays(read_hdf5_block_arrays(path, block, dtype=dtype), backend=backend, dtype=dtype)


def from_zarr_group(
    group: Any,
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> JaxLinearARG:
    """Read one Zarr LinearARG block group as a JAX operator.

    **Arguments:**

    - `group`: Zarr group containing one LinearARG block.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    backend = resolve_backend(backend)
    return from_block_arrays(read_zarr_block_arrays(group, dtype=dtype), backend=backend, dtype=dtype)


def from_block_arrays(
    arrays: LinearARGBlockArrays,
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> JaxLinearARG:
    """Construct a JAX LinearARG operator from canonical host block arrays.

    **Arguments:**

    - `arrays`: Canonical LinearARG block arrays.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    backend = resolve_backend(backend)
    dtype = _normalize_dtype(dtype)
    return JaxLinearARG.from_lineararg_arrays(
        **_block_arrays_kwargs(arrays),
        backend=backend,
        dtype=dtype,
    )


def read_hdf5_block_arrays(path: str | PathLike[str], block: Any, *, dtype: Any = None) -> LinearARGBlockArrays:
    """Read one HDF5 LinearARG block into canonical JAX-ingress host arrays.

    **Arguments:**

    - `path`: HDF5 file path.
    - `block`: Block name inside the HDF5 file.
    - `dtype`: Optional data dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Canonical host arrays for [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    _ensure_hdf5_plugins()
    with h5py.File(_hdf5_path(path), "r") as file:
        return _read_block_arrays_from_group(file[block], dtype=dtype)


def read_hdf5_blocks(
    path: str | PathLike[str],
    block_names: Iterable[Any],
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> tuple[JaxLinearARG, ...]:
    """Read HDF5 LinearARG blocks directly as eager JAX operators.

    !!! info

        This convenience function materializes every requested block eagerly on
        the default device. Use
        [`linear_dag.core.jaxlinarg.JaxParallelOperator.from_hdf5`][] for
        device-aware multi-block ingress without transient graph duplication.

    **Arguments:**

    - `path`: HDF5 file path.
    - `block_names`: Block names inside the HDF5 file.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Tuple of [`linear_dag.core.jaxlinarg.JaxLinearARG`][] blocks.
    """
    backend = resolve_backend(backend)
    _ensure_hdf5_plugins()
    with h5py.File(_hdf5_path(path), "r") as file:
        return tuple(
            from_block_arrays(
                _read_block_arrays_from_group(file[block_name], dtype=dtype),
                backend=backend,
                dtype=dtype,
            )
            for block_name in block_names
        )


def read_zarr_block_arrays(group: Any, *, dtype: Any = None) -> LinearARGBlockArrays:
    """Read one Zarr LinearARG block group into canonical JAX-ingress host arrays.

    **Arguments:**

    - `group`: Zarr group containing one LinearARG block.
    - `dtype`: Optional data dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Canonical host arrays for [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    return _read_block_arrays_from_group(group, dtype=dtype)


def read_zarr_blocks(
    reader: Any,
    block_names: Iterable[str],
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> tuple[JaxLinearARG, ...]:
    """Read Zarr LinearARG blocks directly as eager JAX operators.

    !!! info

        This convenience function materializes every requested block eagerly on
        the default device. Use
        [`linear_dag.core.jaxlinarg.JaxParallelOperator.from_zarr`][] for
        device-aware multi-block ingress without transient graph duplication.

    **Arguments:**

    - `reader`: Open Zarr reader with a `root["blocks"]` group.
    - `block_names`: Block names inside `blocks/`.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Tuple of [`linear_dag.core.jaxlinarg.JaxLinearARG`][] blocks.
    """
    backend = resolve_backend(backend)
    blocks_group = reader.root["blocks"]
    return tuple(from_zarr_group(blocks_group[block_name], backend=backend, dtype=dtype) for block_name in block_names)


def _as_csc(matrix: Any) -> sparse.csc_matrix:
    if sparse.isspmatrix_csc(matrix):
        graph = matrix
    elif sparse.issparse(matrix) and hasattr(matrix, "tocsc"):
        graph = matrix.tocsc(copy=False)
    else:
        raise ValueError("linarg.A must be a scipy sparse matrix with CSC-compatible semantics")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("linarg.A must be square")
    return graph


def _canonical_nonunique_indices(nonunique_indices: Any, n_nodes: int) -> np.ndarray:
    if nonunique_indices is None:
        return np.arange(n_nodes, dtype=np.int32)
    return np.asarray(nonunique_indices, dtype=np.int32)


def _cached_allele_counts(linarg: LinearARG) -> np.ndarray | None:
    allele_counts = linarg.__dict__.get("allele_counts")
    if allele_counts is None:
        return None
    return np.asarray(allele_counts, dtype=np.int32)


def _read_block_arrays_from_group(group: Any, *, dtype: Any = None) -> LinearARGBlockArrays:
    dtype = _normalize_dtype(dtype)
    n_nodes = int(group.attrs["n"])
    n_samples = int(group.attrs["n_samples"])
    n_individuals = group.attrs.get("n_individuals", None)
    return LinearARGBlockArrays(
        indptr=np.asarray(group["indptr"][:], dtype=np.int32),
        indices=np.asarray(group["indices"][:], dtype=np.int32),
        data=np.asarray(group["data"][:], dtype=np.dtype(dtype)),
        variant_indices=np.asarray(group["variant_indices"][:], dtype=np.int32),
        flip=np.asarray(group["flip"][:], dtype=np.bool_),
        sample_indices=_sample_indices(n_nodes, n_samples, n_individuals),
        nonunique_indices=_optional_array(group, "nonunique_indices", dtype=np.int32),
        allele_counts=_optional_array(group, "allele_counts", dtype=np.int32),
        n_variants=int(group.attrs["n_variants"]),
        n_samples=n_samples,
    )


def _block_packing_summary_from_group(group: Any, *, dtype: Any) -> _BlockPackingSummary:
    """Read only group attributes and dataset shapes needed for packing."""
    data_dtype = np.dtype(dtype)
    if not np.issubdtype(data_dtype, np.floating):
        raise ValueError("data dtype must be a floating dtype")
    n_nodes = int(group.attrs["n"])
    n_samples = int(group.attrs["n_samples"])
    n_variants = int(group.attrs["n_variants"])
    if n_nodes < 1 or n_samples < 0 or n_variants < 0:
        raise ValueError("block shape metadata must describe nonnegative dimensions and at least one node")

    indptr_length = _group_array_length(group, "indptr")
    edge_length = _group_array_length(group, "indices")
    if "n_entries" not in group.attrs or int(group.attrs["n_entries"]) != edge_length:
        raise ValueError("n_entries metadata must match indices")
    if indptr_length != n_nodes + 1:
        raise ValueError("indptr metadata length must equal n + 1")
    if _group_array_length(group, "data") != edge_length:
        raise ValueError("data metadata length must match indices")
    for name in ("variant_indices", "flip"):
        if _group_array_length(group, name) != n_variants:
            raise ValueError(f"{name} metadata length must match n_variants")
    nonunique_length = _group_array_length(group, "nonunique_indices") if "nonunique_indices" in group else n_nodes
    allele_count_length = _group_array_length(group, "allele_counts") if "allele_counts" in group else n_variants
    if nonunique_length != n_nodes:
        raise ValueError("nonunique_indices metadata length must match n")
    if allele_count_length != n_variants:
        raise ValueError("allele_counts metadata length must match n_variants")

    sample_indices = _sample_indices(n_nodes, n_samples, group.attrs.get("n_individuals", None))
    if sample_indices.size != n_samples:
        raise ValueError("sample metadata cannot produce the declared n_samples")
    return _BlockPackingSummary(
        field_lengths=(
            indptr_length,
            edge_length,
            edge_length,
            n_variants,
            n_variants,
            n_samples,
            nonunique_length,
            allele_count_length,
            n_variants,
        ),
        data_dtype=data_dtype,
        n_samples=n_samples,
        n_variants=n_variants,
        min_index_to_keep=int(sample_indices[-1]) if sample_indices.size else 0,
        compressed_length=None,
    )


def _group_array_length(group: Any, name: str) -> int:
    array = group[name]
    shape = tuple(array.shape)
    if len(shape) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return int(shape[0])


def _block_arrays_kwargs(arrays: LinearARGBlockArrays) -> dict[str, Any]:
    return {
        "indptr": arrays.indptr,
        "indices": arrays.indices,
        "data": arrays.data,
        "variant_indices": arrays.variant_indices,
        "flip": arrays.flip,
        "sample_indices": arrays.sample_indices,
        "nonunique_indices": arrays.nonunique_indices,
        "allele_counts": arrays.allele_counts,
        "n_variants": arrays.n_variants,
        "n_samples": arrays.n_samples,
    }


def _optional_array(group: Any, name: str, *, dtype: Any) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][:], dtype=dtype)


def _normalize_dtype(dtype: Any) -> jnp.dtype:
    requested_dtype = jnp.float32 if dtype is None else jnp.dtype(dtype)
    return jax.dtypes.canonicalize_dtype(requested_dtype)


def _hdf5_path(path: str | PathLike[str]) -> str | PathLike[str]:
    if str(path).endswith(".h5"):
        return path
    return f"{path}.h5"


def _sample_indices(n_nodes: int, n_samples: int, n_individuals: Any) -> np.ndarray:
    if n_individuals is None:
        stop = n_nodes - n_samples - 1
        return np.arange(n_nodes - 1, stop, -1, dtype=np.int32)
    start = n_nodes - int(n_individuals) - 1
    stop = n_nodes - int(n_individuals) - n_samples - 1
    return np.arange(start, stop, -1, dtype=np.int32)


def _ensure_hdf5_plugins() -> None:
    if find_spec("hdf5plugin") is None:
        warnings.warn("hdf5plugin is required for blosc compression; this may impact reading", stacklevel=2)
        return
    # Importing the optional package registers its HDF5 filters process-wide.
    import_module("hdf5plugin")
