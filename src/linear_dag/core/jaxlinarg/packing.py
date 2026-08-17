# pattern: Functional Core

"""Pure planning and host packing for fixed-shape LinearARG graph shards."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, NamedTuple

import numpy as np

DEVICE_METADATA_DTYPE = np.dtype(np.int32)
GRAPH_FIELD_NAMES = (
    "indptr",
    "indices",
    "data",
    "variant_indices",
    "flip",
    "sample_indices",
    "nonunique_indices",
    "allele_counts",
    "logical_variant_indices",
)
PACKED_COMPONENT_NAMES = (*GRAPH_FIELD_NAMES, "block_descriptors", "valid_lengths")
VALID_LENGTH_FIELDS = (*GRAPH_FIELD_NAMES, "block_descriptors")
# Native packed solves consume a narrower, versioned view of the host packing
# descriptor. Every offset below is rebased into its device-local flattened
# graph buffer; lengths are counts in that buffer's element units. Padded rows
# retain the version, set ``valid`` to zero, and zero every remaining column.
PACKED_FFI_DESCRIPTOR_VERSION = 1
PACKED_FFI_DESCRIPTOR_FIELDS = (
    "version",
    "valid",
    "node_start",
    "node_length",
    "indptr_start",
    "indptr_length",
    "edge_start",
    "edge_length",
    "compressed_start",
    "compressed_length",
    "min_index_to_keep",
)
BLOCK_DESCRIPTOR_FIELDS = (
    "logical_block_index",
    "indptr_start",
    "indptr_length",
    "edge_start",
    "edge_length",
    "node_start",
    "node_length",
    "variant_start",
    "variant_length",
    "sample_start",
    "sample_length",
    "compressed_start",
    "compressed_length",
    "min_index_to_keep",
    "logical_variant_start",
    "logical_variant_stop",
)


@dataclass(frozen=True)
class LinearARGBlockArrays:
    """Canonical host arrays for one LinearARG block.

    This immutable transfer object separates storage I/O from JAX device
    construction. Array dtypes are normalized before packing or device ingress.

    !!! Example

        ```python
        from linear_dag.core.jaxlinarg.ingress import read_hdf5_block_arrays

        arrays = read_hdf5_block_arrays("lineararg.h5", "block_0")
        ```
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    variant_indices: np.ndarray
    flip: np.ndarray
    sample_indices: np.ndarray
    nonunique_indices: np.ndarray | None
    allele_counts: np.ndarray | None
    n_variants: int
    n_samples: int


@dataclass(frozen=True)
class PackingConfig:
    """Host-side configuration for whole-block packing."""

    num_devices: int
    data_dtype: np.dtype[Any]
    max_padding_ratio: float | None = 1.25
    allow_excess_padding: bool = False


@dataclass(frozen=True)
class BlockDescriptor:
    """One source block's spans in its assigned physical device shard."""

    device: int
    logical_block_index: int
    indptr_start: int
    indptr_length: int
    edge_start: int
    edge_length: int
    node_start: int
    node_length: int
    variant_start: int
    variant_length: int
    sample_start: int
    sample_length: int
    compressed_start: int
    compressed_length: int
    min_index_to_keep: int
    logical_variant_start: int
    logical_variant_stop: int

    def as_array_row(self) -> tuple[int, ...]:
        """Return descriptor values in [`BLOCK_DESCRIPTOR_FIELDS`][] order."""
        return tuple(getattr(self, name) for name in BLOCK_DESCRIPTOR_FIELDS)


@dataclass(frozen=True)
class PackingDiagnostics:
    """Deterministic byte and work accounting for a packing plan."""

    canonical_graph_bytes: int
    padded_graph_bytes: int
    descriptor_bytes: int
    padding_ratio: float
    max_padding_ratio: float | None
    device_graph_bytes: tuple[int, ...]
    device_solve_work: tuple[int, ...]
    device_score_loads: tuple[int, ...]
    padding_override: bool
    rebalance_steps: int


@dataclass(frozen=True)
class PackingPlan:
    """Validated whole-block assignment and fixed packed capacities."""

    config: PackingConfig
    assignment: tuple[int, ...]
    blocks_by_device: tuple[tuple[int, ...], ...]
    descriptors: tuple[BlockDescriptor, ...]
    capacities: Mapping[str, int]
    valid_lengths_by_device: tuple[Mapping[str, int], ...]
    descriptor_capacity: int
    n_samples: int
    n_variants: int
    diagnostics: PackingDiagnostics
    component_names: tuple[str, ...] = PACKED_COMPONENT_NAMES
    descriptor_schema: tuple[str, ...] = BLOCK_DESCRIPTOR_FIELDS

    @property
    def data_dtype(self) -> np.dtype[Any]:
        """Return the canonical floating dtype for packed graph edge data."""
        return self.config.data_dtype


class PackedHostBuffers(NamedTuple):
    """Fixed host array components ready for sharded device assembly."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    variant_indices: np.ndarray
    flip: np.ndarray
    sample_indices: np.ndarray
    nonunique_indices: np.ndarray
    allele_counts: np.ndarray
    logical_variant_indices: np.ndarray
    block_descriptors: np.ndarray
    valid_lengths: np.ndarray


@dataclass(frozen=True)
class PackedGraph:
    """Validated host buffers together with their immutable packing plan."""

    buffers: PackedHostBuffers
    plan: PackingPlan


@dataclass(frozen=True)
class _PackedGraphLogicalMetadata:
    """Compact logical facts carried by the private packed JAX graph value."""

    n_samples: int
    n_variants: int
    capacities: tuple[int, ...]


def _make_packed_graph_value(
    components: tuple[Any, ...],
    *,
    metadata: _PackedGraphLogicalMetadata,
) -> Any:
    """Construct the private high-level graph value without importing HiJAX here."""
    from ._hijax import _PackedGraphValue

    return _PackedGraphValue(components=components, metadata=metadata)


def _packed_graph_component(graph: Any, index: int) -> Any:
    """Read one lowered component through the private HiJAX adapter."""
    from ._hijax import _packed_graph_component as read_component

    return read_component(graph, index)


def _packed_graph_sharding_spec(graph: Any) -> Any:
    """Return the private high-level sharding spec for a packed graph value."""
    import jax

    from ._hijax import _graph_pspec_for_type

    return _graph_pspec_for_type(jax.typeof(graph))


@dataclass(frozen=True)
class _BlockPackingSummary:
    """Lightweight block facts sufficient for deterministic packing."""

    field_lengths: tuple[int, ...]
    data_dtype: np.dtype[Any]
    n_samples: int
    n_variants: int
    min_index_to_keep: int
    compressed_length: int | None


def _block_packing_summary_from_arrays(
    arrays: LinearARGBlockArrays,
    *,
    dtype: Any,
) -> _BlockPackingSummary:
    """Derive planning facts without materializing canonical array copies."""
    data_dtype = np.dtype(dtype)
    if not np.issubdtype(data_dtype, np.floating):
        raise ValueError("data dtype must be a floating dtype")
    n_samples = _nonnegative_int(arrays.n_samples, name="n_samples")
    n_variants = _nonnegative_int(arrays.n_variants, name="n_variants")

    def length(values: Any, *, name: str) -> int:
        shape = np.shape(values)
        if len(shape) != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return int(shape[0])

    indptr_length = length(arrays.indptr, name="indptr")
    node_count = indptr_length - 1
    if node_count < 1:
        raise ValueError("indptr must describe at least one node")
    edge_length = length(arrays.indices, name="indices")
    if length(arrays.data, name="data") != edge_length:
        raise ValueError("data length must match indices")
    if length(arrays.variant_indices, name="variant_indices") != n_variants:
        raise ValueError("variant_indices length must match n_variants")
    if length(arrays.flip, name="flip") != n_variants:
        raise ValueError("flip length must match n_variants")
    if length(arrays.sample_indices, name="sample_indices") != n_samples:
        raise ValueError("sample_indices length must match n_samples")
    nonunique_length = (
        node_count if arrays.nonunique_indices is None else length(arrays.nonunique_indices, name="nonunique_indices")
    )
    allele_count_length = (
        n_variants if arrays.allele_counts is None else length(arrays.allele_counts, name="allele_counts")
    )
    if nonunique_length != node_count:
        raise ValueError("nonunique_indices length must match node count")
    if allele_count_length != n_variants:
        raise ValueError("allele_counts length must match n_variants")

    sample_indices = np.asarray(arrays.sample_indices)
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


@dataclass(frozen=True)
class _BlockMetrics:
    field_lengths: tuple[int, ...]
    canonical_bytes: int
    solve_work: int
    score_load: int


def canonicalize_block_arrays(arrays: LinearARGBlockArrays, *, dtype: Any = None) -> LinearARGBlockArrays:
    """Normalize and validate one source block before packing."""
    data_dtype = np.asarray(arrays.data).dtype if dtype is None else np.dtype(dtype)
    if not np.issubdtype(data_dtype, np.floating):
        raise ValueError("data dtype must be a floating dtype")

    n_variants = _nonnegative_int(arrays.n_variants, name="n_variants")
    n_samples = _nonnegative_int(arrays.n_samples, name="n_samples")
    indptr = _int32_array(arrays.indptr, name="indptr")
    indices = _int32_array(arrays.indices, name="indices")
    data = _data_array(arrays.data, dtype=data_dtype)
    variant_indices = _int32_array(arrays.variant_indices, name="variant_indices")
    flip = _bool_array(arrays.flip, name="flip")
    sample_indices = _int32_array(arrays.sample_indices, name="sample_indices")
    node_count = indptr.size - 1

    if arrays.nonunique_indices is None:
        nonunique_indices = np.arange(node_count, dtype=np.int32)
    else:
        nonunique_indices = _int32_array(arrays.nonunique_indices, name="nonunique_indices")
    if arrays.allele_counts is None:
        allele_counts = np.full(n_variants, -1, dtype=np.int32)
    else:
        allele_counts = _int32_array(arrays.allele_counts, name="allele_counts")

    _validate_canonical_block(
        indptr=indptr,
        indices=indices,
        data=data,
        variant_indices=variant_indices,
        flip=flip,
        sample_indices=sample_indices,
        nonunique_indices=nonunique_indices,
        allele_counts=allele_counts,
        n_variants=n_variants,
        n_samples=n_samples,
    )
    return LinearARGBlockArrays(
        indptr=indptr,
        indices=indices,
        data=data,
        variant_indices=variant_indices,
        flip=flip,
        sample_indices=sample_indices,
        nonunique_indices=nonunique_indices,
        allele_counts=allele_counts,
        n_variants=n_variants,
        n_samples=n_samples,
    )


def plan_packing(
    blocks: Iterable[LinearARGBlockArrays],
    *,
    num_devices: int,
    dtype: Any = None,
    max_padding_ratio: float | None = 1.25,
    allow_excess_padding: bool = False,
) -> PackingPlan:
    """Build a deterministic whole-block packing plan.

    Candidate placement uses canonical graph bytes plus node/edge solve work.
    A deterministic local move/swap pass then minimizes aggregate per-field
    padding before the configured bound is enforced.
    """
    source_blocks = tuple(blocks)
    if not source_blocks:
        raise ValueError("at least one LinearARG block is required")
    data_dtype = _resolve_data_dtype(source_blocks, dtype=dtype)
    canonical_blocks = tuple(canonicalize_block_arrays(block, dtype=data_dtype) for block in source_blocks)
    summaries = tuple(_summarize_canonical_block(block) for block in canonical_blocks)
    return _plan_packing_from_summaries(
        summaries,
        num_devices=num_devices,
        dtype=data_dtype,
        max_padding_ratio=max_padding_ratio,
        allow_excess_padding=allow_excess_padding,
    )


def _plan_packing_from_summaries(
    summaries: Iterable[_BlockPackingSummary],
    *,
    num_devices: int,
    dtype: Any,
    max_padding_ratio: float | None = 1.25,
    allow_excess_padding: bool = False,
) -> PackingPlan:
    """Plan from metadata summaries without owning source graph arrays."""
    if isinstance(num_devices, bool) or not isinstance(num_devices, (int, np.integer)) or num_devices < 1:
        raise ValueError("num_devices must be at least 1")
    if max_padding_ratio is not None and (
        isinstance(max_padding_ratio, bool) or not np.isfinite(max_padding_ratio) or max_padding_ratio < 1.0
    ):
        raise ValueError("max_padding_ratio must be None or a finite value of at least 1.0")
    if not isinstance(allow_excess_padding, (bool, np.bool_)):
        raise ValueError("allow_excess_padding must be Boolean")

    block_summaries = tuple(summaries)
    if not block_summaries:
        raise ValueError("at least one LinearARG block is required")
    data_dtype = np.dtype(dtype)
    if not np.issubdtype(data_dtype, np.floating):
        raise ValueError("data dtype must be a floating dtype")
    if any(summary.data_dtype != data_dtype for summary in block_summaries):
        raise ValueError("all packing summaries must use the requested data dtype")
    sample_counts = {summary.n_samples for summary in block_summaries}
    if len(sample_counts) != 1:
        raise ValueError("all blocks must have the same n_samples")

    config = PackingConfig(
        num_devices=int(num_devices),
        data_dtype=data_dtype,
        max_padding_ratio=None if max_padding_ratio is None else float(max_padding_ratio),
        allow_excess_padding=bool(allow_excess_padding),
    )
    metrics = tuple(_summary_metrics(summary) for summary in block_summaries)
    assignment = _initial_assignment(metrics, num_devices=config.num_devices)
    assignment, rebalance_steps = _rebalance_assignment(
        assignment,
        metrics,
        num_devices=config.num_devices,
        data_dtype=data_dtype,
    )
    blocks_by_device = _ordered_blocks_by_device(assignment, metrics, num_devices=config.num_devices)
    valid_lengths = _field_lengths_by_device(blocks_by_device, metrics)
    capacities = tuple(
        max(device_lengths[field] for device_lengths in valid_lengths) for field in range(len(GRAPH_FIELD_NAMES))
    )
    descriptors = _build_descriptors(block_summaries, blocks_by_device)
    descriptor_capacity = max(len(device_blocks) for device_blocks in blocks_by_device)
    _device_metadata_array(
        [
            (*device_lengths, len(device_blocks))
            for device_lengths, device_blocks in zip(valid_lengths, blocks_by_device)
        ],
        name="valid_lengths",
    )

    canonical_bytes = sum(metric.canonical_bytes for metric in metrics)
    padded_bytes = config.num_devices * sum(
        capacity * _field_dtype(name, data_dtype).itemsize
        for name, capacity in zip(GRAPH_FIELD_NAMES, capacities, strict=True)
    )
    descriptor_bytes = (
        config.num_devices
        * (descriptor_capacity * len(BLOCK_DESCRIPTOR_FIELDS) + len(VALID_LENGTH_FIELDS))
        * DEVICE_METADATA_DTYPE.itemsize
    )
    device_graph_bytes = tuple(
        sum(metrics[index].canonical_bytes for index in device_blocks) for device_blocks in blocks_by_device
    )
    device_solve_work = tuple(
        sum(metrics[index].solve_work for index in device_blocks) for device_blocks in blocks_by_device
    )
    device_score_loads = tuple(
        sum(metrics[index].score_load for index in device_blocks) for device_blocks in blocks_by_device
    )
    padding_ratio = padded_bytes / canonical_bytes
    diagnostics = PackingDiagnostics(
        canonical_graph_bytes=canonical_bytes,
        padded_graph_bytes=padded_bytes,
        descriptor_bytes=descriptor_bytes,
        padding_ratio=padding_ratio,
        max_padding_ratio=config.max_padding_ratio,
        device_graph_bytes=device_graph_bytes,
        device_solve_work=device_solve_work,
        device_score_loads=device_score_loads,
        padding_override=config.allow_excess_padding or config.max_padding_ratio is None,
        rebalance_steps=rebalance_steps,
    )

    if (
        config.max_padding_ratio is not None
        and padding_ratio > config.max_padding_ratio
        and not config.allow_excess_padding
    ):
        raise ValueError(
            "whole-block packing exceeds max_padding_ratio after rebalancing: "
            f"canonical bytes={canonical_bytes}, padded bytes={padded_bytes}, "
            f"padding ratio={padding_ratio:.6f}, per-device loads={device_graph_bytes}. "
            "Pass a larger max_padding_ratio or max_padding_ratio=None for an explicit override "
            "(legacy allow_excess_padding=True is also accepted), or use the exact-ragged fallback; "
            "source graph blocks are indivisible."
        )

    return PackingPlan(
        config=config,
        assignment=assignment,
        blocks_by_device=blocks_by_device,
        descriptors=descriptors,
        capacities=_length_mapping(capacities),
        valid_lengths_by_device=tuple(_length_mapping(lengths) for lengths in valid_lengths),
        descriptor_capacity=descriptor_capacity,
        n_samples=block_summaries[0].n_samples,
        n_variants=sum(summary.n_variants for summary in block_summaries),
        diagnostics=diagnostics,
    )


def pack_blocks(
    blocks: Iterable[LinearARGBlockArrays],
    *,
    num_devices: int,
    dtype: Any = None,
    max_padding_ratio: float | None = 1.25,
    allow_excess_padding: bool = False,
) -> PackedGraph:
    """Pack canonical LinearARG blocks into equal-capacity host shard buffers."""
    source_blocks = tuple(blocks)
    plan = plan_packing(
        source_blocks,
        num_devices=num_devices,
        dtype=dtype,
        max_padding_ratio=max_padding_ratio,
        allow_excess_padding=allow_excess_padding,
    )
    canonical_blocks = tuple(canonicalize_block_arrays(block, dtype=plan.data_dtype) for block in source_blocks)
    buffers = _allocate_host_buffers(plan)

    descriptor_rows = [0] * plan.config.num_devices
    for descriptor in plan.descriptors:
        block = canonical_blocks[descriptor.logical_block_index]
        device = descriptor.device
        _pack_block_into_buffers(buffers, block, descriptor)
        row = descriptor_rows[device]
        buffers.block_descriptors[device, row] = descriptor.as_array_row()
        descriptor_rows[device] += 1

    for device in range(plan.config.num_devices):
        indptr_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index("indptr")])
        edge_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index("indices")])
        buffers.indptr[device, indptr_length:] = edge_length

    packed = PackedGraph(buffers=buffers, plan=plan)
    validate_packed_graph(packed)
    return packed


def validate_packed_graph(packed: PackedGraph) -> None:
    """Reject malformed descriptors, mappings, dtypes, and padding on the host."""
    buffers = packed.buffers
    plan = packed.plan
    num_devices = plan.config.num_devices
    if not isinstance(buffers, PackedHostBuffers):
        raise ValueError("packed buffers must use the fixed PackedHostBuffers component schema")

    expected_dtypes = {name: _field_dtype(name, plan.data_dtype) for name in GRAPH_FIELD_NAMES}
    for name in GRAPH_FIELD_NAMES:
        values = getattr(buffers, name)
        expected_shape = (num_devices, plan.capacities[name])
        if not isinstance(values, np.ndarray) or values.shape != expected_shape:
            raise ValueError(f"{name} shape must be {expected_shape}")
        if values.dtype != expected_dtypes[name]:
            if name == "data":
                raise ValueError(f"data dtype must match the packing plan dtype {plan.data_dtype}")
            raise ValueError(f"{name} dtype must be {expected_dtypes[name]}")

    expected_descriptor_shape = (num_devices, plan.descriptor_capacity, len(BLOCK_DESCRIPTOR_FIELDS))
    if (
        buffers.block_descriptors.shape != expected_descriptor_shape
        or buffers.block_descriptors.dtype != DEVICE_METADATA_DTYPE
    ):
        raise ValueError(f"block_descriptors must have int32 shape {expected_descriptor_shape}")
    expected_lengths_shape = (num_devices, len(VALID_LENGTH_FIELDS))
    if buffers.valid_lengths.shape != expected_lengths_shape or buffers.valid_lengths.dtype != DEVICE_METADATA_DTYPE:
        raise ValueError(f"valid_lengths must have int32 shape {expected_lengths_shape}")

    expected_valid_lengths = _device_metadata_array(
        [
            (*tuple(plan.valid_lengths_by_device[device].values()), len(plan.blocks_by_device[device]))
            for device in range(num_devices)
        ],
        name="valid_lengths",
    )
    if not np.array_equal(buffers.valid_lengths, expected_valid_lengths):
        raise ValueError("valid_lengths do not match the packing plan")

    descriptors = _validated_descriptors(buffers, plan)
    _validate_graph_values(buffers, descriptors, plan)
    _validate_inert_padding(buffers, plan)


def unpack_packed_blocks(packed: PackedGraph) -> tuple[LinearARGBlockArrays, ...]:
    """Reconstruct canonical source blocks in their original logical order."""
    validate_packed_graph(packed)
    blocks: dict[int, LinearARGBlockArrays] = {}
    for descriptor in _descriptor_rows(packed.buffers, packed.plan):
        device = descriptor.device
        indptr_slice = slice(descriptor.indptr_start, descriptor.indptr_start + descriptor.indptr_length)
        edge_slice = slice(descriptor.edge_start, descriptor.edge_start + descriptor.edge_length)
        node_slice = slice(descriptor.node_start, descriptor.node_start + descriptor.node_length)
        variant_slice = slice(descriptor.variant_start, descriptor.variant_start + descriptor.variant_length)
        sample_slice = slice(descriptor.sample_start, descriptor.sample_start + descriptor.sample_length)
        blocks[descriptor.logical_block_index] = LinearARGBlockArrays(
            indptr=np.asarray(packed.buffers.indptr[device, indptr_slice] - descriptor.edge_start, dtype=np.int32),
            indices=np.asarray(packed.buffers.indices[device, edge_slice] - descriptor.node_start, dtype=np.int32),
            data=np.asarray(packed.buffers.data[device, edge_slice], dtype=packed.plan.data_dtype),
            variant_indices=np.asarray(
                packed.buffers.variant_indices[device, variant_slice] - descriptor.node_start,
                dtype=np.int32,
            ),
            flip=np.asarray(packed.buffers.flip[device, variant_slice], dtype=np.bool_),
            sample_indices=np.asarray(
                packed.buffers.sample_indices[device, sample_slice] - descriptor.node_start,
                dtype=np.int32,
            ),
            nonunique_indices=np.asarray(
                packed.buffers.nonunique_indices[device, node_slice] - descriptor.compressed_start,
                dtype=np.int32,
            ),
            allele_counts=np.asarray(packed.buffers.allele_counts[device, variant_slice], dtype=np.int32),
            n_variants=descriptor.variant_length,
            n_samples=descriptor.sample_length,
        )
    return tuple(blocks[index] for index in range(len(packed.plan.assignment)))


def _allocate_host_buffers(plan: PackingPlan) -> PackedHostBuffers:
    def shape(name: str) -> tuple[int, int]:
        return (plan.config.num_devices, plan.capacities[name])

    valid_lengths = _device_metadata_array(
        [
            (*tuple(plan.valid_lengths_by_device[device].values()), len(plan.blocks_by_device[device]))
            for device in range(plan.config.num_devices)
        ],
        name="valid_lengths",
    )
    return PackedHostBuffers(
        indptr=np.zeros(shape("indptr"), dtype=np.int32),
        indices=np.zeros(shape("indices"), dtype=np.int32),
        data=np.zeros(shape("data"), dtype=plan.data_dtype),
        variant_indices=np.zeros(shape("variant_indices"), dtype=np.int32),
        flip=np.zeros(shape("flip"), dtype=np.bool_),
        sample_indices=np.zeros(shape("sample_indices"), dtype=np.int32),
        nonunique_indices=np.zeros(shape("nonunique_indices"), dtype=np.int32),
        allele_counts=np.full(shape("allele_counts"), -1, dtype=np.int32),
        logical_variant_indices=np.full(shape("logical_variant_indices"), -1, dtype=np.int32),
        block_descriptors=np.full(
            (plan.config.num_devices, plan.descriptor_capacity, len(BLOCK_DESCRIPTOR_FIELDS)),
            -1,
            dtype=DEVICE_METADATA_DTYPE,
        ),
        valid_lengths=valid_lengths,
    )


def _pack_block_into_buffers(
    buffers: PackedHostBuffers,
    block: LinearARGBlockArrays,
    descriptor: BlockDescriptor,
) -> None:
    nonunique_indices = block.nonunique_indices
    allele_counts = block.allele_counts
    assert nonunique_indices is not None
    assert allele_counts is not None
    device = descriptor.device
    indptr_slice = slice(descriptor.indptr_start, descriptor.indptr_start + descriptor.indptr_length)
    edge_slice = slice(descriptor.edge_start, descriptor.edge_start + descriptor.edge_length)
    node_slice = slice(descriptor.node_start, descriptor.node_start + descriptor.node_length)
    variant_slice = slice(descriptor.variant_start, descriptor.variant_start + descriptor.variant_length)
    sample_slice = slice(descriptor.sample_start, descriptor.sample_start + descriptor.sample_length)
    buffers.indptr[device, indptr_slice] = block.indptr + descriptor.edge_start
    buffers.indices[device, edge_slice] = block.indices + descriptor.node_start
    buffers.data[device, edge_slice] = block.data
    buffers.variant_indices[device, variant_slice] = block.variant_indices + descriptor.node_start
    buffers.flip[device, variant_slice] = block.flip
    buffers.sample_indices[device, sample_slice] = block.sample_indices + descriptor.node_start
    buffers.nonunique_indices[device, node_slice] = nonunique_indices + descriptor.compressed_start
    buffers.allele_counts[device, variant_slice] = allele_counts
    buffers.logical_variant_indices[device, variant_slice] = np.arange(
        descriptor.logical_variant_start,
        descriptor.logical_variant_stop,
        dtype=np.int32,
    )


def _descriptor_rows(buffers: PackedHostBuffers, plan: PackingPlan) -> tuple[BlockDescriptor, ...]:
    descriptors = []
    block_count_index = VALID_LENGTH_FIELDS.index("block_descriptors")
    for device in range(plan.config.num_devices):
        block_count = int(buffers.valid_lengths[device, block_count_index])
        for row in buffers.block_descriptors[device, :block_count]:
            values = {name: int(row[index]) for index, name in enumerate(BLOCK_DESCRIPTOR_FIELDS)}
            descriptors.append(BlockDescriptor(device=device, **values))
    return tuple(descriptors)


def _validated_descriptors(buffers: PackedHostBuffers, plan: PackingPlan) -> tuple[BlockDescriptor, ...]:
    descriptors = _descriptor_rows(buffers, plan)
    logical_indices = [descriptor.logical_block_index for descriptor in descriptors]
    if sorted(logical_indices) != list(range(len(plan.assignment))):
        raise ValueError("block assignments must be complete and non-overlapping")

    for descriptor in descriptors:
        device = descriptor.device
        lengths = {
            name: int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index(name)]) for name in GRAPH_FIELD_NAMES
        }
        _validate_span(descriptor.indptr_start, descriptor.indptr_length, lengths["indptr"], name="indptr")
        _validate_span(descriptor.edge_start, descriptor.edge_length, lengths["indices"], name="edge")
        _validate_span(descriptor.node_start, descriptor.node_length, lengths["nonunique_indices"], name="node")
        _validate_span(descriptor.variant_start, descriptor.variant_length, lengths["variant_indices"], name="variant")
        _validate_span(descriptor.sample_start, descriptor.sample_length, lengths["sample_indices"], name="sample")
        if descriptor.indptr_length != descriptor.node_length + 1:
            raise ValueError("descriptor indptr length must equal node length plus one")
        if descriptor.sample_length != plan.n_samples:
            raise ValueError("descriptor sample count must match the packing plan")
        if descriptor.logical_variant_stop - descriptor.logical_variant_start != descriptor.variant_length:
            raise ValueError("descriptor logical variant span must match its variant count")
        if descriptor.compressed_start < 0 or descriptor.compressed_length < 0:
            raise ValueError("descriptor compressed-row extent must be nonnegative")
        if not descriptor.node_start <= descriptor.min_index_to_keep < descriptor.node_start + descriptor.node_length:
            raise ValueError("descriptor min_index_to_keep must lie within its node span")
        if plan.assignment[descriptor.logical_block_index] != descriptor.device:
            raise ValueError("descriptor assignment does not match the packing plan")

    for device in range(plan.config.num_devices):
        device_descriptors = [descriptor for descriptor in descriptors if descriptor.device == device]
        valid = {
            name: int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index(name)]) for name in GRAPH_FIELD_NAMES
        }
        _validate_partition(device_descriptors, "indptr_start", "indptr_length", valid["indptr"], name="indptr")
        _validate_partition(device_descriptors, "edge_start", "edge_length", valid["indices"], name="edge")
        _validate_partition(device_descriptors, "node_start", "node_length", valid["nonunique_indices"], name="node")
        _validate_partition(
            device_descriptors,
            "variant_start",
            "variant_length",
            valid["variant_indices"],
            name="variant",
        )
        _validate_partition(
            device_descriptors,
            "sample_start",
            "sample_length",
            valid["sample_indices"],
            name="sample",
        )
        block_count = len(device_descriptors)
        if not np.all(buffers.block_descriptors[device, block_count:] == -1):
            raise ValueError("non-inert block descriptor padding")

    expected_by_logical_index = {descriptor.logical_block_index: descriptor for descriptor in plan.descriptors}
    for descriptor in descriptors:
        if descriptor != expected_by_logical_index[descriptor.logical_block_index]:
            raise ValueError("packed descriptor does not match the immutable packing plan")
    return descriptors


def _validate_span(start: int, length: int, valid_length: int, *, name: str) -> None:
    if start < 0 or length < 0 or start + length > valid_length:
        raise ValueError(f"descriptor {name} span is out of range")


def _validate_partition(
    descriptors: list[BlockDescriptor],
    start_name: str,
    length_name: str,
    valid_length: int,
    *,
    name: str,
) -> None:
    expected_start = 0
    for descriptor in descriptors:
        start = getattr(descriptor, start_name)
        length = getattr(descriptor, length_name)
        if start != expected_start:
            raise ValueError(f"descriptor {name} spans must be non-overlapping and gap-free")
        expected_start += length
    if expected_start != valid_length:
        raise ValueError(f"descriptor {name} spans must cover every valid row")


def _validate_graph_values(
    buffers: PackedHostBuffers,
    descriptors: tuple[BlockDescriptor, ...],
    plan: PackingPlan,
) -> None:
    logical_values = []
    for descriptor in descriptors:
        device = descriptor.device
        indptr = buffers.indptr[
            device,
            descriptor.indptr_start : descriptor.indptr_start + descriptor.indptr_length,
        ]
        if (
            indptr[0] != descriptor.edge_start
            or indptr[-1] != descriptor.edge_start + descriptor.edge_length
            or np.any(np.diff(indptr) < 0)
        ):
            raise ValueError("packed indptr is inconsistent with its descriptor edge span")
        edge_slice = slice(descriptor.edge_start, descriptor.edge_start + descriptor.edge_length)
        indices = buffers.indices[device, edge_slice]
        if indices.size and (
            int(indices.min()) < descriptor.node_start
            or int(indices.max()) >= descriptor.node_start + descriptor.node_length
        ):
            raise ValueError("packed edge indices leave their descriptor node span")
        data = buffers.data[device, edge_slice]
        _validate_edge_order(indptr, indices, data, node_start=descriptor.node_start)
        if not np.all(np.isfinite(data)):
            raise ValueError("packed data must contain only finite values")

        variant_slice = slice(descriptor.variant_start, descriptor.variant_start + descriptor.variant_length)
        variant_indices = buffers.variant_indices[device, variant_slice]
        sample_slice = slice(descriptor.sample_start, descriptor.sample_start + descriptor.sample_length)
        sample_indices = buffers.sample_indices[device, sample_slice]
        for name, values in (("variant", variant_indices), ("sample", sample_indices)):
            if values.size and (
                int(values.min()) < descriptor.node_start
                or int(values.max()) >= descriptor.node_start + descriptor.node_length
            ):
                raise ValueError(f"packed {name} indices leave their descriptor node span")
        node_slice = slice(descriptor.node_start, descriptor.node_start + descriptor.node_length)
        compressed = buffers.nonunique_indices[device, node_slice]
        if compressed.size and (
            int(compressed.min()) < descriptor.compressed_start
            or int(compressed.max()) >= descriptor.compressed_start + descriptor.compressed_length
        ):
            raise ValueError("packed nonunique indices leave their descriptor compressed-row extent")

        mapping = buffers.logical_variant_indices[device, variant_slice]
        expected_mapping = np.arange(
            descriptor.logical_variant_start,
            descriptor.logical_variant_stop,
            dtype=np.int32,
        )
        if not np.array_equal(mapping, expected_mapping):
            raise ValueError("logical variant mapping must be bijective and preserve block order")
        logical_values.append(mapping)

    all_logical_values = np.concatenate(logical_values) if logical_values else np.empty(0, dtype=np.int32)
    if not np.array_equal(np.sort(all_logical_values), np.arange(plan.n_variants, dtype=np.int32)):
        raise ValueError("logical variant mapping must be bijective")


def _validate_inert_padding(buffers: PackedHostBuffers, plan: PackingPlan) -> None:
    pad_values: dict[str, int | float | bool] = {
        "indices": 0,
        "data": 0,
        "variant_indices": 0,
        "flip": False,
        "sample_indices": 0,
        "nonunique_indices": 0,
        "allele_counts": -1,
        "logical_variant_indices": -1,
    }
    for device in range(plan.config.num_devices):
        edge_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index("indices")])
        indptr_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index("indptr")])
        if not np.all(buffers.indptr[device, indptr_length:] == edge_length):
            raise ValueError("non-inert indptr padding; padded CSC columns must be edge-free")
        for name, pad_value in pad_values.items():
            valid_length = int(buffers.valid_lengths[device, VALID_LENGTH_FIELDS.index(name)])
            if not np.all(getattr(buffers, name)[device, valid_length:] == pad_value):
                raise ValueError(f"non-inert {name} padding")


def _resolve_data_dtype(blocks: tuple[LinearARGBlockArrays, ...], *, dtype: Any) -> np.dtype[Any]:
    if dtype is not None:
        resolved = np.dtype(dtype)
    else:
        source_dtypes = tuple(np.asarray(block.data).dtype for block in blocks)
        if any(source_dtype != source_dtypes[0] for source_dtype in source_dtypes[1:]):
            raise ValueError("all blocks must have the same data dtype unless dtype is requested explicitly")
        resolved = source_dtypes[0]
    if not np.issubdtype(resolved, np.floating):
        raise ValueError("data dtype must be a floating dtype")
    return resolved


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _int32_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integers")
    if array.size:
        limits = np.iinfo(np.int32)
        if int(array.min()) < limits.min or int(array.max()) > limits.max:
            raise ValueError(f"{name} values must fit in int32")
    return np.asarray(array, dtype=np.int32)


def _bool_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != np.bool_ and (not np.issubdtype(array.dtype, np.integer) or np.any((array != 0) & (array != 1))):
        raise ValueError(f"{name} must contain Boolean values")
    return np.asarray(array, dtype=np.bool_)


def _data_array(values: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("data must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("data must contain numeric values")
    result = np.asarray(array, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError("data must contain only finite values")
    return result


def _validate_canonical_block(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    variant_indices: np.ndarray,
    flip: np.ndarray,
    sample_indices: np.ndarray,
    nonunique_indices: np.ndarray,
    allele_counts: np.ndarray,
    n_variants: int,
    n_samples: int,
) -> None:
    if indptr.size < 2:
        raise ValueError("indptr must describe at least one graph node")
    if indptr[0] != 0:
        raise ValueError("indptr must start at zero")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be nondecreasing")
    if int(indptr[-1]) != indices.size or indices.size != data.size:
        raise ValueError("indptr, indices, and data must describe the same edge count")

    node_count = indptr.size - 1
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= node_count):
        raise ValueError("indices contains an out-of-range node index")
    _validate_edge_order(indptr, indices, data, node_start=0)
    if variant_indices.size != n_variants:
        raise ValueError("variant_indices length must match n_variants")
    if flip.size != n_variants:
        raise ValueError("flip length must match n_variants")
    if allele_counts.size != n_variants:
        raise ValueError("allele_counts length must match n_variants")
    if sample_indices.size != n_samples:
        raise ValueError("sample_indices length must match n_samples")
    if nonunique_indices.size != node_count:
        raise ValueError("nonunique_indices length must match the graph node count")

    for name, values in (("variant_indices", variant_indices), ("sample_indices", sample_indices)):
        if values.size and (int(values.min()) < 0 or int(values.max()) >= node_count):
            raise ValueError(f"{name} contains an out-of-range node index")
    if nonunique_indices.size and int(nonunique_indices.min()) < 0:
        raise ValueError("nonunique_indices must be nonnegative")
    if sample_indices.size != np.unique(sample_indices).size:
        raise ValueError("sample_indices must be unique")


def _validate_edge_order(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    *,
    node_start: int,
) -> None:
    source_indices = node_start + np.repeat(
        np.arange(indptr.size - 1, dtype=DEVICE_METADATA_DTYPE),
        np.diff(indptr),
    )
    invalid_edge_order = (indices < source_indices) | ((indices == source_indices) & (data != 0))
    if indices.size and np.any(invalid_edge_order):
        raise ValueError("indices must be greater than their source nodes")


def _summarize_canonical_block(block: LinearARGBlockArrays) -> _BlockPackingSummary:
    nonunique_indices = block.nonunique_indices
    assert nonunique_indices is not None
    metrics = _block_metrics(block)
    return _BlockPackingSummary(
        field_lengths=metrics.field_lengths,
        data_dtype=np.dtype(block.data.dtype),
        n_samples=block.n_samples,
        n_variants=block.n_variants,
        min_index_to_keep=int(block.sample_indices[-1]) if block.sample_indices.size else 0,
        compressed_length=int(nonunique_indices.max()) + 1 if nonunique_indices.size else 0,
    )


def _finalize_staged_descriptor(
    planned: BlockDescriptor,
    block: LinearARGBlockArrays,
) -> BlockDescriptor:
    """Validate one staged block against its plan and finalize compressed extent."""
    summary = _summarize_canonical_block(block)
    expected_lengths = (
        planned.indptr_length,
        planned.edge_length,
        planned.edge_length,
        planned.variant_length,
        planned.variant_length,
        planned.sample_length,
        planned.node_length,
        planned.variant_length,
        planned.variant_length,
    )
    if summary.field_lengths != expected_lengths:
        raise ValueError("staged block field lengths do not match the packing plan")
    if planned.min_index_to_keep != planned.node_start + summary.min_index_to_keep:
        raise ValueError("staged block minimum retained index does not match the packing plan")
    compressed_length = summary.compressed_length
    assert compressed_length is not None
    if compressed_length > planned.compressed_length:
        raise ValueError("staged block compressed extent exceeds its reserved packing span")
    finalized = replace(planned, compressed_length=compressed_length)
    _device_metadata_array(finalized.as_array_row(), name="block descriptor")
    return finalized


def _block_metrics(block: LinearARGBlockArrays) -> _BlockMetrics:
    nonunique_indices = block.nonunique_indices
    allele_counts = block.allele_counts
    assert nonunique_indices is not None
    assert allele_counts is not None
    field_lengths = (
        block.indptr.size,
        block.indices.size,
        block.data.size,
        block.variant_indices.size,
        block.flip.size,
        block.sample_indices.size,
        nonunique_indices.size,
        allele_counts.size,
        block.n_variants,
    )
    canonical_bytes = sum(
        length * _field_dtype(name, block.data.dtype).itemsize
        for name, length in zip(GRAPH_FIELD_NAMES, field_lengths, strict=True)
    )
    solve_work = (block.indptr.size - 1) + block.indices.size
    return _BlockMetrics(
        field_lengths=field_lengths,
        canonical_bytes=canonical_bytes,
        solve_work=solve_work,
        score_load=canonical_bytes + np.dtype(np.int32).itemsize * solve_work,
    )


def _summary_metrics(summary: _BlockPackingSummary) -> _BlockMetrics:
    canonical_bytes = sum(
        length * _field_dtype(name, summary.data_dtype).itemsize
        for name, length in zip(GRAPH_FIELD_NAMES, summary.field_lengths, strict=True)
    )
    node_count = summary.field_lengths[GRAPH_FIELD_NAMES.index("indptr")] - 1
    edge_count = summary.field_lengths[GRAPH_FIELD_NAMES.index("indices")]
    solve_work = node_count + edge_count
    return _BlockMetrics(
        field_lengths=summary.field_lengths,
        canonical_bytes=canonical_bytes,
        solve_work=solve_work,
        score_load=canonical_bytes + np.dtype(np.int32).itemsize * solve_work,
    )


def _field_dtype(name: str, data_dtype: np.dtype[Any]) -> np.dtype[Any]:
    if name == "data":
        return np.dtype(data_dtype)
    if name == "flip":
        return np.dtype(np.bool_)
    return np.dtype(np.int32)


def _initial_assignment(metrics: tuple[_BlockMetrics, ...], *, num_devices: int) -> tuple[int, ...]:
    score_loads = [0] * num_devices
    byte_loads = [0] * num_devices
    work_loads = [0] * num_devices
    assignment = [-1] * len(metrics)
    order = sorted(range(len(metrics)), key=lambda index: (-metrics[index].score_load, index))
    for index in order:
        metric = metrics[index]
        device = min(
            range(num_devices),
            key=lambda candidate: (
                score_loads[candidate] + metric.score_load,
                byte_loads[candidate] + metric.canonical_bytes,
                work_loads[candidate] + metric.solve_work,
                candidate,
            ),
        )
        assignment[index] = device
        score_loads[device] += metric.score_load
        byte_loads[device] += metric.canonical_bytes
        work_loads[device] += metric.solve_work
    return tuple(assignment)


def _rebalance_assignment(
    assignment: tuple[int, ...],
    metrics: tuple[_BlockMetrics, ...],
    *,
    num_devices: int,
    data_dtype: np.dtype[Any],
) -> tuple[tuple[int, ...], int]:
    current = assignment
    current_objective = _assignment_objective(
        current,
        metrics,
        num_devices=num_devices,
        data_dtype=data_dtype,
    )
    steps = 0
    while True:
        best = current
        best_objective = current_objective
        for index, source in enumerate(current):
            for destination in range(num_devices):
                if destination == source:
                    continue
                candidate = (*current[:index], destination, *current[index + 1 :])
                objective = _assignment_objective(
                    candidate,
                    metrics,
                    num_devices=num_devices,
                    data_dtype=data_dtype,
                )
                if objective < best_objective:
                    best, best_objective = candidate, objective
        for left in range(len(current)):
            for right in range(left + 1, len(current)):
                if current[left] == current[right]:
                    continue
                candidate_values = list(current)
                candidate_values[left], candidate_values[right] = candidate_values[right], candidate_values[left]
                candidate = tuple(candidate_values)
                objective = _assignment_objective(
                    candidate,
                    metrics,
                    num_devices=num_devices,
                    data_dtype=data_dtype,
                )
                if objective < best_objective:
                    best, best_objective = candidate, objective
        if best == current:
            return current, steps
        current, current_objective = best, best_objective
        steps += 1


def _assignment_objective(
    assignment: tuple[int, ...],
    metrics: tuple[_BlockMetrics, ...],
    *,
    num_devices: int,
    data_dtype: np.dtype[Any],
) -> tuple[int, int, int, int]:
    field_lengths = [[0] * len(GRAPH_FIELD_NAMES) for _ in range(num_devices)]
    score_loads = [0] * num_devices
    byte_loads = [0] * num_devices
    for index, device in enumerate(assignment):
        metric = metrics[index]
        field_lengths[device] = [
            existing + added for existing, added in zip(field_lengths[device], metric.field_lengths, strict=True)
        ]
        score_loads[device] += metric.score_load
        byte_loads[device] += metric.canonical_bytes
    padded_bytes_per_device = sum(
        max(device[field] for device in field_lengths) * _field_dtype(name, data_dtype).itemsize
        for field, name in enumerate(GRAPH_FIELD_NAMES)
    )
    return padded_bytes_per_device, max(score_loads), max(byte_loads), sum(load * load for load in score_loads)


def _ordered_blocks_by_device(
    assignment: tuple[int, ...],
    metrics: tuple[_BlockMetrics, ...],
    *,
    num_devices: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sorted(
                (index for index, assigned_device in enumerate(assignment) if assigned_device == device),
                key=lambda index: (-metrics[index].score_load, index),
            )
        )
        for device in range(num_devices)
    )


def _field_lengths_by_device(
    blocks_by_device: tuple[tuple[int, ...], ...],
    metrics: tuple[_BlockMetrics, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            sum(metrics[index].field_lengths[field] for index in device_blocks)
            for field in range(len(GRAPH_FIELD_NAMES))
        )
        for device_blocks in blocks_by_device
    )


def _build_descriptors(
    summaries: tuple[_BlockPackingSummary, ...],
    blocks_by_device: tuple[tuple[int, ...], ...],
) -> tuple[BlockDescriptor, ...]:
    logical_variant_starts = np.insert(np.cumsum([summary.n_variants for summary in summaries]), 0, 0)
    descriptors: list[BlockDescriptor] = []
    for device, device_blocks in enumerate(blocks_by_device):
        indptr_start = edge_start = node_start = variant_start = sample_start = compressed_start = 0
        for logical_index in device_blocks:
            summary = summaries[logical_index]
            indptr_length = summary.field_lengths[GRAPH_FIELD_NAMES.index("indptr")]
            edge_length = summary.field_lengths[GRAPH_FIELD_NAMES.index("indices")]
            node_length = summary.field_lengths[GRAPH_FIELD_NAMES.index("nonunique_indices")]
            variant_length = summary.n_variants
            sample_length = summary.n_samples
            compressed_length = summary.compressed_length
            reserved_compressed_length = node_length if compressed_length is None else compressed_length
            descriptor = BlockDescriptor(
                device=device,
                logical_block_index=logical_index,
                indptr_start=indptr_start,
                indptr_length=indptr_length,
                edge_start=edge_start,
                edge_length=edge_length,
                node_start=node_start,
                node_length=node_length,
                variant_start=variant_start,
                variant_length=variant_length,
                sample_start=sample_start,
                sample_length=sample_length,
                compressed_start=compressed_start,
                compressed_length=reserved_compressed_length,
                min_index_to_keep=node_start + summary.min_index_to_keep,
                logical_variant_start=int(logical_variant_starts[logical_index]),
                logical_variant_stop=int(logical_variant_starts[logical_index + 1]),
            )
            _device_metadata_array(descriptor.as_array_row(), name="block descriptor")
            descriptors.append(descriptor)
            indptr_start += indptr_length
            edge_start += edge_length
            node_start += node_length
            variant_start += variant_length
            sample_start += sample_length
            compressed_start += reserved_compressed_length
    return tuple(descriptors)


def _device_metadata_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integers")
    if array.size:
        limits = np.iinfo(np.int32)
        if int(array.min()) < limits.min or int(array.max()) > limits.max:
            raise ValueError(f"{name} values must fit in int32 device metadata")
    return np.asarray(array, dtype=DEVICE_METADATA_DTYPE)


def _length_mapping(lengths: tuple[int, ...]) -> Mapping[str, int]:
    return MappingProxyType(dict(zip(GRAPH_FIELD_NAMES, lengths, strict=True)))
