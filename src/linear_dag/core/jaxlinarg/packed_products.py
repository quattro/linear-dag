# pattern: Mixed (unavoidable)
# Reason: Fixed-shape packed LinearARG algebra and explicit device-local mesh
# execution must share one boundary so graph operands remain visible to JAX.

"""Private packed LinearARG products with explicit graph operands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Generic, TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, ArrayLike

from .kernels.pure_jax import (
    pure_jax_solve_backward_compressed,
    pure_jax_solve_forward_compressed,
)
from .operator import _as_rank2_matrix
from .packing import (
    _packed_graph_sharding_spec,
    BLOCK_DESCRIPTOR_FIELDS,
    GRAPH_FIELD_NAMES,
    VALID_LENGTH_FIELDS,
)

if TYPE_CHECKING:
    from .ingress import _PackedJaxLinearARG


_DESCRIPTOR_INDEX = {name: index for index, name in enumerate(BLOCK_DESCRIPTOR_FIELDS)}
_VALID_LENGTH_INDEX = {name: index for index, name in enumerate(VALID_LENGTH_FIELDS)}
_CAPACITY_INDEX = {name: index for index, name in enumerate(GRAPH_FIELD_NAMES)}


def lineararg_matmat(
    operator: _PackedJaxLinearARG,
    values: ArrayLike,
    *,
    out_sharding: NamedSharding | None = None,
) -> Array:
    r"""Multiply a dense operand by an explicitly supplied packed graph.

    !!! info
        The packed carrier must be an explicit argument to compiled code. Raw
        closure capture such as ``jax.jit(lambda x: operator.matmat(x))`` is
        outside the graph-memory guarantee. Use this function with `operator`
        as an argument, or use `operator.compile_matmat()`.

    **Arguments:**

    - `operator`: Private packed LinearARG carrier.
    - `values`: Rank-1 or rank-2 values with leading dimension equal to the
      logical variant count.
    - `out_sharding`: Optional output `NamedSharding`. `None` defaults to a
      result replicated across the carrier graph mesh. For a rank-1 result,
      accepted specs are `P()`, `P(None)`, and `P("graph")`.
      For a rank-2 result, `P(None, None)` and `P("graph", None)` are also
      accepted. Shorter specs imply replicated trailing dimensions. The
      sharding must use the carrier graph mesh, and `"graph"` may appear only
      on the leading sample axis. A graph-sharded leading axis requires the
      logical sample count to be divisible by the graph mesh size.

    **Returns:**

    - Sample-space product with the input rank convention preserved.

    **Raises:**

    - `ValueError`: If the operand shape or packed carrier contract is invalid;
      if `out_sharding` is not a `NamedSharding`, uses another mesh, exceeds
      the logical result rank, or shards a non-leading axis; or if the sample
      count is not divisible by the graph mesh size for graph sharding.
    """
    matrix, was_vector = _as_rank2_matrix(
        values,
        expected_rows=operator.n_variants,
        dtype=operator.data.dtype,
    )
    mesh = _operator_mesh(operator)
    graph_spec = _packed_graph_sharding_spec(operator.graph)
    output_spec = _forward_output_spec(
        operator,
        out_sharding=out_sharding,
        mesh=mesh,
        logical_result_rank=1 if was_vector else 2,
    )
    if output_spec == P():
        result = jax.jit(
            jax.shard_map(
                _replicated_matmat_rank2,
                mesh=mesh,
                in_specs=(graph_spec, P()),
                out_specs=P(),
                axis_names={"graph"},
                check_vma=True,
            )
        )(operator, matrix)
    else:
        result = jax.jit(
            jax.shard_map(
                _sample_sharded_matmat_rank2,
                mesh=mesh,
                in_specs=(graph_spec, P()),
                out_specs=P("graph"),
                axis_names={"graph"},
                check_vma=True,
            )
        )(operator, matrix)
    return result[:, 0] if was_vector else result


def lineararg_rmatmat(operator: _PackedJaxLinearARG, values: ArrayLike) -> Array:
    r"""Multiply by the transpose of an explicitly supplied packed graph.

    !!! info
        The packed carrier must be an explicit argument to compiled code. Raw
        closure capture such as ``jax.jit(lambda x: operator.rmatmat(x))`` is
        outside the graph-memory guarantee. Use this function with `operator`
        as an argument, or use `operator.compile_rmatmat()`.

    **Arguments:**

    - `operator`: Private packed LinearARG carrier.
    - `values`: Rank-1 or rank-2 values with leading dimension equal to the
      logical sample count.

    **Returns:**

    - Variant-space product in exact logical order with the input rank
      convention preserved.

    **Raises:**

    - `ValueError`: If the operand shape or packed carrier contract is invalid.
    """
    matrix, was_vector = _as_rank2_matrix(
        values,
        expected_rows=operator.n_samples,
        dtype=operator.data.dtype,
    )
    mesh = _operator_mesh(operator)
    graph_spec = _packed_graph_sharding_spec(operator.graph)
    result = jax.jit(
        jax.shard_map(
            _replicated_rmatmat_rank2,
            mesh=mesh,
            in_specs=(graph_spec, P()),
            out_specs=P(),
            axis_names={"graph"},
            check_vma=True,
        )
    )(operator, matrix)
    return result[:, 0] if was_vector else result


def compile_matmat(operator: _PackedJaxLinearARG) -> _CompiledProduct[_PackedJaxLinearARG]:
    """Return a safe compiled forward wrapper with explicit graph operands."""
    _concrete_operator_mesh(operator)
    return _CompiledProduct(operator=operator, compiled_function=_compiled_lineararg_matmat)


def compile_rmatmat(operator: _PackedJaxLinearARG) -> _CompiledProduct[_PackedJaxLinearARG]:
    """Return a safe compiled transpose wrapper with explicit graph operands."""
    _concrete_operator_mesh(operator)
    return _CompiledProduct(operator=operator, compiled_function=_compiled_lineararg_rmatmat)


_Operator = TypeVar("_Operator")


@dataclass(frozen=True)
class _CompiledProduct(Generic[_Operator]):
    """Python convenience layer that supplies graph state to module-level JIT."""

    operator: _Operator
    compiled_function: Any

    def __call__(self, values: ArrayLike) -> Array:
        return self.compiled_function(self.operator, values)

    def lower(self, values: ArrayLike) -> Any:
        """Lower with the carrier supplied as an executable argument."""
        return self.compiled_function.lower(self.operator, values)


@jax.jit
def _compiled_lineararg_matmat(operator: _PackedJaxLinearARG, values: ArrayLike) -> Array:
    return lineararg_matmat(operator, values)


@jax.jit
def _compiled_lineararg_rmatmat(operator: _PackedJaxLinearARG, values: ArrayLike) -> Array:
    return lineararg_rmatmat(operator, values)


def _replicated_matmat_rank2(operator: _PackedJaxLinearARG, values: Array) -> Array:
    return lax.psum(_local_matmat_rank2(operator, values), "graph")


def _sample_sharded_matmat_rank2(operator: _PackedJaxLinearARG, values: Array) -> Array:
    partial = _local_matmat_rank2(operator, values)
    return lax.psum_scatter(partial, "graph", scatter_dimension=0, tiled=True)


def _replicated_rmatmat_rank2(operator: _PackedJaxLinearARG, values: Array) -> Array:
    return lax.psum(_local_rmatmat_rank2(operator, values), "graph")


def _local_matmat_rank2(operator: _PackedJaxLinearARG, values: Array) -> Array:
    """Return one graph shard's sample-space partial product."""
    graph = _remove_local_graph_axis(operator)
    descriptors = graph["block_descriptors"]
    block_count = graph["valid_lengths"][_VALID_LENGTH_INDEX["block_descriptors"]]
    initial = jnp.zeros((operator.n_samples, values.shape[1]), dtype=values.dtype)
    initial = initial + block_count.astype(values.dtype) * 0

    def add_descriptor(slot: int, result: Array) -> Array:
        descriptor = descriptors[slot]
        valid = slot < block_count
        return result + _forward_descriptor_product(operator, graph, descriptor, valid, values)

    return lax.fori_loop(0, descriptors.shape[0], add_descriptor, initial)


def _local_rmatmat_rank2(operator: _PackedJaxLinearARG, values: Array) -> Array:
    """Return one graph shard's logical variant-space partial product."""
    graph = _remove_local_graph_axis(operator)
    descriptors = graph["block_descriptors"]
    block_count = graph["valid_lengths"][_VALID_LENGTH_INDEX["block_descriptors"]]
    initial = jnp.zeros((operator.n_variants, values.shape[1]), dtype=values.dtype)
    initial = initial + block_count.astype(values.dtype) * 0

    def add_descriptor(slot: int, result: Array) -> Array:
        descriptor = descriptors[slot]
        valid = slot < block_count
        return result + _reverse_descriptor_product(operator, graph, descriptor, valid, values)

    return lax.fori_loop(0, descriptors.shape[0], add_descriptor, initial)


def _forward_descriptor_product(
    operator: _PackedJaxLinearARG,
    graph: dict[str, Array],
    descriptor: Array,
    valid: Array,
    values: Array,
) -> Array:
    views = _descriptor_views(operator, graph, descriptor, valid)
    variant_capacity = views["variant_indices"].shape[0]
    variant_mask = jnp.arange(variant_capacity, dtype=jnp.int32) < views["variant_length"]
    logical_indices = jnp.where(variant_mask, views["logical_variant_indices"], 0)
    variant_nodes = jnp.where(variant_mask, views["variant_indices"], 0)
    variant_rows = views["nonunique_indices"][variant_nodes]
    flip = variant_mask & views["flip"]
    flip_sign = jnp.where(flip, -1, 1).astype(values.dtype)
    seeds = values[logical_indices, :] * flip_sign[:, None]
    seeds = jnp.where(variant_mask[:, None], seeds, jnp.zeros_like(seeds))
    compressed = jnp.zeros(
        (operator.capacities[_CAPACITY_INDEX["nonunique_indices"]], values.shape[1]),
        dtype=values.dtype,
    )
    compressed = compressed.at[variant_rows, :].add(seeds)
    solved = pure_jax_solve_forward_compressed(
        views["indptr"],
        views["indices"],
        views["data"],
        views["nonunique_indices"],
        cast(Any, views["min_index_to_keep"]),
        compressed,
    )
    sample_rows = views["nonunique_indices"][views["sample_indices"]]
    flip_sum = jnp.sum(jnp.where(flip[:, None], values[logical_indices, :], 0), axis=0)
    result = solved[sample_rows, :] + flip_sum
    return jnp.where(valid, result, jnp.zeros_like(result))


def _reverse_descriptor_product(
    operator: _PackedJaxLinearARG,
    graph: dict[str, Array],
    descriptor: Array,
    valid: Array,
    values: Array,
) -> Array:
    views = _descriptor_views(operator, graph, descriptor, valid)
    sample_rows = views["nonunique_indices"][views["sample_indices"]]
    compressed = jnp.zeros(
        (operator.capacities[_CAPACITY_INDEX["nonunique_indices"]], values.shape[1]),
        dtype=values.dtype,
    )
    sample_seeds = jnp.where(valid, values, jnp.zeros_like(values))
    compressed = compressed.at[sample_rows, :].set(sample_seeds)
    solved = pure_jax_solve_backward_compressed(
        views["indptr"],
        views["indices"],
        views["data"],
        views["nonunique_indices"],
        cast(Any, views["min_index_to_keep"]),
        compressed,
    )
    variant_capacity = views["variant_indices"].shape[0]
    variant_mask = jnp.arange(variant_capacity, dtype=jnp.int32) < views["variant_length"]
    variant_nodes = jnp.where(variant_mask, views["variant_indices"], 0)
    variant_rows = views["nonunique_indices"][variant_nodes]
    block_values = solved[variant_rows, :]
    total = jnp.sum(values, axis=0)
    block_values = jnp.where(views["flip"][:, None], total[None, :] - block_values, block_values)
    block_values = jnp.where(variant_mask[:, None], block_values, jnp.zeros_like(block_values))
    logical_indices = jnp.where(variant_mask, views["logical_variant_indices"], 0)
    result = jnp.zeros((operator.n_variants, values.shape[1]), dtype=values.dtype)
    return result.at[logical_indices, :].add(block_values)


def _descriptor_views(
    operator: _PackedJaxLinearARG,
    graph: dict[str, Array],
    descriptor: Array,
    valid: Array,
) -> dict[str, Array]:
    def value(name: str) -> Array:
        return jnp.where(valid, descriptor[_DESCRIPTOR_INDEX[name]], 0)

    indptr_start = value("indptr_start")
    indptr_length = value("indptr_length")
    edge_start = value("edge_start")
    edge_length = value("edge_length")
    node_start = value("node_start")
    node_length = value("node_length")
    variant_start = value("variant_start")
    variant_length = value("variant_length")
    sample_start = value("sample_start")
    sample_length = value("sample_length")
    compressed_start = value("compressed_start")

    indptr = _fixed_span(graph["indptr"], indptr_start, indptr_length, edge_start + edge_length)
    indptr = indptr - edge_start
    indices = _fixed_span(graph["indices"], edge_start, edge_length, node_start) - node_start
    data = _fixed_span(graph["data"], edge_start, edge_length, 0)
    nonunique = _fixed_span(graph["nonunique_indices"], node_start, node_length, compressed_start)
    nonunique = nonunique - compressed_start
    variant_indices = _fixed_span(graph["variant_indices"], variant_start, variant_length, node_start)
    variant_indices = variant_indices - node_start
    flip = _fixed_span(graph["flip"], variant_start, variant_length, False)
    logical_indices = _fixed_span(graph["logical_variant_indices"], variant_start, variant_length, 0)
    sample_indices = _fixed_span(
        graph["sample_indices"],
        sample_start,
        sample_length,
        node_start,
        size=operator.n_samples,
    )
    sample_indices = sample_indices - node_start
    min_index_to_keep = value("min_index_to_keep") - node_start
    return {
        "indptr": indptr,
        "indices": indices,
        "data": data,
        "nonunique_indices": nonunique,
        "variant_indices": variant_indices,
        "flip": flip,
        "logical_variant_indices": logical_indices,
        "sample_indices": sample_indices,
        "variant_length": variant_length,
        "min_index_to_keep": min_index_to_keep,
    }


def _fixed_span(
    values: Array,
    start: Array,
    length: Array,
    pad_value: Any,
    *,
    size: int | None = None,
) -> Array:
    result_size = values.shape[0] if size is None else size
    if result_size == 0:
        return values[:0]
    offsets = jnp.arange(result_size, dtype=jnp.int32)
    positions = jnp.clip(start + offsets, 0, max(values.shape[0] - 1, 0))
    gathered = values[positions]
    padding = jnp.asarray(pad_value, dtype=values.dtype)
    return jnp.where(offsets < length, gathered, padding)


def _remove_local_graph_axis(operator: _PackedJaxLinearARG) -> dict[str, Array]:
    return {name: getattr(operator, name)[0] for name in (*GRAPH_FIELD_NAMES, "block_descriptors", "valid_lengths")}


def _operator_mesh(operator: _PackedJaxLinearARG) -> Mesh:
    """Return concrete global execution metadata retained across outer JIT."""
    mesh = operator.graph_mesh
    if not isinstance(mesh, Mesh) or mesh.axis_names != ("graph",):
        raise ValueError('packed products require a concrete carrier mesh with the dedicated "graph" axis')
    return mesh


def _concrete_operator_mesh(operator: _PackedJaxLinearARG) -> Mesh:
    mesh = _operator_mesh(operator)
    sharding = operator.indptr.sharding
    if not isinstance(sharding, NamedSharding) or _addressable_device_count(sharding) != mesh.size:
        raise ValueError("safe compilation helpers require every graph mesh device to be addressable on one host")
    return mesh


def _addressable_device_count(sharding: NamedSharding) -> int:
    """Return the local ownership count used by single-host validation."""
    return len(sharding.addressable_devices)


def _forward_output_spec(
    operator: _PackedJaxLinearARG,
    *,
    out_sharding: NamedSharding | None,
    mesh: Mesh,
    logical_result_rank: int,
) -> P:
    if out_sharding is None:
        return P()
    if not isinstance(out_sharding, NamedSharding):
        raise ValueError("packed forward output sharding must use NamedSharding")
    if isinstance(mesh, Mesh) and out_sharding.mesh != mesh:
        raise ValueError("packed forward output sharding must use the carrier graph mesh")
    spec = out_sharding.spec
    if len(spec) > logical_result_rank:
        raise ValueError(
            f"packed forward output sharding rank {len(spec)} exceeds logical result rank {logical_result_rank}"
        )
    if all(axis is None for axis in spec):
        return P()
    if len(spec) > 0 and spec[0] == "graph" and all(axis is None for axis in spec[1:]):
        if operator.n_samples % out_sharding.mesh.size != 0:
            raise ValueError("sample leading dimension must be divisible by the graph mesh size for reduce-scatter")
        return P("graph")
    raise ValueError('sample output sharding may use "graph" only on the leading axis')


def _validate_packed_carrier(operator: _PackedJaxLinearARG) -> None:
    """Validate a concrete carrier before it can reach packed numerics."""
    arrays = {name: getattr(operator, name) for name in (*GRAPH_FIELD_NAMES, "block_descriptors", "valid_lengths")}
    first_sharding = arrays["indptr"].sharding
    if not isinstance(first_sharding, NamedSharding) or not isinstance(first_sharding.mesh, Mesh):
        raise ValueError("packed products require a concrete single-host NamedSharding mesh")
    mesh = first_sharding.mesh
    if operator.graph_mesh != mesh:
        raise ValueError("packed graph arrays must use the carrier graph mesh")
    if mesh.axis_names != ("graph",):
        raise ValueError('packed products require the dedicated mesh axis "graph"')
    if _addressable_device_count(first_sharding) != mesh.size:
        raise ValueError("packed products require every graph mesh device to be addressable on one host")

    num_devices = mesh.size
    expected_dtypes = {
        "indptr": jnp.int32,
        "indices": jnp.int32,
        "variant_indices": jnp.int32,
        "flip": jnp.bool_,
        "sample_indices": jnp.int32,
        "nonunique_indices": jnp.int32,
        "allele_counts": jnp.int32,
        "logical_variant_indices": jnp.int32,
        "block_descriptors": jnp.int32,
        "valid_lengths": jnp.int32,
    }
    for name, array in arrays.items():
        if not isinstance(array, jax.Array):
            raise ValueError(f"{name} must be a JAX array")
        if not isinstance(array.sharding, NamedSharding) or array.sharding.mesh != mesh:
            raise ValueError(f"{name} must use the packed graph mesh")
        if array.sharding.spec[0] != "graph" or any(axis is not None for axis in array.sharding.spec[1:]):
            raise ValueError(f"{name} must be sharded only on the dedicated graph axis")
        if array.shape[0] != num_devices:
            raise ValueError(f"{name} leading dimension must equal the graph mesh size")
        if name == "data":
            if not jnp.issubdtype(array.dtype, jnp.floating):
                raise ValueError("packed data dtype must be floating")
        elif array.dtype != expected_dtypes[name]:
            raise ValueError(f"{name} dtype must be {expected_dtypes[name]}")

    for name, capacity in zip(GRAPH_FIELD_NAMES, operator.capacities, strict=True):
        if arrays[name].shape != (num_devices, capacity):
            raise ValueError(f"{name} shape must match its packed capacity {capacity}")
    if arrays["block_descriptors"].ndim != 3 or arrays["block_descriptors"].shape[2] != len(BLOCK_DESCRIPTOR_FIELDS):
        raise ValueError("block_descriptors must use the fixed descriptor schema")
    if arrays["valid_lengths"].shape != (num_devices, len(VALID_LENGTH_FIELDS)):
        raise ValueError("valid_lengths must use the fixed packed-field schema")

    descriptors = np.asarray(jax.device_get(arrays["block_descriptors"]), dtype=np.int32)
    valid_lengths = np.asarray(jax.device_get(arrays["valid_lengths"]), dtype=np.int32)
    logical_mappings: list[np.ndarray] = []
    logical_block_indices: list[int] = []
    for device in range(num_devices):
        lengths = {name: int(valid_lengths[device, _VALID_LENGTH_INDEX[name]]) for name in VALID_LENGTH_FIELDS}
        for name, capacity in zip(GRAPH_FIELD_NAMES, operator.capacities, strict=True):
            if lengths[name] < 0 or lengths[name] > capacity:
                raise ValueError(f"valid {name} length is outside its packed capacity")
        block_count = lengths["block_descriptors"]
        descriptor_capacity = descriptors.shape[1]
        if block_count < 0 or block_count > descriptor_capacity:
            raise ValueError("valid block descriptor count is outside its packed capacity")
        if not np.all(descriptors[device, block_count:] == -1):
            raise ValueError("non-inert block descriptor padding")

        device_rows = descriptors[device, :block_count]
        _validate_descriptor_partitions(device_rows, lengths)
        for row in device_rows:
            descriptor = {name: int(row[index]) for index, name in enumerate(BLOCK_DESCRIPTOR_FIELDS)}
            logical_block_indices.append(descriptor["logical_block_index"])
            _validate_descriptor_metadata(descriptor, lengths, operator)
            logical_mappings.append(_validate_descriptor_graph(operator, device, descriptor))
        _validate_device_padding(operator, device, lengths)

    if sorted(logical_block_indices) != list(range(len(logical_block_indices))):
        raise ValueError("block assignments must be complete and non-overlapping")
    logical = np.concatenate(logical_mappings) if logical_mappings else np.empty(0, dtype=np.int32)
    if not np.array_equal(np.sort(logical), np.arange(operator.n_variants, dtype=np.int32)):
        raise ValueError("logical variant mapping must be bijective")


def _validate_descriptor_partitions(rows: np.ndarray, lengths: dict[str, int]) -> None:
    for start_name, length_name, field_name in (
        ("indptr_start", "indptr_length", "indptr"),
        ("edge_start", "edge_length", "indices"),
        ("node_start", "node_length", "nonunique_indices"),
        ("variant_start", "variant_length", "variant_indices"),
        ("sample_start", "sample_length", "sample_indices"),
    ):
        expected_start = 0
        for row in rows:
            start = int(row[_DESCRIPTOR_INDEX[start_name]])
            length = int(row[_DESCRIPTOR_INDEX[length_name]])
            if start != expected_start or length < 0 or start + length > lengths[field_name]:
                label = "edge" if field_name == "indices" else field_name.removesuffix("_indices")
                raise ValueError(f"descriptor {label} span is out of range or not a partition")
            expected_start += length
        if expected_start != lengths[field_name]:
            label = "edge" if field_name == "indices" else field_name.removesuffix("_indices")
            raise ValueError(f"descriptor {label} spans must cover every valid row")


def _validate_descriptor_metadata(
    descriptor: dict[str, int],
    lengths: dict[str, int],
    operator: _PackedJaxLinearARG,
) -> None:
    if descriptor["logical_block_index"] < 0:
        raise ValueError("logical block indices must be nonnegative")
    if descriptor["indptr_length"] != descriptor["node_length"] + 1:
        raise ValueError("descriptor indptr length must equal node length plus one")
    if descriptor["sample_length"] != operator.n_samples:
        raise ValueError("descriptor sample count must match the packed operator")
    if descriptor["logical_variant_stop"] - descriptor["logical_variant_start"] != descriptor["variant_length"]:
        raise ValueError("descriptor logical variant span must match its variant count")
    if not 0 <= descriptor["logical_variant_start"] <= descriptor["logical_variant_stop"] <= operator.n_variants:
        raise ValueError("descriptor logical variant span is out of range")
    if descriptor["compressed_start"] < 0 or descriptor["compressed_length"] < 0:
        raise ValueError("descriptor compressed-row extent must be nonnegative")
    if descriptor["compressed_start"] + descriptor["compressed_length"] > lengths["nonunique_indices"]:
        raise ValueError("descriptor compressed-row extent is out of range")
    if not (
        descriptor["node_start"]
        <= descriptor["min_index_to_keep"]
        < descriptor["node_start"] + descriptor["node_length"]
    ):
        raise ValueError("descriptor min_index_to_keep must lie within its node span")
    if lengths["data"] != lengths["indices"]:
        raise ValueError("packed data and edge valid lengths must match")
    for name in ("flip", "allele_counts", "logical_variant_indices"):
        if lengths[name] != lengths["variant_indices"]:
            raise ValueError(f"packed {name} valid length must match variant_indices")


def _validate_descriptor_graph(
    operator: _PackedJaxLinearARG,
    device: int,
    descriptor: dict[str, int],
) -> np.ndarray:
    edge_slice = slice(descriptor["edge_start"], descriptor["edge_start"] + descriptor["edge_length"])
    node_slice = slice(descriptor["node_start"], descriptor["node_start"] + descriptor["node_length"])
    variant_slice = slice(descriptor["variant_start"], descriptor["variant_start"] + descriptor["variant_length"])
    sample_slice = slice(descriptor["sample_start"], descriptor["sample_start"] + descriptor["sample_length"])
    indptr = operator.indptr[
        device,
        descriptor["indptr_start"] : descriptor["indptr_start"] + descriptor["indptr_length"],
    ]
    indices = operator.indices[device, edge_slice]
    data = operator.data[device, edge_slice]
    if not _device_all(
        (indptr[0] == descriptor["edge_start"])
        & (indptr[-1] == descriptor["edge_start"] + descriptor["edge_length"])
        & jnp.all(jnp.diff(indptr) >= 0)
    ):
        raise ValueError("packed indptr is inconsistent with its descriptor edge span")
    if descriptor["edge_length"] and not _device_all(
        (indices >= descriptor["node_start"]) & (indices < descriptor["node_start"] + descriptor["node_length"])
    ):
        raise ValueError("packed graph indices leave their descriptor node span")
    if not _device_all(jnp.all(jnp.isfinite(data))):
        raise ValueError("packed data must contain only finite values")

    variant_indices = operator.variant_indices[device, variant_slice]
    sample_indices = operator.sample_indices[device, sample_slice]
    for name, values in (("variant", variant_indices), ("sample", sample_indices)):
        if values.size and not _device_all(
            (values >= descriptor["node_start"]) & (values < descriptor["node_start"] + descriptor["node_length"])
        ):
            raise ValueError(f"packed {name} indices leave their descriptor node span")
    nonunique = operator.nonunique_indices[device, node_slice]
    if nonunique.size and not _device_all(
        (nonunique >= descriptor["compressed_start"])
        & (nonunique < descriptor["compressed_start"] + descriptor["compressed_length"])
    ):
        raise ValueError("packed nonunique indices leave their descriptor compressed-row extent")

    mapping = np.asarray(jax.device_get(operator.logical_variant_indices[device, variant_slice]), dtype=np.int32)
    expected = np.arange(
        descriptor["logical_variant_start"],
        descriptor["logical_variant_stop"],
        dtype=np.int32,
    )
    if not np.array_equal(mapping, expected):
        raise ValueError("logical variant mapping must be bijective and preserve block order")
    return mapping


def _validate_device_padding(
    operator: _PackedJaxLinearARG,
    device: int,
    lengths: dict[str, int],
) -> None:
    edge_length = lengths["indices"]
    indptr_padding = operator.indptr[device, lengths["indptr"] :]
    if indptr_padding.size and not _device_all(indptr_padding == edge_length):
        raise ValueError("non-inert indptr padding")
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
    for name, pad_value in pad_values.items():
        padding = getattr(operator, name)[device, lengths[name] :]
        if padding.size and not _device_all(padding == pad_value):
            raise ValueError(f"non-inert {name} padding")


def _device_all(values: Array) -> bool:
    return bool(np.asarray(jax.device_get(jnp.all(values))))
