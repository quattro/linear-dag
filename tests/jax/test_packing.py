from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.tree_util as jtu
import numpy as np
import pytest

from jax.sharding import Mesh, NamedSharding, PartitionSpec

from linear_dag.core.jaxlinarg.ingress import (
    _assemble_single_device_arrays,
    _packed_from_block_arrays,
    LinearARGBlockArrays as IngressLinearARGBlockArrays,
    read_hdf5_block_arrays,
)
from linear_dag.core.jaxlinarg.packing import (
    BLOCK_DESCRIPTOR_FIELDS,
    canonicalize_block_arrays,
    GRAPH_FIELD_NAMES,
    LinearARGBlockArrays,
    pack_blocks,
    PACKED_COMPONENT_NAMES,
    plan_packing,
    unpack_packed_blocks,
    VALID_LENGTH_FIELDS,
    validate_packed_graph,
)
from tests.jax.bench.test_parallel_benchmarks import (
    _format_results_table,
    _packed_gate_failures,
    _packed_memory_result,
)


def _two_device_graph_mesh_or_skip() -> Mesh:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip(
            "requires at least two CPU devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import"
        )
    return Mesh(np.asarray(devices[:2]), ("graph",))


def _block(
    *,
    n_nodes: int = 4,
    n_edges: int = 3,
    n_variants: int = 2,
    n_samples: int = 2,
    dtype: np.dtype | type[np.floating] = np.float32,
    block_seed: int = 0,
    optional_arrays: bool = True,
) -> LinearARGBlockArrays:
    if n_edges < n_nodes - 1:
        raise ValueError("test blocks require at least n_nodes - 1 edges")
    edge_columns = np.minimum(np.arange(n_edges, dtype=np.int64), n_nodes - 1)
    indptr = np.searchsorted(edge_columns, np.arange(n_nodes + 1), side="left")
    indices = (np.arange(n_edges, dtype=np.int64) + block_seed) % n_nodes
    variant_indices = np.arange(n_variants, dtype=np.int64) % n_nodes
    sample_indices = np.arange(n_nodes - 1, n_nodes - n_samples - 1, -1, dtype=np.int64)
    return LinearARGBlockArrays(
        indptr=indptr,
        indices=indices,
        data=np.linspace(1.0, 2.0, n_edges, dtype=dtype),
        variant_indices=variant_indices,
        flip=np.arange(n_variants) % 2 == 1,
        sample_indices=sample_indices,
        nonunique_indices=np.arange(n_nodes, dtype=np.int64) if optional_arrays else None,
        allele_counts=np.arange(n_variants, dtype=np.int64) if optional_arrays else None,
        n_variants=n_variants,
        n_samples=n_samples,
    )


def _canonical_graph_bytes(block: LinearARGBlockArrays) -> int:
    return (
        sum(getattr(block, name).nbytes for name in GRAPH_FIELD_NAMES if name != "logical_variant_indices")
        + block.n_variants * np.dtype(np.int32).itemsize
    )


def _assert_block_arrays_equal(actual: LinearARGBlockArrays, expected: LinearARGBlockArrays) -> None:
    for name in (
        "indptr",
        "indices",
        "data",
        "variant_indices",
        "flip",
        "sample_indices",
        "nonunique_indices",
        "allele_counts",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)
    assert actual.n_variants == expected.n_variants
    assert actual.n_samples == expected.n_samples


def _replace_buffer(packed, name: str, values: np.ndarray):
    return replace(packed, buffers=packed.buffers._replace(**{name: values}))


def test_lineararg_block_arrays_remains_available_from_ingress() -> None:
    assert IngressLinearARGBlockArrays is LinearARGBlockArrays


def test_canonicalize_block_arrays_normalizes_every_graph_field() -> None:
    canonical = canonicalize_block_arrays(_block(dtype=np.float64, optional_arrays=False), dtype=np.float32)

    assert canonical.indptr.dtype == np.int32
    assert canonical.indices.dtype == np.int32
    assert canonical.data.dtype == np.float32
    assert canonical.variant_indices.dtype == np.int32
    assert canonical.flip.dtype == np.bool_
    assert canonical.sample_indices.dtype == np.int32
    assert canonical.nonunique_indices is not None
    assert canonical.nonunique_indices.dtype == np.int32
    assert canonical.allele_counts is not None
    assert canonical.allele_counts.dtype == np.int32
    np.testing.assert_array_equal(canonical.nonunique_indices, np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(canonical.allele_counts, np.full(2, -1, dtype=np.int32))


@pytest.mark.parametrize(
    "blocks",
    [
        (_block(n_nodes=7, n_edges=6, n_variants=4),),
        (
            _block(n_nodes=4, n_edges=3, n_variants=2),
            _block(n_nodes=3, n_edges=3, n_variants=2, block_seed=1),
        ),
        tuple(_block(n_nodes=3, n_edges=2, n_variants=1, block_seed=i) for i in range(4)),
    ],
)
def test_plan_uses_one_fixed_component_and_descriptor_schema(blocks) -> None:
    plan = plan_packing(blocks, num_devices=1)

    assert plan.component_names == PACKED_COMPONENT_NAMES
    assert plan.descriptor_schema == BLOCK_DESCRIPTOR_FIELDS
    assert tuple(plan.capacities) == GRAPH_FIELD_NAMES
    assert tuple(plan.valid_lengths_by_device[0]) == GRAPH_FIELD_NAMES
    assert VALID_LENGTH_FIELDS == (*GRAPH_FIELD_NAMES, "block_descriptors")


def test_plan_byte_accounting_counts_mapping_and_excludes_descriptors() -> None:
    blocks = (_block(n_nodes=5, n_edges=7, n_variants=3), _block(n_nodes=3, n_edges=2, n_variants=1))
    canonical = tuple(canonicalize_block_arrays(block) for block in blocks)

    plan = plan_packing(blocks, num_devices=1)

    expected = sum(_canonical_graph_bytes(block) for block in canonical)
    assert plan.diagnostics.canonical_graph_bytes == expected
    assert plan.diagnostics.padded_graph_bytes == expected
    assert plan.diagnostics.descriptor_bytes > 0
    assert plan.diagnostics.padding_ratio == 1.0


def test_assignment_is_stable_complete_and_nonoverlapping() -> None:
    blocks = tuple(_block(block_seed=i) for i in range(4))

    first = plan_packing(blocks, num_devices=2)
    second = plan_packing(blocks, num_devices=2)

    assert first.assignment == second.assignment == (0, 1, 0, 1)
    assigned = tuple(block for device in first.blocks_by_device for block in device)
    assert sorted(assigned) == list(range(len(blocks)))
    assert len(set(assigned)) == len(blocks)


def test_plan_permits_empty_device_assignments_with_explicit_override() -> None:
    plan = plan_packing((_block(),), num_devices=3, allow_excess_padding=True)

    assert len(plan.blocks_by_device) == 3
    assert sum(not blocks for blocks in plan.blocks_by_device) == 2
    assert plan.diagnostics.padding_override
    assert plan.diagnostics.padding_ratio > 1.25


@pytest.mark.parametrize("num_devices", [0, -1])
def test_plan_rejects_nonpositive_device_count(num_devices: int) -> None:
    with pytest.raises(ValueError, match="num_devices must be at least 1"):
        plan_packing((_block(),), num_devices=num_devices)


def test_plan_rejects_inconsistent_sample_counts() -> None:
    with pytest.raises(ValueError, match="same n_samples"):
        plan_packing((_block(n_samples=2), _block(n_samples=1)), num_devices=1)


def test_plan_rejects_inconsistent_implicit_data_dtypes() -> None:
    with pytest.raises(ValueError, match="same data dtype"):
        plan_packing((_block(dtype=np.float32), _block(dtype=np.float64)), num_devices=1)

    plan = plan_packing(
        (_block(dtype=np.float32), _block(dtype=np.float64)),
        num_devices=1,
        dtype=np.float32,
    )
    assert plan.data_dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda block: replace(block, n_variants=block.n_variants + 1), "n_variants"),
        (lambda block: replace(block, indptr=np.array([1, 1, 2, 3, 3])), "indptr must start at zero"),
        (lambda block: replace(block, indices=np.array([0, 1, 99])), "indices contains an out-of-range"),
        (lambda block: replace(block, flip=np.array([True])), "flip"),
        (lambda block: replace(block, nonunique_indices=np.array([0, 1])), "nonunique_indices"),
    ],
)
def test_plan_rejects_invalid_block_metadata(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plan_packing((mutate(_block()),), num_devices=1)


def test_skewed_plan_rejects_with_complete_diagnostics_and_override() -> None:
    blocks = (_block(n_nodes=20, n_edges=30, n_variants=10), _block(n_nodes=2, n_edges=1, n_variants=1))

    with pytest.raises(ValueError) as exc_info:
        plan_packing(blocks, num_devices=2)

    message = str(exc_info.value)
    assert "canonical bytes=" in message
    assert "padded bytes=" in message
    assert "padding ratio=" in message
    assert "per-device loads=" in message
    assert "allow_excess_padding=True" in message
    assert "exact-ragged fallback" in message

    plan = plan_packing(blocks, num_devices=2, allow_excess_padding=True)
    assert plan.diagnostics.padding_ratio > 1.25
    assert plan.diagnostics.padding_override
    assert len(plan.diagnostics.device_graph_bytes) == 2
    assert len(plan.diagnostics.device_solve_work) == 2


def test_bundled_two_block_fixture_requires_explicit_padding_override(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list())
    blocks = tuple(read_hdf5_block_arrays(linarg_h5_path, name) for name in block_names)

    with pytest.raises(ValueError, match="allow_excess_padding=True"):
        plan_packing(blocks, num_devices=2)

    plan = plan_packing(blocks, num_devices=2, allow_excess_padding=True)
    assert plan.diagnostics.padding_ratio > 1.25


@pytest.mark.parametrize(
    "blocks",
    [
        (_block(n_nodes=7, n_edges=6, n_variants=4),),
        (_block(), _block(n_nodes=3, n_edges=3, n_variants=1)),
        tuple(_block(block_seed=i) for i in range(4)),
    ],
)
def test_packed_buffers_have_one_fixed_pytree_definition(blocks) -> None:
    packed = pack_blocks(blocks, num_devices=1)

    assert packed.buffers._fields == PACKED_COMPONENT_NAMES
    assert len(jtu.tree_leaves(packed.buffers)) == len(PACKED_COMPONENT_NAMES)
    assert jtu.tree_structure(packed.buffers) == jtu.tree_structure(pack_blocks((_block(),), num_devices=1).buffers)


def test_pack_unpack_round_trip_preserves_synthetic_canonical_fields() -> None:
    first = replace(
        _block(block_seed=3),
        flip=np.array([True, False]),
        nonunique_indices=np.array([0, 1, 1, 2]),
        allele_counts=np.array([7, 3]),
    )
    second = _block(n_nodes=3, n_edges=4, n_variants=1, block_seed=2, optional_arrays=False)
    expected = tuple(canonicalize_block_arrays(block) for block in (first, second))

    packed = pack_blocks((first, second), num_devices=1)
    actual = unpack_packed_blocks(packed)

    assert len(actual) == len(expected)
    for actual_block, expected_block in zip(actual, expected, strict=True):
        _assert_block_arrays_equal(actual_block, expected_block)


def test_pack_unpack_round_trip_preserves_hdf5_fixture_blocks(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list())
    blocks = tuple(read_hdf5_block_arrays(linarg_h5_path, name) for name in block_names)
    expected = tuple(canonicalize_block_arrays(block) for block in blocks)

    packed = pack_blocks(blocks, num_devices=2, allow_excess_padding=True)
    actual = unpack_packed_blocks(packed)

    for actual_block, expected_block in zip(actual, expected, strict=True):
        _assert_block_arrays_equal(actual_block, expected_block)


def test_unpack_restores_logical_order_after_physical_block_reordering() -> None:
    small = _block(n_nodes=3, n_edges=2, n_variants=1, block_seed=1)
    large = _block(n_nodes=9, n_edges=12, n_variants=6, block_seed=2)

    packed = pack_blocks((small, large), num_devices=1)
    unpacked = unpack_packed_blocks(packed)

    assert packed.plan.blocks_by_device == ((1, 0),)
    _assert_block_arrays_equal(unpacked[0], canonicalize_block_arrays(small))
    _assert_block_arrays_equal(unpacked[1], canonicalize_block_arrays(large))
    mapping = packed.buffers.logical_variant_indices[0]
    valid = packed.buffers.valid_lengths[0, VALID_LENGTH_FIELDS.index("logical_variant_indices")]
    np.testing.assert_array_equal(np.sort(mapping[:valid]), np.arange(small.n_variants + large.n_variants))


def test_empty_device_buffers_and_all_padding_are_inert() -> None:
    packed = pack_blocks((_block(),), num_devices=3, allow_excess_padding=True)

    validate_packed_graph(packed)
    empty_devices = [device for device, blocks in enumerate(packed.plan.blocks_by_device) if not blocks]
    assert len(empty_devices) == 2
    for device in empty_devices:
        assert np.all(packed.buffers.valid_lengths[device] == 0)
        assert np.all(packed.buffers.indptr[device] == 0)
        assert np.all(packed.buffers.indices[device] == 0)
        assert np.all(packed.buffers.data[device] == 0)
        assert np.all(packed.buffers.variant_indices[device] == 0)
        assert np.all(~packed.buffers.flip[device])
        assert np.all(packed.buffers.sample_indices[device] == 0)
        assert np.all(packed.buffers.nonunique_indices[device] == 0)
        assert np.all(packed.buffers.allele_counts[device] == -1)
        assert np.all(packed.buffers.logical_variant_indices[device] == -1)
        assert np.all(packed.buffers.block_descriptors[device] == -1)


def test_padding_is_masked_edge_free_and_index_safe() -> None:
    packed = pack_blocks(
        (_block(n_nodes=10, n_edges=15, n_variants=5), _block(n_nodes=3, n_edges=2, n_variants=1)),
        num_devices=2,
        allow_excess_padding=True,
    )
    lengths = packed.buffers.valid_lengths

    for device in range(2):
        edge_length = lengths[device, VALID_LENGTH_FIELDS.index("indices")]
        indptr_length = lengths[device, VALID_LENGTH_FIELDS.index("indptr")]
        variant_length = lengths[device, VALID_LENGTH_FIELDS.index("variant_indices")]
        assert np.all(packed.buffers.indptr[device, indptr_length:] == edge_length)
        assert np.all(packed.buffers.indices[device, edge_length:] == 0)
        assert np.all(packed.buffers.variant_indices[device, variant_length:] == 0)
        assert np.all(~packed.buffers.flip[device, variant_length:])
        assert np.all(packed.buffers.logical_variant_indices[device, variant_length:] == -1)


def test_validate_rejects_out_of_range_descriptor_span() -> None:
    packed = pack_blocks((_block(),), num_devices=1)
    descriptors = packed.buffers.block_descriptors.copy()
    descriptors[0, 0, BLOCK_DESCRIPTOR_FIELDS.index("indptr_length")] = packed.buffers.indptr.shape[1] + 1

    with pytest.raises(ValueError, match="descriptor indptr span is out of range"):
        validate_packed_graph(_replace_buffer(packed, "block_descriptors", descriptors))


def test_validate_rejects_nonbijective_logical_variant_mapping() -> None:
    packed = pack_blocks((_block(),), num_devices=1)
    mapping = packed.buffers.logical_variant_indices.copy()
    mapping[0, 1] = mapping[0, 0]

    with pytest.raises(ValueError, match="logical variant mapping must be bijective"):
        validate_packed_graph(_replace_buffer(packed, "logical_variant_indices", mapping))


def test_validate_rejects_overlapping_block_assignments() -> None:
    packed = pack_blocks((_block(), _block(block_seed=1)), num_devices=1)
    descriptors = packed.buffers.block_descriptors.copy()
    descriptors[0, 1, BLOCK_DESCRIPTOR_FIELDS.index("logical_block_index")] = 0

    with pytest.raises(ValueError, match="block assignments must be complete and non-overlapping"):
        validate_packed_graph(_replace_buffer(packed, "block_descriptors", descriptors))


def test_validate_rejects_inconsistent_sample_counts_and_data_dtype() -> None:
    packed = pack_blocks((_block(),), num_devices=1)
    descriptors = packed.buffers.block_descriptors.copy()
    descriptors[0, 0, BLOCK_DESCRIPTOR_FIELDS.index("sample_length")] -= 1
    with pytest.raises(ValueError, match="descriptor sample count"):
        validate_packed_graph(_replace_buffer(packed, "block_descriptors", descriptors))

    with pytest.raises(ValueError, match="data dtype"):
        validate_packed_graph(_replace_buffer(packed, "data", packed.buffers.data.astype(np.float64)))


def test_validate_rejects_non_inert_padding() -> None:
    packed = pack_blocks(
        (_block(n_nodes=10, n_edges=15, n_variants=5), _block(n_nodes=3, n_edges=2, n_variants=1)),
        num_devices=2,
        allow_excess_padding=True,
    )
    data = packed.buffers.data.copy()
    short_device = int(np.argmin(packed.buffers.valid_lengths[:, VALID_LENGTH_FIELDS.index("data")]))
    valid = packed.buffers.valid_lengths[short_device, VALID_LENGTH_FIELDS.index("data")]
    data[short_device, valid] = 7

    with pytest.raises(ValueError, match="non-inert data padding"):
        validate_packed_graph(_replace_buffer(packed, "data", data))


def test_padding_override_never_bypasses_structural_validation() -> None:
    packed = pack_blocks((_block(),), num_devices=2, allow_excess_padding=True)
    mapping = packed.buffers.logical_variant_indices.copy()
    assigned_device = packed.plan.assignment[0]
    mapping[assigned_device, 0] = -1

    assert packed.plan.diagnostics.padding_override
    with pytest.raises(ValueError, match="logical variant mapping"):
        validate_packed_graph(_replace_buffer(packed, "logical_variant_indices", mapping))


def test_packed_ingress_assembles_graph_axis_shards_with_exclusive_local_residency() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    blocks = tuple(_block(block_seed=index) for index in range(4))

    op = _packed_from_block_arrays(blocks, mesh=mesh)

    expected_devices = set(mesh.devices.reshape(-1).tolist())
    for name in PACKED_COMPONENT_NAMES:
        array = getattr(op, name)
        assert isinstance(array.sharding, NamedSharding)
        assert array.sharding.mesh.axis_names == ("graph",)
        assert array.shape[0] == 2
        assert array.devices() == expected_devices
        assert len(array.addressable_shards) == 2
        observed_rows = set()
        for shard in array.addressable_shards:
            assert shard.data.committed
            assert shard.data.devices() == {shard.device}
            assert shard.data.shape[0] == 1
            assert shard.data.on_device_size_in_bytes() == shard.data.nbytes
            row_index = shard.index[0]
            assert isinstance(row_index, slice)
            observed_rows.add((row_index.start, row_index.stop))
        assert observed_rows == {(0, 1), (1, 2)}
        assert array.on_device_size_in_bytes() == array.nbytes


def test_packed_ingress_commits_equal_local_arrays_in_addressable_device_order(monkeypatch) -> None:
    mesh = _two_device_graph_mesh_or_skip()
    original = jax.make_array_from_single_device_arrays
    calls = []

    def record_call(global_shape, sharding, local_arrays):
        calls.append((global_shape, sharding, tuple(local_arrays)))
        return original(global_shape, sharding, local_arrays)

    monkeypatch.setattr(jax, "make_array_from_single_device_arrays", record_call)

    _packed_from_block_arrays(tuple(_block(block_seed=index) for index in range(4)), mesh=mesh)

    assert len(calls) == len(PACKED_COMPONENT_NAMES)
    for global_shape, sharding, local_arrays in calls:
        expected_devices = tuple(sharding.addressable_devices_indices_map(global_shape))
        assert len(local_arrays) == len(expected_devices)
        assert {array.shape for array in local_arrays} == {(1, *global_shape[1:])}
        for local_array, expected_device in zip(local_arrays, expected_devices, strict=True):
            assert local_array.committed
            assert local_array.devices() == {expected_device}


def test_single_device_array_assembly_rejects_malformed_local_layout() -> None:
    device = jax.devices("cpu")[0]
    mesh = Mesh(np.asarray([device]), ("graph",))
    sharding = NamedSharding(mesh, PartitionSpec("graph", None))
    malformed = (jax.device_put(np.zeros((1, 2), dtype=np.float32), device),)

    with pytest.raises(ValueError, match="local shard shape"):
        _assemble_single_device_arrays((1, 3), sharding, malformed)


def test_packed_ingress_empty_assignment_is_inert_on_its_device() -> None:
    mesh = _two_device_graph_mesh_or_skip()

    op = _packed_from_block_arrays((_block(),), mesh=mesh, allow_excess_padding=True)

    empty_device = op.diagnostics.staging_block_owners.index(0) ^ 1
    valid_lengths = np.asarray(op.valid_lengths)
    assert np.all(valid_lengths[empty_device] == 0)
    for name in GRAPH_FIELD_NAMES:
        array = np.asarray(getattr(op, name))
        if name in ("allele_counts", "logical_variant_indices"):
            expected = -1
        elif name == "flip":
            expected = False
        else:
            expected = 0
        assert np.all(array[empty_device] == expected)


def test_packed_ingress_reports_structural_staging_and_final_residency() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    blocks = tuple(_block(block_seed=index) for index in range(4))

    op = _packed_from_block_arrays(blocks, mesh=mesh)
    diagnostics = op.diagnostics

    assert diagnostics.staging_block_owners == (0, 1, 0, 1)
    expected_staging_bytes = max(_canonical_graph_bytes(canonicalize_block_arrays(block)) for block in blocks)
    assert diagnostics.staging_bytes == expected_staging_bytes
    assert sum(diagnostics.final_graph_bytes_by_device) == diagnostics.padded_graph_bytes
    assert sum(diagnostics.final_bytes_by_device) == diagnostics.padded_graph_bytes + diagnostics.descriptor_bytes
    assert diagnostics.component_count == len(PACKED_COMPONENT_NAMES)
    assert diagnostics.pytree_leaf_count == len(PACKED_COMPONENT_NAMES)
    assert diagnostics.staging_accounting == (
        "deterministic one-source-block ingress accounting; not a JAX allocator high-water mark"
    )


def test_packed_benchmark_metrics_and_table_are_calculated_from_diagnostics() -> None:
    diagnostics = SimpleNamespace(
        canonical_graph_bytes=1_000,
        padded_graph_bytes=1_200,
        descriptor_bytes=80,
        padding_ratio=1.2,
        staging_bytes=300,
        final_graph_bytes_by_device=(620, 580),
        component_count=11,
        pytree_leaf_count=11,
    )

    result = _packed_memory_result(
        diagnostics,
        operator="packed_jax_lineararg_2_device",
        construction_seconds=0.25,
        resident_devices_valid=True,
    )
    table = _format_results_table([result])

    assert result.canonical_graph_bytes == 1_000
    assert result.padded_graph_bytes == 1_200
    assert result.descriptor_bytes == 80
    assert result.padding_ratio == 1.2
    assert result.resident_graph_bytes == 1_200
    assert result.max_device_graph_bytes == 620
    assert result.staging_bytes == 300
    assert result.component_count == result.pytree_leaf_count == 11
    assert "canonical graph MiB" in table
    assert "padding ratio" in table
    assert "packed_jax_lineararg_2_device" in table
    assert "1.200" in table


def test_packed_production_gate_reports_padding_residency_and_placement_failures() -> None:
    passing = _packed_memory_result(
        SimpleNamespace(
            canonical_graph_bytes=1_000,
            padded_graph_bytes=1_200,
            descriptor_bytes=80,
            padding_ratio=1.2,
            staging_bytes=300,
            final_graph_bytes_by_device=(620, 580),
            component_count=11,
            pytree_leaf_count=11,
        ),
        operator="packed",
        construction_seconds=0.25,
        resident_devices_valid=True,
    )

    assert _packed_gate_failures(passing) == ()
    assert _packed_gate_failures(replace(passing, padded_graph_bytes=1_251, padding_ratio=1.251)) == (
        "packed padding ratio 1.251000 exceeds 1.250000",
    )
    assert _packed_gate_failures(replace(passing, max_device_graph_bytes=651)) == (
        "maximum device graph residency 651 exceeds 0.65 * canonical graph bytes (650.000)",
    )
    assert _packed_gate_failures(replace(passing, resident_devices_valid=False)) == (
        "one or more packed fields has an unexpected resident device or shard index",
    )
