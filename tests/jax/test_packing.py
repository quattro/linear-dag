from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from linear_dag.core.jaxlinarg.ingress import (
    LinearARGBlockArrays as IngressLinearARGBlockArrays,
    read_hdf5_block_arrays,
)
from linear_dag.core.jaxlinarg.packing import (
    BLOCK_DESCRIPTOR_FIELDS,
    canonicalize_block_arrays,
    GRAPH_FIELD_NAMES,
    LinearARGBlockArrays,
    PACKED_COMPONENT_NAMES,
    plan_packing,
    VALID_LENGTH_FIELDS,
)


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
