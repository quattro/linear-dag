# pattern: Functional Core

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from linear_dag.core.jaxlinarg.wrapper import (
    split_blocks_by_n_entries,
    variant_offsets_from_metadata,
)
from linear_dag.core.parallel_processing import _ManagerFactory


def _expected_ranges(metadata: pl.DataFrame, num_devices: int) -> tuple[tuple[int, int], ...]:
    return tuple(_ManagerFactory._split_blocks(metadata, num_devices))


@pytest.mark.parametrize("num_devices", [1, 2, 3, 4])
def test_split_blocks_by_n_entries_matches_parallel_manager_on_fixture(
    linarg_block_metadata: pl.DataFrame,
    num_devices: int,
):
    actual = split_blocks_by_n_entries(linarg_block_metadata, num_devices)

    assert actual == _expected_ranges(linarg_block_metadata, num_devices)


def test_split_blocks_by_n_entries_matches_parallel_manager_for_skewed_blocks():
    metadata = pl.DataFrame(
        {
            "block_name": ["a", "b", "c", "d", "e"],
            "n_entries": [1, 100, 1, 1, 1000],
            "n_variants": [2, 3, 5, 7, 11],
            "n_samples": [4, 4, 4, 4, 4],
        }
    )

    actual = split_blocks_by_n_entries(metadata, num_devices=3)

    assert actual == _expected_ranges(metadata, 3)


def test_split_blocks_by_n_entries_rejects_nonpositive_device_count(
    linarg_block_metadata: pl.DataFrame,
):
    with pytest.raises(ValueError, match="num_devices must be positive"):
        split_blocks_by_n_entries(linarg_block_metadata, 0)


def test_split_blocks_by_n_entries_requires_n_entries_column():
    metadata = pl.DataFrame({"n_variants": [2, 3]})

    with pytest.raises(ValueError, match='metadata must contain columns "n_entries"'):
        split_blocks_by_n_entries(metadata, 1)


def test_variant_offsets_from_metadata_returns_leading_zero_cumulative_offsets():
    metadata = pl.DataFrame({"n_variants": [2, 3, 5]})

    offsets = variant_offsets_from_metadata(metadata)

    np.testing.assert_array_equal(offsets, np.array([0, 2, 5, 10], dtype=np.int64))


def test_variant_offsets_from_metadata_requires_n_variants_column():
    metadata = pl.DataFrame({"n_entries": [2, 3]})

    with pytest.raises(ValueError, match='metadata must contain columns "n_variants"'):
        variant_offsets_from_metadata(metadata)
