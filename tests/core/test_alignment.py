# pattern: Functional Core

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from linear_dag.core.alignment import get_iid_alignment
from linear_dag.core.operators import get_inner_merge_operators


def test_iid_alignment_matches_sparse_merge_operator_products() -> None:
    row_ids = pl.Series("iid", ["b", "a", "c", "a", "d"])
    col_ids = pl.Series("iid", ["a", "c", "a", "e"])
    left_op, right_op = get_inner_merge_operators(row_ids, col_ids)
    alignment = get_iid_alignment(row_ids, col_ids)

    left_values = np.arange(row_ids.len() * 2, dtype=np.float32).reshape(row_ids.len(), 2)
    merged_values = np.arange(alignment.n_merged * 2, dtype=np.float32).reshape(alignment.n_merged, 2)
    right_values = np.arange(col_ids.len() * 2, dtype=np.float32).reshape(col_ids.len(), 2)

    np.testing.assert_array_equal(alignment.gather_left(left_values), left_op.T @ left_values)
    np.testing.assert_array_equal(alignment.scatter_left(merged_values), left_op @ merged_values)
    np.testing.assert_array_equal(alignment.gather_right(right_values), right_op @ right_values)
    np.testing.assert_array_equal(alignment.scatter_right(merged_values), right_op.T @ merged_values)


def test_iid_alignment_preserves_sparse_duplicate_addition_semantics() -> None:
    row_ids = pl.Series("iid", ["a", "a"])
    col_ids = pl.Series("iid", ["a"])
    alignment = get_iid_alignment(row_ids, col_ids)

    merged_values = np.array([[1.0], [2.0]], dtype=np.float32)

    np.testing.assert_array_equal(alignment.scatter_right(merged_values), np.array([[3.0]], dtype=np.float32))


def test_iid_alignment_supports_rank_one_inputs() -> None:
    row_ids = pl.Series("iid", ["b", "a", "c"])
    col_ids = pl.Series("iid", ["a", "c"])
    alignment = get_iid_alignment(row_ids, col_ids)
    left_values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    merged_values = np.array([1.0, 2.0], dtype=np.float32)

    np.testing.assert_array_equal(alignment.gather_left(left_values), np.array([20.0, 30.0], dtype=np.float32))
    np.testing.assert_array_equal(alignment.scatter_left(merged_values), np.array([0.0, 1.0, 2.0], dtype=np.float32))


def test_iid_alignment_jax_operations_match_numpy() -> None:
    row_ids = pl.Series("iid", ["b", "a", "c", "a"])
    col_ids = pl.Series("iid", ["a", "c", "a"])
    alignment = get_iid_alignment(row_ids, col_ids)

    left_values = np.arange(row_ids.len() * 2, dtype=np.float32).reshape(row_ids.len(), 2)
    merged_values = np.arange(alignment.n_merged * 2, dtype=np.float32).reshape(alignment.n_merged, 2)

    np.testing.assert_array_equal(
        np.asarray(alignment.gather_left_jax(jnp.asarray(left_values))),
        alignment.gather_left(left_values),
    )
    np.testing.assert_array_equal(
        np.asarray(alignment.scatter_right_jax(jnp.asarray(merged_values))),
        alignment.scatter_right(merged_values),
    )


@pytest.mark.parametrize(
    ("method_name", "values", "message"),
    (
        ("gather_left", np.zeros((4, 1), dtype=np.float32), "expected leading dimension 3"),
        ("gather_right", np.zeros((3, 1), dtype=np.float32), "expected leading dimension 2"),
        ("scatter_left", np.zeros((3, 1), dtype=np.float32), "expected leading dimension 2"),
        ("scatter_right", np.zeros((3, 1), dtype=np.float32), "expected leading dimension 2"),
    ),
)
def test_iid_alignment_rejects_wrong_leading_dimension(method_name: str, values: np.ndarray, message: str) -> None:
    alignment = get_iid_alignment(pl.Series("iid", ["b", "a", "c"]), pl.Series("iid", ["a", "c"]))

    with pytest.raises(ValueError, match=message):
        getattr(alignment, method_name)(values)


@pytest.mark.parametrize("method_name", ("gather_left", "gather_right", "scatter_left", "scatter_right"))
def test_iid_alignment_rejects_rank_three_inputs(method_name: str) -> None:
    alignment = get_iid_alignment(pl.Series("iid", ["a"]), pl.Series("iid", ["a"]))

    with pytest.raises(ValueError, match="expected rank 1 or 2"):
        getattr(alignment, method_name)(np.zeros((1, 1, 1), dtype=np.float32))


def test_iid_alignment_rejects_dtype_mismatch() -> None:
    row_ids = pl.Series("iid", ["a"])
    col_ids = pl.Series("iid", [1])

    with pytest.raises(TypeError, match="Data types"):
        get_iid_alignment(row_ids, col_ids)
