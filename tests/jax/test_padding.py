# pattern: Functional Core

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg.kernels.pure_jax import pure_jax_solve_forward
from linear_dag.core.jaxlinarg.padding import (
    align_bucket_for_mosaic_gpu,
    aligned_length_for_mosaic_gpu_transfer,
    BucketSpec,
    choose_bucket,
    choose_buckets,
    compute_src_of_edge,
    pad_to_bucket,
)
from tests.helpers.linarg_fixtures import load_lineararg_block


def test_compute_src_of_edge_expands_csc_sources() -> None:
    indptr = np.array([0, 2, 3, 3], dtype=np.int64)

    src_of_edge = compute_src_of_edge(indptr)

    np.testing.assert_array_equal(src_of_edge, np.array([0, 0, 1], dtype=np.int32))
    assert src_of_edge.dtype == np.int32


def test_pad_to_bucket_preserves_forward_solve_for_small_graph() -> None:
    indptr = np.array([0, 2, 3, 3], dtype=np.int32)
    indices = np.array([1, 2, 2], dtype=np.int32)
    data = np.array([0.5, 2.0, -1.0], dtype=np.float32)
    b = jnp.asarray([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)

    padded = pad_to_bucket(indptr, indices, data, max_nodes=5, max_nnz=7)
    padded_b = jnp.pad(b, ((0, 2), (0, 0)))
    expected = pure_jax_solve_forward(
        jnp.asarray(indptr),
        jnp.asarray(indices),
        jnp.asarray(data),
        jnp.asarray(compute_src_of_edge(indptr)),
        b,
    )
    actual = pure_jax_solve_forward(
        jnp.asarray(padded.indptr),
        jnp.asarray(padded.indices),
        jnp.asarray(padded.data),
        jnp.asarray(padded.src_of_edge),
        padded_b,
    )

    np.testing.assert_array_equal(padded.indptr[: indptr.shape[0]], indptr)
    np.testing.assert_array_equal(padded.indices[: indices.shape[0]], indices)
    np.testing.assert_array_equal(padded.data[: data.shape[0]], data)
    np.testing.assert_array_equal(padded.src_of_edge, compute_src_of_edge(padded.indptr))
    np.testing.assert_array_equal(padded.data[data.shape[0] :], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(padded.indices[data.shape[0] :], np.full(4, 4, dtype=np.int32))
    np.testing.assert_allclose(np.asarray(actual[: b.shape[0], :]), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_pad_to_bucket_preserves_forward_solve_for_real_lineararg_block(
    linarg_h5_path,
    first_block_name,
) -> None:
    linarg = load_lineararg_block(linarg_h5_path, block_name=first_block_name)
    rng = np.random.default_rng(20260507)
    b = rng.normal(size=(linarg.A.shape[0], 2)).astype(np.float32)

    padded = pad_to_bucket(
        linarg.A.indptr,
        linarg.A.indices,
        linarg.A.data.astype(np.float32),
        max_nodes=linarg.A.shape[0] + 2,
        max_nnz=linarg.A.nnz + 5,
    )
    expected = pure_jax_solve_forward(
        jnp.asarray(linarg.A.indptr),
        jnp.asarray(linarg.A.indices),
        jnp.asarray(linarg.A.data.astype(np.float32)),
        jnp.asarray(compute_src_of_edge(linarg.A.indptr)),
        jnp.asarray(b),
    )
    actual = pure_jax_solve_forward(
        jnp.asarray(padded.indptr),
        jnp.asarray(padded.indices),
        jnp.asarray(padded.data),
        jnp.asarray(padded.src_of_edge),
        jnp.pad(jnp.asarray(b), ((0, 2), (0, 0))),
    )

    np.testing.assert_allclose(np.asarray(actual[: linarg.A.shape[0], :]), np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "max_nodes,max_nnz,match",
    [
        (2, 3, "max_nodes"),
        (3, 2, "max_nnz"),
    ],
)
def test_pad_to_bucket_rejects_shrinking_bucket(max_nodes: int, max_nnz: int, match: str) -> None:
    indptr = np.array([0, 2, 3, 3], dtype=np.int32)
    indices = np.array([1, 2, 2], dtype=np.int32)
    data = np.array([0.5, 2.0, -1.0], dtype=np.float32)

    with pytest.raises(ValueError, match=match):
        pad_to_bucket(indptr, indices, data, max_nodes=max_nodes, max_nnz=max_nnz)


def test_choose_buckets_returns_exact_sorted_unique_shapes_when_under_cap() -> None:
    shapes = [(6, 7), (2, 10), (3, 3), (6, 7)]

    buckets = choose_buckets(shapes, max_buckets=8)

    assert buckets == (BucketSpec(3, 3), BucketSpec(2, 10), BucketSpec(6, 7))


def test_choose_buckets_merges_heterogeneous_shapes_to_cap() -> None:
    shapes = [(node_count, node_count * 3 + 1) for node_count in range(2, 14)]

    buckets = choose_buckets(shapes, max_buckets=8)

    assert len(buckets) == 8
    for shape in shapes:
        bucket = choose_bucket(shape, buckets)
        assert bucket.max_nodes >= shape[0]
        assert bucket.max_nnz >= shape[1]
    assert buckets == choose_buckets(reversed(shapes), max_buckets=8)


def test_choose_bucket_respects_first_explicit_bucket_that_fits() -> None:
    buckets = (BucketSpec(5, 10), BucketSpec(8, 8), BucketSpec(12, 20))

    assert choose_bucket((4, 9), buckets) == BucketSpec(5, 10)
    assert choose_bucket((6, 8), buckets) == BucketSpec(8, 8)
    assert choose_bucket((10, 12), buckets) == BucketSpec(12, 20)


def test_choose_bucket_rejects_oversized_explicit_shape() -> None:
    buckets = (BucketSpec(5, 10), BucketSpec(8, 8))

    with pytest.raises(ValueError, match="No bucket"):
        choose_bucket((9, 9), buckets)


def test_align_bucket_for_mosaic_gpu_separates_indptr_and_node_ref_padding() -> None:
    padding = align_bucket_for_mosaic_gpu(
        BucketSpec(max_nodes=834, max_nnz=2008),
        nonunique_count=834,
        data_dtype=np.float32,
    )

    assert padding.bucket.max_nodes == 863
    assert padding.bucket.max_nnz == 2016
    assert padding.nonunique_indices_length == 864
    assert padding.state_rows == 864
    assert (padding.bucket.max_nodes + 1) * np.dtype(np.int32).itemsize % 128 == 0
    assert padding.bucket.max_nnz * np.dtype(np.int32).itemsize % 128 == 0
    assert padding.bucket.max_nnz * np.dtype(np.float32).itemsize % 128 == 0
    assert padding.nonunique_indices_length * np.dtype(np.int32).itemsize % 128 == 0
    assert padding.state_rows * np.dtype(np.float32).itemsize % 128 == 0


def test_aligned_length_for_mosaic_gpu_transfer_respects_dtype_width() -> None:
    assert aligned_length_for_mosaic_gpu_transfer(np.int32, 835) == 864
    assert aligned_length_for_mosaic_gpu_transfer(np.float64, 2008) == 2016
