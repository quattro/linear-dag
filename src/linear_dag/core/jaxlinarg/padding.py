# pattern: Functional Core

from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np


class BucketSpec(NamedTuple):
    """Static graph dimensions used to pad a LinearARG block for JAX tracing."""

    max_nodes: int
    max_nnz: int


class PaddedGraph(NamedTuple):
    """Padded CSC graph arrays."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray


def pad_to_bucket(indptr: Any, indices: Any, data: Any, *, max_nodes: int, max_nnz: int) -> Any:
    """Pad CSC graph arrays to a static node and edge bucket.

    **Arguments:**

    - `indptr`: CSC index pointer array of length `n_nodes + 1`.
    - `indices`: CSC destination node indices.
    - `data`: CSC edge weights aligned to `indices`.
    - `max_nodes`: Static node count for the padded graph.
    - `max_nnz`: Static edge count for the padded graph.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.padding.PaddedGraph`][].

    **Raises:**

    - `ValueError`: If the input arrays are not valid CSC arrays or the bucket
      is too small.
    """
    indptr = np.asarray(indptr)
    indices = np.asarray(indices)
    data = np.asarray(data)
    n_nodes = indptr.shape[0] - 1
    nnz = indices.shape[0]

    if data.shape[0] != nnz:
        raise ValueError("data must have the same length as indices")
    if indptr.shape[0] == 0:
        raise ValueError("indptr must contain at least one entry")
    if int(indptr[0]) != 0:
        raise ValueError("indptr must start at 0")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be monotonic")
    if int(indptr[-1]) != nnz:
        raise ValueError("final indptr entry must match the edge count")
    if max_nodes < n_nodes:
        raise ValueError(f"max_nodes={max_nodes} cannot be smaller than node count {n_nodes}")
    if max_nnz < nnz:
        raise ValueError(f"max_nnz={max_nnz} cannot be smaller than edge count {nnz}")
    if max_nnz > nnz and max_nodes < 1:
        raise ValueError("max_nodes must be positive when padding edges")

    padded_indptr = np.empty(max_nodes + 1, dtype=np.int32)
    if max_nodes == n_nodes:
        padded_indptr[:-1] = indptr[:-1].astype(np.int32, copy=False)
        padded_indptr[-1] = max_nnz
    else:
        padded_indptr[: n_nodes + 1] = indptr.astype(np.int32, copy=False)
        padded_indptr[n_nodes + 1 : max_nodes] = nnz
        padded_indptr[max_nodes] = max_nnz

    self_loop_node = max_nodes - 1
    padded_indices = np.full(max_nnz, self_loop_node, dtype=np.int32)
    padded_indices[:nnz] = indices.astype(np.int32, copy=False)

    padded_data = np.zeros(max_nnz, dtype=data.dtype)
    padded_data[:nnz] = data

    return PaddedGraph(
        indptr=padded_indptr,
        indices=padded_indices,
        data=padded_data,
    )


def choose_bucket(shape: BucketSpec, buckets: Iterable[BucketSpec]) -> BucketSpec:
    """Choose the first bucket that can contain `shape`.

    **Arguments:**

    - `shape`: Required node and edge dimensions.
    - `buckets`: Candidate buckets in preference order.

    **Returns:**

    - The first bucket whose dimensions are both at least those of `shape`.

    **Raises:**

    - `ValueError`: If no candidate bucket can contain `shape`.
    """
    shape = _as_bucket_spec(shape)
    for bucket in buckets:
        bucket = _as_bucket_spec(bucket)
        if bucket.max_nodes >= shape.max_nodes and bucket.max_nnz >= shape.max_nnz:
            return bucket
    raise ValueError(f"No bucket can contain shape {shape}")


def choose_buckets(shapes: Iterable[BucketSpec], *, max_buckets: int = 8) -> tuple[BucketSpec, ...]:
    """Choose a small set of static buckets for a collection of graph shapes.

    **Arguments:**

    - `shapes`: Required node and edge dimensions.
    - `max_buckets`: Maximum number of buckets to return.

    **Returns:**

    - Bucket specs covering every input shape.

    **Raises:**

    - `ValueError`: If `max_buckets` is less than one.
    """
    if max_buckets < 1:
        raise ValueError("max_buckets must be at least 1")

    sorted_shapes = sorted({_as_bucket_spec(shape) for shape in shapes}, key=_bucket_sort_key)
    if len(sorted_shapes) <= max_buckets:
        return tuple(sorted_shapes)

    group_count = max_buckets
    quotient, remainder = divmod(len(sorted_shapes), group_count)
    buckets = []
    start = 0
    for group_index in range(group_count):
        stop = start + quotient + (1 if group_index < remainder else 0)
        group = sorted_shapes[start:stop]
        buckets.append(
            BucketSpec(
                max_nodes=max(shape.max_nodes for shape in group),
                max_nnz=max(shape.max_nnz for shape in group),
            )
        )
        start = stop
    return tuple(buckets)


def _as_bucket_spec(shape: Any) -> BucketSpec:
    if isinstance(shape, BucketSpec):
        return shape
    max_nodes, max_nnz = shape
    return BucketSpec(int(max_nodes), int(max_nnz))


def _bucket_sort_key(shape: BucketSpec) -> tuple[int, int, int]:
    return (shape.max_nodes * shape.max_nnz, shape.max_nodes, shape.max_nnz)
