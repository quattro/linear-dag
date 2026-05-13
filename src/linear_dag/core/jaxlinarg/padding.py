# pattern: Functional Core

import math

from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np

MOSAIC_GPU_TRANSFER_BYTES = 128


class BucketSpec(NamedTuple):
    """Static graph dimensions used to pad a LinearARG block for JAX tracing."""

    max_nodes: int
    max_nnz: int


class PaddedGraph(NamedTuple):
    """Padded CSC graph arrays and edge-source metadata."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    src_of_edge: np.ndarray


class MosaicGpuPaddingSpec(NamedTuple):
    """Static storage dimensions needed by the current Mosaic GPU lowering."""

    bucket: BucketSpec
    nonunique_indices_length: int
    state_rows: int


def compute_src_of_edge(indptr: Any) -> Any:
    """Return the CSC source node for each edge implied by `indptr`."""
    indptr = np.asarray(indptr)
    n_nodes = indptr.shape[0] - 1
    return np.repeat(np.arange(n_nodes, dtype=np.int32), np.diff(indptr).astype(np.int64))


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
        src_of_edge=compute_src_of_edge(padded_indptr),
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


def align_bucket_for_mosaic_gpu(
    bucket: Any,
    *,
    nonunique_count: int | None = None,
    index_dtype: Any = np.int32,
    data_dtype: Any = np.float32,
) -> MosaicGpuPaddingSpec:
    """Return storage padding that satisfies current Mosaic GPU transfers.

    Mosaic GPU lowering currently requires whole-ref transfers to have byte
    sizes divisible by 128. That is a backend lowering constraint, not a
    mathematical requirement of the LinearARG solve. The graph bucket aligns
    `indptr` by padding `max_nodes + 1`, and edge arrays by padding `max_nnz`.
    The nonunique-index buffer and solve-state rows are padded separately
    because their natural length is `max_nodes`, not `max_nodes + 1`.
    """
    bucket = _as_bucket_spec(bucket)
    index_dtype = np.dtype(index_dtype)
    data_dtype = np.dtype(data_dtype)
    index_multiple = _elements_per_mosaic_transfer(index_dtype)
    data_multiple = _elements_per_mosaic_transfer(data_dtype)
    edge_multiple = math.lcm(index_multiple, data_multiple)

    aligned_bucket = BucketSpec(
        max_nodes=aligned_length_for_mosaic_gpu_transfer(index_dtype, bucket.max_nodes + 1) - 1,
        max_nnz=_round_up(bucket.max_nnz, edge_multiple),
    )
    nonunique_count = bucket.max_nodes if nonunique_count is None else int(nonunique_count)
    nonunique_indices_length = aligned_length_for_mosaic_gpu_transfer(index_dtype, aligned_bucket.max_nodes)
    state_rows = aligned_length_for_mosaic_gpu_transfer(data_dtype, max(nonunique_count, 1))
    return MosaicGpuPaddingSpec(
        bucket=aligned_bucket,
        nonunique_indices_length=nonunique_indices_length,
        state_rows=state_rows,
    )


def aligned_length_for_mosaic_gpu_transfer(dtype: Any, length: int) -> int:
    """Round an array length up to a 128-byte Mosaic GPU transfer boundary."""
    return _round_up(int(length), _elements_per_mosaic_transfer(np.dtype(dtype)))


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


def _elements_per_mosaic_transfer(dtype: np.dtype) -> int:
    itemsize = np.dtype(dtype).itemsize
    if MOSAIC_GPU_TRANSFER_BYTES % itemsize != 0:
        raise ValueError(f"dtype itemsize {itemsize} does not divide {MOSAIC_GPU_TRANSFER_BYTES} bytes")
    return MOSAIC_GPU_TRANSFER_BYTES // itemsize


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _bucket_sort_key(shape: BucketSpec) -> tuple[int, int, int]:
    return (shape.max_nodes * shape.max_nnz, shape.max_nodes, shape.max_nnz)
