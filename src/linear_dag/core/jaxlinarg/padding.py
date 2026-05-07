# pattern: Functional Core

from collections.abc import Iterable
from typing import Any, NamedTuple

import numpy as np


class BucketSpec(NamedTuple):
    max_nodes: int
    max_nnz: int


class PaddedGraph(NamedTuple):
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    src_of_edge: np.ndarray


def compute_src_of_edge(indptr: Any) -> Any:
    indptr = np.asarray(indptr)
    n_nodes = indptr.shape[0] - 1
    return np.repeat(np.arange(n_nodes, dtype=np.int32), np.diff(indptr).astype(np.int64))


def pad_to_bucket(indptr: Any, indices: Any, data: Any, *, max_nodes: int, max_nnz: int) -> Any:
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
    raise NotImplementedError


def choose_buckets(shapes: Iterable[BucketSpec], *, max_buckets: int = 8) -> tuple[BucketSpec, ...]:
    raise NotImplementedError
