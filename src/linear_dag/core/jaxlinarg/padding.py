# pattern: Functional Core

from collections.abc import Iterable
from typing import Any

BucketSpec = tuple[int, int]


def compute_src_of_edge(indptr: Any) -> Any:
    raise NotImplementedError


def pad_to_bucket(indptr: Any, indices: Any, data: Any, *, max_nodes: int, max_nnz: int) -> Any:
    raise NotImplementedError


def choose_bucket(shape: BucketSpec, buckets: Iterable[BucketSpec]) -> BucketSpec:
    raise NotImplementedError


def choose_buckets(shapes: Iterable[BucketSpec], *, max_buckets: int = 8) -> tuple[BucketSpec, ...]:
    raise NotImplementedError
