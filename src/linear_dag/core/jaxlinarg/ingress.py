# pattern: Imperative Shell

from typing import Any

import jax.numpy as jnp
import numpy as np

from scipy import sparse

from linear_dag.core.lineararg import LinearARG

from .operator import Backend, JaxLinearARG
from .padding import BucketSpec, compute_src_of_edge, pad_to_bucket


def from_lineararg(
    linarg: LinearARG,
    *,
    backend: Backend = Backend.AUTO,
    bucket: BucketSpec | None = None,
    level_schedule: bool = False,
    dtype: Any = None,
) -> JaxLinearARG:
    dtype = jnp.float32 if dtype is None else jnp.dtype(dtype)
    graph = _as_csc(linarg.A)
    n_nodes = graph.shape[0]
    nonunique_indices = _canonical_nonunique_indices(getattr(linarg, "nonunique_indices", None), n_nodes)

    indptr = np.asarray(graph.indptr, dtype=np.int32)
    indices = np.asarray(graph.indices, dtype=np.int32)
    data = np.asarray(graph.data, dtype=np.dtype(dtype))
    src_of_edge = compute_src_of_edge(indptr)
    n_nonunique_indices = None

    if bucket is not None:
        bucket = _as_bucket_spec(bucket)
        padded = pad_to_bucket(
            indptr,
            indices,
            data,
            max_nodes=bucket.max_nodes,
            max_nnz=bucket.max_nnz,
        )
        indptr = padded.indptr
        indices = padded.indices
        data = padded.data
        src_of_edge = padded.src_of_edge
        nonunique_indices = _pad_nonunique_indices(nonunique_indices, bucket.max_nodes)
        n_nonunique_indices = bucket.max_nodes

    return JaxLinearARG.from_lineararg_arrays(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        variant_indices=np.asarray(linarg.variant_indices, dtype=np.int32),
        flip=np.asarray(linarg.flip, dtype=np.bool_),
        sample_indices=np.asarray(linarg.sample_indices, dtype=np.int32),
        nonunique_indices=nonunique_indices,
        allele_counts=_cached_allele_counts(linarg),
        n_variants=int(linarg.shape[1]),
        n_samples=int(linarg.shape[0]),
        n_nonunique_indices=n_nonunique_indices,
        backend=backend,
        level_schedule=level_schedule,
        dtype=dtype,
    )


def from_hdf5_block(
    path: Any,
    block: Any,
    *,
    backend: Backend = Backend.AUTO,
    bucket: BucketSpec | None = None,
    level_schedule: bool = False,
    load_metadata: bool = False,
    dtype: Any = None,
) -> JaxLinearARG:
    linarg = LinearARG.read(path, block=block, load_metadata=load_metadata)
    return from_lineararg(
        linarg,
        backend=backend,
        bucket=bucket,
        level_schedule=level_schedule,
        dtype=dtype,
    )


def _as_csc(matrix: Any) -> sparse.csc_matrix:
    if sparse.isspmatrix_csc(matrix):
        graph = matrix
    elif sparse.issparse(matrix) and hasattr(matrix, "tocsc"):
        graph = matrix.tocsc(copy=False)
    else:
        raise ValueError("linarg.A must be a scipy sparse matrix with CSC-compatible semantics")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("linarg.A must be square")
    return graph


def _canonical_nonunique_indices(nonunique_indices: Any, n_nodes: int) -> np.ndarray:
    if nonunique_indices is None:
        return np.arange(n_nodes, dtype=np.int32)
    return np.asarray(nonunique_indices, dtype=np.int32)


def _pad_nonunique_indices(nonunique_indices: np.ndarray, max_nodes: int) -> np.ndarray:
    if nonunique_indices.shape[0] > max_nodes:
        raise ValueError("bucket max_nodes is smaller than nonunique_indices")
    if nonunique_indices.shape[0] == max_nodes:
        return nonunique_indices
    padded = np.zeros(max_nodes, dtype=np.int32)
    padded[: nonunique_indices.shape[0]] = nonunique_indices
    return padded


def _cached_allele_counts(linarg: LinearARG) -> np.ndarray | None:
    allele_counts = linarg.__dict__.get("allele_counts")
    if allele_counts is None:
        return None
    return np.asarray(allele_counts, dtype=np.int32)


def _as_bucket_spec(bucket: Any) -> BucketSpec:
    if isinstance(bucket, BucketSpec):
        return bucket
    return BucketSpec(*bucket)
