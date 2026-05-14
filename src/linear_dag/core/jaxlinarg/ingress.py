# pattern: Imperative Shell

from typing import Any

import jax.numpy as jnp
import numpy as np

from scipy import sparse

from linear_dag.core.lineararg import LinearARG

from .operator import Backend, JaxLinearARG, resolve_backend


def from_lineararg(
    linarg: LinearARG,
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> JaxLinearARG:
    """Convert a [`linear_dag.core.lineararg.LinearARG`][] to a JAX operator.

    **Arguments:**

    - `linarg`: Source LinearARG object.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    dtype = jnp.float32 if dtype is None else jnp.dtype(dtype)
    graph = _as_csc(linarg.A)
    n_nodes = graph.shape[0]
    nonunique_indices = _canonical_nonunique_indices(getattr(linarg, "nonunique_indices", None), n_nodes)

    indptr = np.asarray(graph.indptr, dtype=np.int32)
    indices = np.asarray(graph.indices, dtype=np.int32)
    data = np.asarray(graph.data, dtype=np.dtype(dtype))
    resolved_backend = resolve_backend(backend)

    return JaxLinearARG.from_lineararg_arrays(
        indptr=indptr,
        indices=indices,
        data=data,
        variant_indices=np.asarray(linarg.variant_indices, dtype=np.int32),
        flip=np.asarray(linarg.flip, dtype=np.bool_),
        sample_indices=np.asarray(linarg.sample_indices, dtype=np.int32),
        nonunique_indices=nonunique_indices,
        allele_counts=_cached_allele_counts(linarg),
        n_variants=int(linarg.shape[1]),
        n_samples=int(linarg.shape[0]),
        backend=resolved_backend,
        dtype=dtype,
    )


def from_hdf5_block(
    path: Any,
    block: Any,
    *,
    backend: Backend = Backend.AUTO,
    load_metadata: bool = False,
    dtype: Any = None,
) -> JaxLinearARG:
    """Read one HDF5 LinearARG block as a JAX operator.

    **Arguments:**

    - `path`: HDF5 file path.
    - `block`: Block name inside the HDF5 file.
    - `backend`: Requested numerical backend.
    - `load_metadata`: Whether to load optional LinearARG metadata.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    linarg = LinearARG.read(path, block=block, load_metadata=load_metadata)
    return from_lineararg(
        linarg,
        backend=backend,
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


def _cached_allele_counts(linarg: LinearARG) -> np.ndarray | None:
    allele_counts = linarg.__dict__.get("allele_counts")
    if allele_counts is None:
        return None
    return np.asarray(allele_counts, dtype=np.int32)
