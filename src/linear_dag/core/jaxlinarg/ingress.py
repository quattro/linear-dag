# pattern: Imperative Shell

"""Host-to-device ingress for LinearARG arrays, HDF5 blocks, and Zarr groups."""

from __future__ import annotations

import warnings

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from os import PathLike
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np

from scipy import sparse

from linear_dag.core.lineararg import LinearARG

from .operator import Backend, JaxLinearARG, resolve_backend


@dataclass(frozen=True)
class LinearARGBlockArrays:
    """Canonical host arrays for one LinearARG block.

    This immutable transfer object separates storage I/O from JAX device
    construction. Array dtypes are normalized by the reader functions before
    an instance is returned.

    !!! Example

        ```python
        from linear_dag.core.jaxlinarg.ingress import (
            from_block_arrays,
            read_hdf5_block_arrays,
        )

        arrays = read_hdf5_block_arrays("lineararg.h5", "block_0")
        operator = from_block_arrays(arrays)
        ```
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    variant_indices: np.ndarray
    flip: np.ndarray
    sample_indices: np.ndarray
    nonunique_indices: np.ndarray | None
    allele_counts: np.ndarray | None
    n_variants: int
    n_samples: int


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
    dtype = _normalize_dtype(dtype)
    graph = _as_csc(linarg.A)
    n_nodes = graph.shape[0]
    nonunique_indices = _canonical_nonunique_indices(getattr(linarg, "nonunique_indices", None), n_nodes)

    indptr = np.asarray(graph.indptr, dtype=np.int32)
    indices = np.asarray(graph.indices, dtype=np.int32)
    data = np.asarray(graph.data, dtype=np.dtype(dtype))
    resolved_backend = resolve_backend(backend)

    return JaxLinearARG.from_lineararg_arrays(
        **_block_arrays_kwargs(
            LinearARGBlockArrays(
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
            )
        ),
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
    - `load_metadata`: Accepted for compatibility. JAX operators do not retain
      variant metadata, so this does not materialize metadata tables.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    del load_metadata
    return from_block_arrays(read_hdf5_block_arrays(path, block, dtype=dtype), backend=backend, dtype=dtype)


def from_zarr_group(
    group: Any,
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> JaxLinearARG:
    """Read one Zarr LinearARG block group as a JAX operator.

    **Arguments:**

    - `group`: Zarr group containing one LinearARG block.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    return from_block_arrays(read_zarr_block_arrays(group, dtype=dtype), backend=backend, dtype=dtype)


def from_block_arrays(
    arrays: LinearARGBlockArrays,
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> JaxLinearARG:
    """Construct a JAX LinearARG operator from canonical host block arrays.

    **Arguments:**

    - `arrays`: Canonical LinearARG block arrays.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - A [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    dtype = _normalize_dtype(dtype)
    return JaxLinearARG.from_lineararg_arrays(
        **_block_arrays_kwargs(arrays),
        backend=backend,
        dtype=dtype,
    )


def read_hdf5_block_arrays(path: str | PathLike[str], block: Any, *, dtype: Any = None) -> LinearARGBlockArrays:
    """Read one HDF5 LinearARG block into canonical JAX-ingress host arrays.

    **Arguments:**

    - `path`: HDF5 file path.
    - `block`: Block name inside the HDF5 file.
    - `dtype`: Optional data dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Canonical host arrays for [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    _ensure_hdf5_plugins()
    with h5py.File(_hdf5_path(path), "r") as file:
        return _read_block_arrays_from_group(file[block], dtype=dtype)


def read_hdf5_blocks(
    path: str | PathLike[str],
    block_names: Iterable[Any],
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> tuple[JaxLinearARG, ...]:
    """Read HDF5 LinearARG blocks directly as eager JAX operators.

    !!! info

        This convenience function materializes every requested block eagerly on
        the default device. Use
        [`linear_dag.core.jaxlinarg.JaxParallelOperator.from_hdf5`][] for
        device-aware multi-block ingress without transient graph duplication.

    **Arguments:**

    - `path`: HDF5 file path.
    - `block_names`: Block names inside the HDF5 file.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Tuple of [`linear_dag.core.jaxlinarg.JaxLinearARG`][] blocks.
    """
    _ensure_hdf5_plugins()
    with h5py.File(_hdf5_path(path), "r") as file:
        return tuple(
            from_block_arrays(
                _read_block_arrays_from_group(file[block_name], dtype=dtype),
                backend=backend,
                dtype=dtype,
            )
            for block_name in block_names
        )


def read_zarr_block_arrays(group: Any, *, dtype: Any = None) -> LinearARGBlockArrays:
    """Read one Zarr LinearARG block group into canonical JAX-ingress host arrays.

    **Arguments:**

    - `group`: Zarr group containing one LinearARG block.
    - `dtype`: Optional data dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Canonical host arrays for [`linear_dag.core.jaxlinarg.JaxLinearARG`][].
    """
    return _read_block_arrays_from_group(group, dtype=dtype)


def read_zarr_blocks(
    reader: Any,
    block_names: Iterable[str],
    *,
    backend: Backend = Backend.AUTO,
    dtype: Any = None,
) -> tuple[JaxLinearARG, ...]:
    """Read Zarr LinearARG blocks directly as eager JAX operators.

    !!! info

        This convenience function materializes every requested block eagerly on
        the default device. Use
        [`linear_dag.core.jaxlinarg.JaxParallelOperator.from_zarr`][] for
        device-aware multi-block ingress without transient graph duplication.

    **Arguments:**

    - `reader`: Open Zarr reader with a `root["blocks"]` group.
    - `block_names`: Block names inside `blocks/`.
    - `backend`: Requested numerical backend.
    - `dtype`: Optional computation dtype. Defaults to `jax.numpy.float32`.

    **Returns:**

    - Tuple of [`linear_dag.core.jaxlinarg.JaxLinearARG`][] blocks.
    """
    blocks_group = reader.root["blocks"]
    return tuple(from_zarr_group(blocks_group[block_name], backend=backend, dtype=dtype) for block_name in block_names)


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


def _read_block_arrays_from_group(group: Any, *, dtype: Any = None) -> LinearARGBlockArrays:
    dtype = _normalize_dtype(dtype)
    n_nodes = int(group.attrs["n"])
    n_samples = int(group.attrs["n_samples"])
    n_individuals = group.attrs.get("n_individuals", None)
    return LinearARGBlockArrays(
        indptr=np.asarray(group["indptr"][:], dtype=np.int32),
        indices=np.asarray(group["indices"][:], dtype=np.int32),
        data=np.asarray(group["data"][:], dtype=np.dtype(dtype)),
        variant_indices=np.asarray(group["variant_indices"][:], dtype=np.int32),
        flip=np.asarray(group["flip"][:], dtype=np.bool_),
        sample_indices=_sample_indices(n_nodes, n_samples, n_individuals),
        nonunique_indices=_optional_array(group, "nonunique_indices", dtype=np.int32),
        allele_counts=_optional_array(group, "allele_counts", dtype=np.int32),
        n_variants=int(group.attrs["n_variants"]),
        n_samples=n_samples,
    )


def _block_arrays_kwargs(arrays: LinearARGBlockArrays) -> dict[str, Any]:
    return {
        "indptr": arrays.indptr,
        "indices": arrays.indices,
        "data": arrays.data,
        "variant_indices": arrays.variant_indices,
        "flip": arrays.flip,
        "sample_indices": arrays.sample_indices,
        "nonunique_indices": arrays.nonunique_indices,
        "allele_counts": arrays.allele_counts,
        "n_variants": arrays.n_variants,
        "n_samples": arrays.n_samples,
    }


def _optional_array(group: Any, name: str, *, dtype: Any) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][:], dtype=dtype)


def _normalize_dtype(dtype: Any) -> jnp.dtype:
    return jnp.float32 if dtype is None else jnp.dtype(dtype)


def _hdf5_path(path: str | PathLike[str]) -> str | PathLike[str]:
    if str(path).endswith(".h5"):
        return path
    return f"{path}.h5"


def _sample_indices(n_nodes: int, n_samples: int, n_individuals: Any) -> np.ndarray:
    if n_individuals is None:
        stop = n_nodes - n_samples - 1
        return np.arange(n_nodes - 1, stop, -1, dtype=np.int32)
    start = n_nodes - int(n_individuals) - 1
    stop = n_nodes - int(n_individuals) - n_samples - 1
    return np.arange(start, stop, -1, dtype=np.int32)


def _ensure_hdf5_plugins() -> None:
    if find_spec("hdf5plugin") is None:
        warnings.warn("hdf5plugin is required for blosc compression; this may impact reading", stacklevel=2)
        return
    # Importing the optional package registers its HDF5 filters process-wide.
    import_module("hdf5plugin")
