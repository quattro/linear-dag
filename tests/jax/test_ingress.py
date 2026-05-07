# pattern: Imperative Shell

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.ingress import from_lineararg
from linear_dag.core.jaxlinarg.padding import BucketSpec
from linear_dag.core.lineararg import LinearARG


def test_from_lineararg_matches_in_memory_lineararg_products(oracle_case) -> None:
    op = from_lineararg(oracle_case.linarg, backend=Backend.PURE_JAX)

    assert op.shape == oracle_case.linarg.shape
    np.testing.assert_allclose(np.asarray(op.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(op.rmatmat(oracle_case.y)), oracle_case.XTy, rtol=1e-5, atol=1e-5)


def test_from_lineararg_canonicalizes_arrays_and_preserves_cached_allele_counts(oracle_case) -> None:
    linarg = oracle_case.linarg
    op = from_lineararg(linarg, backend=Backend.PURE_JAX)

    assert op.indptr.dtype == jnp.int32
    assert op.indices.dtype == jnp.int32
    assert op.src_of_edge.dtype == jnp.int32
    assert op.variant_indices.dtype == jnp.int32
    assert op.sample_indices.dtype == jnp.int32
    assert op.nonunique_indices.dtype == jnp.int32
    assert op.data.dtype == jnp.float32
    assert op.flip.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(op.allele_counts), linarg.allele_counts.astype(np.int32))


def test_jax_lineararg_from_hdf5_block_matches_lineararg_read(
    linarg_h5_path,
    first_block_name,
) -> None:
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    op = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.PURE_JAX)
    rng = np.random.default_rng(20260507)
    w = rng.normal(size=(linarg.shape[1], 3)).astype(np.float32)
    y = rng.normal(size=(linarg.shape[0], 3)).astype(np.float32)

    assert op.shape == linarg.shape
    np.testing.assert_allclose(np.asarray(op.matmat(w)), np.asarray(linarg @ w), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(op.rmatmat(y)), np.asarray(linarg.T @ y), rtol=1e-5, atol=1e-5)


def test_from_lineararg_bucket_padding_preserves_matmat_and_rmatmat(oracle_case) -> None:
    linarg = oracle_case.linarg
    unpadded = JaxLinearARG.from_lineararg(linarg, backend=Backend.PURE_JAX)
    padded = JaxLinearARG.from_lineararg(
        linarg,
        backend=Backend.PURE_JAX,
        bucket=BucketSpec(linarg.A.shape[0] + 3, linarg.A.nnz + 7),
    )

    assert padded.indptr.shape == (linarg.A.shape[0] + 4,)
    assert padded.indices.shape == (linarg.A.nnz + 7,)
    assert padded.nonunique_indices.shape == (linarg.A.shape[0] + 3,)
    np.testing.assert_allclose(
        np.asarray(padded.matmat(oracle_case.w)),
        np.asarray(unpadded.matmat(oracle_case.w)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(padded.rmatmat(oracle_case.y)),
        np.asarray(unpadded.rmatmat(oracle_case.y)),
        rtol=1e-5,
        atol=1e-5,
    )
