# pattern: Functional Core

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from linear_dag.core.jaxlinarg.kernels.pure_jax import (
    pure_jax_solve_backward,
    pure_jax_solve_forward,
)


def _src_of_edge(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(indptr.shape[0] - 1, dtype=np.int32), np.diff(indptr))


def _as_matrix(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


def test_pure_jax_forward_kernel_matches_oracle_case(oracle_case):
    linarg = oracle_case.linarg
    w = _as_matrix(oracle_case.w)
    sign = np.where(linarg.flip, -1, 1).astype(w.dtype)
    b = np.zeros((linarg.A.shape[0], w.shape[1]), dtype=w.dtype)
    np.add.at(b, linarg.variant_indices, w * sign[:, None])

    solved = pure_jax_solve_forward(
        jnp.asarray(linarg.A.indptr, dtype=jnp.int32),
        jnp.asarray(linarg.A.indices, dtype=jnp.int32),
        jnp.asarray(linarg.A.data, dtype=w.dtype),
        jnp.asarray(_src_of_edge(linarg.A.indptr), dtype=jnp.int32),
        jnp.asarray(b),
    )
    actual = np.asarray(solved)[linarg.sample_indices] + np.sum(w[linarg.flip], axis=0)

    np.testing.assert_allclose(actual, _as_matrix(oracle_case.Xw), rtol=1e-5, atol=1e-5)


def test_pure_jax_backward_kernel_matches_oracle_case(oracle_case):
    linarg = oracle_case.linarg
    y = _as_matrix(oracle_case.y)
    b = np.zeros((linarg.A.shape[0], y.shape[1]), dtype=y.dtype)
    b[linarg.sample_indices] = y

    solved = pure_jax_solve_backward(
        jnp.asarray(linarg.A.indptr, dtype=jnp.int32),
        jnp.asarray(linarg.A.indices, dtype=jnp.int32),
        jnp.asarray(linarg.A.data, dtype=y.dtype),
        jnp.asarray(_src_of_edge(linarg.A.indptr), dtype=jnp.int32),
        jnp.asarray(b),
    )
    actual = np.asarray(solved)[linarg.variant_indices]
    if np.any(linarg.flip):
        actual[linarg.flip] = np.sum(y, axis=0) - actual[linarg.flip]

    np.testing.assert_allclose(actual, _as_matrix(oracle_case.XTy), rtol=1e-5, atol=1e-5)
