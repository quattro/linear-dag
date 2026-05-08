# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg.kernels import pallas_gpu


def _as_matrix(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _require_pallas_gpu() -> None:
    if not pallas_gpu.is_pallas_gpu_available():
        pytest.skip(
            "Pallas GPU kernels require jax.default_backend() == 'gpu' " "and importable jax.experimental.pallas"
        )


def test_pallas_gpu_availability_is_false_without_gpu_backend() -> None:
    if jax.default_backend() == "gpu":
        pytest.skip("default JAX backend is GPU; availability is covered by integration tests")

    assert not pallas_gpu.is_pallas_gpu_available()


def test_pallas_gpu_forward_kernel_matches_oracle_case(oracle_case) -> None:
    _require_pallas_gpu()
    linarg = oracle_case.linarg
    w = _as_matrix(oracle_case.w)
    sign = np.where(linarg.flip, -1, 1).astype(w.dtype)
    b = np.zeros((linarg.A.shape[0], w.shape[1]), dtype=w.dtype)
    np.add.at(b, linarg.variant_indices, w * sign[:, None])

    solved = pallas_gpu.pallas_gpu_solve_forward(
        jnp.asarray(linarg.A.indptr, dtype=jnp.int32),
        jnp.asarray(linarg.A.indices, dtype=jnp.int32),
        jnp.asarray(linarg.A.data, dtype=w.dtype),
        jnp.asarray(np.repeat(np.arange(linarg.A.shape[0], dtype=np.int32), np.diff(linarg.A.indptr))),
        jnp.arange(linarg.A.shape[0], dtype=jnp.int32),
        0,
        jnp.asarray(b),
    )
    actual = np.asarray(solved)[linarg.sample_indices] + np.sum(w[linarg.flip], axis=0)

    np.testing.assert_allclose(actual, _as_matrix(oracle_case.Xw), rtol=1e-5, atol=1e-5)


def test_pallas_gpu_backward_kernel_matches_oracle_case(oracle_case) -> None:
    _require_pallas_gpu()
    linarg = oracle_case.linarg
    y = _as_matrix(oracle_case.y)
    b = np.zeros((linarg.A.shape[0], y.shape[1]), dtype=y.dtype)
    b[linarg.sample_indices] = y

    solved = pallas_gpu.pallas_gpu_solve_backward(
        jnp.asarray(linarg.A.indptr, dtype=jnp.int32),
        jnp.asarray(linarg.A.indices, dtype=jnp.int32),
        jnp.asarray(linarg.A.data, dtype=y.dtype),
        jnp.asarray(np.repeat(np.arange(linarg.A.shape[0], dtype=np.int32), np.diff(linarg.A.indptr))),
        jnp.arange(linarg.A.shape[0], dtype=jnp.int32),
        0,
        jnp.asarray(b),
    )
    actual = np.asarray(solved)[linarg.variant_indices]
    if np.any(linarg.flip):
        actual[linarg.flip] = np.sum(y, axis=0) - actual[linarg.flip]

    np.testing.assert_allclose(actual, _as_matrix(oracle_case.XTy), rtol=1e-5, atol=1e-5)
