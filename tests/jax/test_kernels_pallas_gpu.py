# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.kernels import pallas_gpu


def _as_matrix(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _require_pallas_gpu() -> None:
    if not pallas_gpu.is_pallas_gpu_available():
        pytest.skip(
            "Pallas GPU kernels require jax.default_backend() == 'gpu' " "and importable jax.experimental.pallas"
        )


def _src_of_edge(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(indptr.shape[0] - 1, dtype=np.int32), np.diff(indptr))


def _operator_from_case(oracle_case, *, level_schedule: bool) -> JaxLinearARG:
    linarg = oracle_case.linarg
    return JaxLinearARG.from_lineararg_arrays(
        indptr=linarg.A.indptr,
        indices=linarg.A.indices,
        data=linarg.A.data,
        src_of_edge=_src_of_edge(linarg.A.indptr),
        variant_indices=linarg.variant_indices,
        flip=linarg.flip,
        sample_indices=linarg.sample_indices,
        nonunique_indices=linarg.nonunique_indices,
        n_variants=linarg.shape[1],
        n_samples=linarg.shape[0],
        backend=Backend.PALLAS_GPU,
        dtype=jnp.float32,
        level_schedule=level_schedule,
    )


def test_pallas_gpu_availability_is_false_without_gpu_backend() -> None:
    if jax.default_backend() == "gpu":
        pytest.skip("default JAX backend is GPU; availability is covered by integration tests")

    assert not pallas_gpu.is_pallas_gpu_available()


def test_compute_level_schedule_groups_edges_by_source_wavefront() -> None:
    schedule = pallas_gpu.compute_level_schedule(
        np.asarray([0, 2, 3, 3], dtype=np.int32),
        np.asarray([1, 2, 2], dtype=np.int32),
    )

    np.testing.assert_array_equal(schedule.edge_order, np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(schedule.level_offsets, np.asarray([0, 2, 3], dtype=np.int32))


def test_jax_lineararg_pallas_gpu_level_schedule_matches_oracle(oracle_case) -> None:
    _require_pallas_gpu()
    op = _operator_from_case(oracle_case, level_schedule=True)

    xw = np.asarray(op.matmat(oracle_case.w))
    xty = np.asarray(op.rmatmat(oracle_case.y))

    assert op.backend is Backend.PALLAS_GPU
    assert op.level_schedule is True
    assert xw.shape == oracle_case.Xw.shape
    assert xty.shape == oracle_case.XTy.shape
    np.testing.assert_allclose(xw, oracle_case.Xw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(xty, oracle_case.XTy, rtol=1e-5, atol=1e-5)


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
