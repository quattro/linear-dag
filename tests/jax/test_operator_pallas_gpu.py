# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.kernels import pallas_gpu
from tests.jax.oracle import make_oracle_cases


def _require_pallas_gpu() -> None:
    if not pallas_gpu.is_pallas_gpu_available():
        pytest.skip(
            "Pallas GPU operator tests require jax.default_backend() == 'gpu' " "and importable jax.experimental.pallas"
        )


def _src_of_edge(indptr: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(indptr.shape[0] - 1, dtype=np.int32), np.diff(indptr))


def _operator_from_case(oracle_case, *, backend: Backend) -> JaxLinearARG:
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
        backend=backend,
        dtype=jnp.float32,
    )


@pytest.mark.parametrize("backend", [Backend.PALLAS_GPU, Backend.AUTO])
def test_jax_lineararg_pallas_gpu_forward_product_matches_oracle(backend, oracle_case) -> None:
    _require_pallas_gpu()
    op = _operator_from_case(oracle_case, backend=backend)

    assert op.backend is Backend.PALLAS_GPU
    np.testing.assert_allclose(np.asarray(op.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", [Backend.PALLAS_GPU, Backend.AUTO])
def test_jax_lineararg_pallas_gpu_reverse_product_matches_oracle(backend, oracle_case) -> None:
    _require_pallas_gpu()
    op = _operator_from_case(oracle_case, backend=backend)

    assert op.backend is Backend.PALLAS_GPU
    np.testing.assert_allclose(np.asarray(op.rmatmat(oracle_case.y)), oracle_case.XTy, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case_name", ["unflipped_k3", "flipped_k3"])
def test_jax_lineararg_pallas_gpu_forward_gradient_uses_transpose_solve(
    case_name,
    linarg_h5_path,
    first_block_name,
) -> None:
    _require_pallas_gpu()
    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    case = cases[case_name]
    pallas_op = _operator_from_case(case, backend=Backend.PALLAS_GPU)
    pure_op = _operator_from_case(case, backend=Backend.PURE_JAX)
    target = jnp.zeros_like(jnp.asarray(case.Xw, dtype=jnp.float32))

    @jax.jit
    def loss(values):
        residual = pallas_op.matmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    values = jnp.asarray(case.w, dtype=jnp.float32)
    residual = pallas_op.matmat(values) - target
    expected = pure_op.rmatmat(residual)
    actual = jax.jit(jax.grad(loss))(values)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("case_name", ["unflipped_k3", "flipped_k3"])
def test_jax_lineararg_pallas_gpu_reverse_gradient_uses_forward_solve(
    case_name,
    linarg_h5_path,
    first_block_name,
) -> None:
    _require_pallas_gpu()
    cases = {case.name: case for case in make_oracle_cases(linarg_h5_path, first_block_name)}
    case = cases[case_name]
    pallas_op = _operator_from_case(case, backend=Backend.PALLAS_GPU)
    pure_op = _operator_from_case(case, backend=Backend.PURE_JAX)
    target = jnp.zeros_like(jnp.asarray(case.XTy, dtype=jnp.float32))

    @jax.jit
    def loss(values):
        residual = pallas_op.rmatmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    values = jnp.asarray(case.y, dtype=jnp.float32)
    residual = pallas_op.rmatmat(values) - target
    expected = pure_op.matmat(residual)
    actual = jax.jit(jax.grad(loss))(values)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
