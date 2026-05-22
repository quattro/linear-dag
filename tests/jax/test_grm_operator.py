# pattern: Functional Core

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.sharding import Mesh

from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxLinearARG, JaxParallelOperator
from linear_dag.core.parallel_processing import GRMOperator


def _mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]), ("blocks",))


def _two_device_cpu_mesh_or_skip() -> Mesh:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip(
            "requires at least two CPU devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import"
        )
    return Mesh(np.asarray(devices[:2]), ("blocks",))


def _jax_block(linearg) -> JaxLinearARG:
    return JaxLinearARG.from_lineararg(
        linearg,
        backend=Backend.PURE_JAX,
        dtype=jnp.float32,
    )


def _dense_genotypes(op) -> np.ndarray:
    return np.asarray(op.matmat(jnp.eye(op.shape[1], dtype=jnp.float32)))


def _expected_grm_product(op, y, *, alpha=-1.0, center=True):
    x = _dense_genotypes(op)
    y = np.asarray(y)
    was_vector = y.ndim == 1
    if y.ndim == 1:
        y = y[:, None]
    weights = _weights_from_operator(op, alpha=alpha)
    if center:
        freqs = _frequencies_from_operator(op)
        x = x - freqs[None, :]
    result = x @ (weights[:, None] * (x.T @ np.asarray(y)))
    return result[:, 0] if was_vector else result


def _frequencies_from_operator(op):
    if isinstance(op, JaxParallelOperator):
        return np.concatenate([np.asarray(block.allele_counts) / block.n_samples for block in op.blocks])
    return np.asarray(op.allele_counts) / op.n_samples


def _weights_from_operator(op, *, alpha):
    frequencies = _frequencies_from_operator(op)
    pq = frequencies * (1.0 - frequencies)
    safe_pq = np.where(pq > 0, pq, 1.0)
    return np.where(pq > 0, safe_pq**alpha, 0.0)


def test_jax_grm_single_block_matches_dense_centered_alpha(oracle_case):
    op = _jax_block(oracle_case.linarg)
    grm = JaxGRMOperator(op, alpha=0.5)
    y = jnp.asarray(oracle_case.y, dtype=jnp.float32)

    result = grm.matmat(y)
    expected = _expected_grm_product(op, y, alpha=0.5)

    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-4)


def test_jax_grm_uncentered_uses_raw_genotypes(oracle_case):
    op = _jax_block(oracle_case.linarg)
    grm = JaxGRMOperator(op, center=False)
    y = jnp.asarray(oracle_case.y, dtype=jnp.float32)

    result = grm.matmat(y)
    expected = _expected_grm_product(op, y, center=False)

    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-4)


def test_jax_grm_vector_inputs_restore_vector_outputs(oracle_case):
    op = _jax_block(oracle_case.linarg)
    grm = JaxGRMOperator(op)
    y = jnp.asarray(oracle_case.y, dtype=jnp.float32).reshape(op.shape[0], -1)[:, 0]

    result = grm.matvec(y)
    expected = _expected_grm_product(op, y[:, None])[:, 0]

    assert grm.shape == (op.shape[0], op.shape[0])
    assert result.shape == (op.shape[0],)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-4)


def test_jax_grm_jit_matches_eager(oracle_case):
    op = _jax_block(oracle_case.linarg)
    grm = JaxGRMOperator(op)
    y = jnp.asarray(oracle_case.y, dtype=jnp.float32)

    eager = grm.matmat(y)
    compiled = eqx.filter_jit(lambda operator, values: operator.matmat(values))(grm, y)

    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-4, atol=1e-4)


def test_jax_grm_reverse_mode_gradient_uses_symmetric_adjoint(oracle_case):
    op = _jax_block(oracle_case.linarg)
    grm = JaxGRMOperator(op, alpha=0.5)
    y = jnp.asarray(oracle_case.y, dtype=jnp.float32)
    target = jnp.zeros_like(y)

    @jax.jit
    def loss(values):
        residual = grm.matmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    residual = grm.matmat(y) - target
    expected = grm.matmat(residual)
    actual = jax.jit(jax.grad(loss))(y)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-3, atol=1e-3)


def test_jax_grm_parallel_operator_matches_dense_blocks(linarg_h5_path, linarg_block_metadata):
    op = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    grm = JaxGRMOperator(op, alpha=-1.0)
    y = jnp.arange(op.shape[0] * 3, dtype=jnp.float32).reshape(op.shape[0], 3)

    result = grm.matmat(y)
    expected = _expected_grm_product(op, y, alpha=-1.0)

    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-4)


def test_jax_grm_blockwise_matmat_matches_default_parallel_operator(linarg_h5_path, linarg_block_metadata):
    op = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    grm = JaxGRMOperator(op, alpha=-1.0)
    y = jnp.arange(op.shape[0] * 3, dtype=jnp.float32).reshape(op.shape[0], 3)

    result = grm.matmat_blockwise(y)
    expected = grm.matmat(y)

    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-4, atol=1e-4)


def test_jax_grm_blockwise_matvec_restores_vector_outputs(linarg_h5_path, linarg_block_metadata):
    op = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    grm = JaxGRMOperator(op, alpha=-1.0)
    y = jnp.arange(op.shape[0], dtype=jnp.float32)

    result = grm.matmat_blockwise(y)
    expected = grm.matmat(y)

    assert result.shape == (op.shape[0],)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-4, atol=1e-4)


def test_jax_grm_matches_existing_grm_operator(linarg_h5_path, linarg_block_metadata):
    op = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    grm = JaxGRMOperator(op, alpha=0.5)
    y = jnp.arange(op.shape[0] * 2, dtype=jnp.float32).reshape(op.shape[0], 2)

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=0.5) as expected_grm:
        expected = expected_grm @ np.asarray(y)

    np.testing.assert_allclose(np.asarray(grm @ y), expected, rtol=1e-3, atol=1e-3)


def test_jax_grm_parallel_operator_sharded_matmat_matches_single_device(linarg_h5_path, linarg_block_metadata):
    two_device = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_two_device_cpu_mesh_or_skip(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    single_device = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    y = jnp.arange(two_device.shape[0] * 2, dtype=jnp.float32).reshape(two_device.shape[0], 2)

    sharded = JaxGRMOperator(two_device).matmat(y)
    expected = JaxGRMOperator(single_device).matmat(y)

    np.testing.assert_allclose(np.asarray(sharded), np.asarray(expected), rtol=1e-4, atol=1e-4)
