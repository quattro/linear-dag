# pattern: Mixed (fixture ingress and numerical contract checks)

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from linear_dag import as_lineax_operator
from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxLinearARG, JaxParallelOperator


@pytest.mark.parametrize("backend", [Backend.PURE_JAX, Backend.FFI_CPU])
@pytest.mark.parametrize("multi", [False, True])
def test_products_transpose_and_materialization(linarg_h5_path, first_block_name, backend, multi):
    if multi:
        op = JaxParallelOperator.from_hdf5(linarg_h5_path, backend=backend)
    else:
        op = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=backend)
    adapted = as_lineax_operator(op)
    assert isinstance(adapted, lx.AbstractLinearOperator)
    assert adapted.in_size() == op.shape[1]
    assert adapted.out_size() == op.shape[0]
    b = jnp.linspace(-1, 1, op.shape[1])
    u = jnp.linspace(-1, 1, op.shape[0])
    np.testing.assert_allclose(jax.jit(lambda a, b: a.mv(b))(adapted, b), op.matvec(b), atol=1e-5)
    np.testing.assert_allclose(jax.jit(lambda a, u: a.T.mv(u))(adapted, u), op.rmatvec(u), atol=1e-5)
    np.testing.assert_allclose(adapted.T.T.mv(b), adapted.mv(b), atol=1e-5)
    np.testing.assert_allclose(
        jax.vmap(adapted.mv)(jnp.stack([b, 2 * b])), jnp.stack([op.matvec(b), op.matvec(2 * b)]), atol=1e-5
    )
    dense = op.matmat(jnp.eye(op.shape[1]))
    np.testing.assert_allclose(lx.materialise(adapted).as_matrix(), dense, atol=1e-5)
    np.testing.assert_allclose(lx.diagonal(adapted), jnp.diag(dense), atol=1e-5)
    assert not lx.is_positive_semidefinite(adapted)


def test_cg_solve_derivatives_and_failure(linarg_h5_path, first_block_name, monkeypatch):
    with jax.enable_x64():
        op = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.FFI_CPU, dtype=jnp.float64)
        grm = JaxGRMOperator(op, diagonal=jnp.ones(op.shape[1], dtype=jnp.float64) / op.shape[1])
        adapter = as_lineax_operator(grm)
        assert lx.is_symmetric(adapter)
        assert not lx.is_positive_semidefinite(adapter)
        dense = adapter.as_matrix()
        identity = lx.IdentityLinearOperator(adapter.in_structure())
        rhs = jnp.linspace(-1, 1, op.shape[0], dtype=jnp.float64)
        theta = jnp.array([-0.3, 0.2])

        def no_dense(*args):
            raise AssertionError("iterative solve materialized the operator")

        monkeypatch.setattr(type(adapter), "as_matrix", no_dense)

        def solve(t, b, a):
            variance = jnp.exp(t)
            covariance = lx.TaggedLinearOperator(a * variance[0] + identity * variance[1], lx.positive_semidefinite_tag)
            return lx.linear_solve(covariance, b, lx.CG(rtol=1e-10, atol=1e-10, max_steps=200)).value

        def reference(t, b):
            v = jnp.exp(t)
            return jnp.linalg.solve(v[0] * dense + v[1] * jnp.eye(len(rhs)), b)

        observed = jax.jit(solve)(theta, rhs, adapter)
        np.testing.assert_allclose(observed, reference(theta, rhs), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(
            jax.grad(lambda b: solve(theta, b, adapter).sum())(rhs),
            jax.grad(lambda b: reference(theta, b).sum())(rhs),
            rtol=1e-7,
            atol=1e-7,
        )

        def loss(t):
            return rhs @ solve(t, rhs, adapter)

        def ref(t):
            return rhs @ reference(t, rhs)

        np.testing.assert_allclose(jax.jit(jax.grad(loss))(theta), jax.grad(ref)(theta), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(
            jax.jvp(loss, (theta,), (jnp.ones_like(theta),))[1],
            jax.jvp(ref, (theta,), (jnp.ones_like(theta),))[1],
            rtol=1e-7,
        )
        bad = lx.linear_solve(
            lx.TaggedLinearOperator(adapter + identity, lx.positive_semidefinite_tag),
            rhs,
            lx.CG(rtol=1e-14, atol=0, max_steps=1),
            throw=False,
        )
        assert bad.result != lx.RESULTS.successful


def test_signed_diagonal_gradients(linarg_h5_path, first_block_name):
    op = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.FFI_CPU)
    base = JaxGRMOperator(op, center=False, diagonal=jnp.ones(op.shape[1]))
    x = op.matmat(jnp.eye(op.shape[1]))
    u = jnp.linspace(-1, 1, op.shape[0])
    d = jnp.linspace(-0.1, 0.1, op.shape[1])

    def loss(weights):
        grm = eqx.tree_at(lambda g: g.diagonal, base, weights)
        adapted = as_lineax_operator(grm)
        assert not lx.is_positive_semidefinite(adapted)
        return u @ adapted.mv(u)

    np.testing.assert_allclose(jax.jit(jax.grad(loss))(d), (x.T @ u) ** 2, rtol=2e-5, atol=1e-4)
    with pytest.raises(TypeError, match="expected"):
        as_lineax_operator(jnp.eye(2))
