# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG


def _tiny_two_by_two_operator() -> JaxLinearARG:
    return JaxLinearARG.from_lineararg_arrays(
        indptr=np.array([0, 1, 2, 3, 3, 3], dtype=np.int32),
        indices=np.array([2, 4, 3], dtype=np.int32),
        data=np.ones(3, dtype=np.float32),
        src_of_edge=np.array([0, 1, 2], dtype=np.int32),
        variant_indices=np.array([0, 1], dtype=np.int32),
        flip=np.array([False, False]),
        sample_indices=np.array([3, 4], dtype=np.int32),
        nonunique_indices=None,
        n_variants=2,
        n_samples=2,
        backend=Backend.PURE_JAX,
        dtype=jnp.float32,
    )


def _operator_with_disconnected_variant() -> JaxLinearARG:
    return JaxLinearARG.from_lineararg_arrays(
        indptr=np.array([0, 1, 1, 1], dtype=np.int32),
        indices=np.array([2], dtype=np.int32),
        data=np.ones(1, dtype=np.float32),
        src_of_edge=np.array([0], dtype=np.int32),
        variant_indices=np.array([0, 1], dtype=np.int32),
        flip=np.array([False, False]),
        sample_indices=np.array([2], dtype=np.int32),
        nonunique_indices=None,
        n_variants=2,
        n_samples=1,
        backend=Backend.PURE_JAX,
        dtype=jnp.float32,
    )


def _finite_difference_grad(fn, x: np.ndarray, *, eps: float = 1e-2) -> np.ndarray:
    grad = np.zeros_like(x, dtype=np.float32)
    for index in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[index] += eps
        minus[index] -= eps
        grad[index] = (float(fn(plus)) - float(fn(minus))) / (2 * eps)
    return grad


def test_forward_product_reverse_mode_gradient_matches_finite_difference_under_jit():
    op = _tiny_two_by_two_operator()
    w = jnp.array([0.25, -1.5], dtype=jnp.float32)
    target = jnp.array([1.0, 0.5], dtype=jnp.float32)

    @jax.jit
    def loss(values):
        residual = op.matmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    actual = np.asarray(jax.grad(loss)(w))
    expected = _finite_difference_grad(lambda values: loss(jnp.asarray(values)), np.asarray(w))

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_reverse_product_reverse_mode_gradient_matches_finite_difference_under_jit():
    op = _tiny_two_by_two_operator()
    y = jnp.array([1.25, -0.5], dtype=jnp.float32)
    target = jnp.array([-0.75, 2.0], dtype=jnp.float32)

    @jax.jit
    def loss(values):
        residual = op.T.matmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    actual = np.asarray(jax.grad(loss)(y))
    expected = _finite_difference_grad(lambda values: loss(jnp.asarray(values)), np.asarray(y))

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_forward_product_gradient_is_zero_for_disconnected_variant_entry():
    op = _operator_with_disconnected_variant()
    w = jnp.array([0.25, 10.0], dtype=jnp.float32)
    target = jnp.array([1.0], dtype=jnp.float32)

    @jax.jit
    def loss(values):
        residual = op.matmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    grad = np.asarray(jax.grad(loss)(w))

    assert grad[1] == 0.0


def test_solve_dispatch_uses_custom_vjp_not_forward_mode_jvp():
    op = _tiny_two_by_two_operator()
    w = jnp.array([0.25, -1.5], dtype=jnp.float32)

    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jvp(lambda values: op.matmat(values), (w,), (jnp.ones_like(w),))
