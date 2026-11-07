"""Independent dense oracles for the internal CPU prototype."""

from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from packed import pack, product, weighted


def example(workers=2, chunk=2):
    # Two ragged graphs with 2 and 1 variants, 5 and 4 nodes.
    blocks = [
        dict(
            indptr=[0, 2, 3, 3, 3, 3],
            indices=[2, 4, 3],
            data=[1, 1, 1],
            nonunique_indices=list(range(5)),
            variant_indices=[0, 1],
            sample_indices=[2, 3, 4],
            flip=[False, True],
            min_index_to_keep=4,
        ),
        dict(
            indptr=[0, 2, 2, 2, 2],
            indices=[1, 2],
            data=[1, 1],
            nonunique_indices=list(range(4)),
            variant_indices=[0],
            sample_indices=[1, 2, 3],
            flip=[False],
            min_index_to_keep=3,
        ),
    ]
    dense = jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    graph, config = pack(blocks, workers=workers, chunk=chunk)
    return graph, config, dense


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("workers,chunk", [(1, 1), (2, 2), (4, 8)])
def test_products_and_weighted_derivatives(dtype, workers, chunk):
    graph, cfg, x = example(workers, chunk)
    x = x.astype(dtype)
    u = jnp.arange(15, dtype=dtype).reshape(3, 5) / 7
    d = jnp.array([0.3, -0.7, 1.2], dtype=dtype)
    for transpose in (False, True):

        def f(v):
            return product(graph, v, cfg, transpose=transpose)

        matrix = x.T if transpose else x
        np.testing.assert_allclose(jax.jit(f)(u), matrix @ u, rtol=2e-6, atol=2e-6)
        _, tangent = jax.jvp(f, (u,), (jnp.ones_like(u),))
        np.testing.assert_allclose(tangent, matrix @ jnp.ones_like(u), rtol=2e-6)
        np.testing.assert_allclose(jax.grad(lambda v: f(v).sum())(u), matrix.T @ jnp.ones_like(u))

    def f(d, u):
        return weighted(graph, d, u, cfg)

    def oracle(d, u):
        return x @ (d[:, None] * (x.T @ u))

    np.testing.assert_allclose(jax.jit(f)(d, u), oracle(d, u), rtol=3e-6, atol=3e-6)

    def loss(d, u):
        return jnp.sum(f(d, u) ** 2)

    def ref(d, u):
        return jnp.sum(oracle(d, u) ** 2)

    for actual, expected in zip(jax.jit(jax.grad(loss, (0, 1)))(d, u), jax.grad(ref, (0, 1))(d, u)):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    direction = (jnp.ones_like(d), jnp.ones_like(u))
    np.testing.assert_allclose(
        jax.jvp(f, (d, u), direction)[1], jax.jvp(oracle, (d, u), direction)[1], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(jax.jacfwd(jax.grad(loss, 0), 0)(d, u), jax.hessian(ref, 0)(d, u), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mapped_d,mapped_u", [(False, True), (True, False), (True, True)])
def test_batching_weights_and_inputs(mapped_d, mapped_u):
    graph, cfg, x = example()
    d = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 5 if mapped_d else jnp.ones(3, dtype=jnp.float32)
    u = jnp.arange(24, dtype=jnp.float32).reshape(4, 3, 2) / 7 if mapped_u else jnp.ones((3, 2), dtype=jnp.float32)
    axes = (0 if mapped_d else None, 0 if mapped_u else None)
    f = jax.vmap(lambda d, u: weighted(graph, d, u, cfg), in_axes=axes)
    ref = jax.vmap(lambda d, u: x @ (d[:, None] * (x.T @ u)), in_axes=axes)
    np.testing.assert_allclose(jax.jit(f)(d, u), ref(d, u), rtol=3e-6, atol=3e-6)
    for a, b in zip(
        jax.grad(lambda d, u: f(d, u).sum(), (0, 1))(d, u), jax.grad(lambda d, u: ref(d, u).sum(), (0, 1))(d, u)
    ):
        np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-6)


def test_mapped_vectors_linearize_checkpoint_and_concurrency():
    graph, cfg, x = example()
    u = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)

    def f(v):
        return product(graph, v, cfg)

    np.testing.assert_allclose(jax.jit(jax.vmap(f, in_axes=1, out_axes=1))(u), x @ u)
    _, linear = jax.linearize(f, u)
    np.testing.assert_allclose(linear(u), x @ u)

    def loss(v):
        return jnp.sum(jax.checkpoint(f)(v) ** 2)

    np.testing.assert_allclose(jax.grad(loss)(u), 2 * x.T @ x @ u)
    compiled = jax.jit(f)
    compiled(u).block_until_ready()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: np.asarray(compiled(u + i)), range(8)))
    for i, result in enumerate(results):
        np.testing.assert_allclose(result, x @ (u + i))


def test_reject_bad_boundaries():
    graph, cfg, _ = example()
    with pytest.raises(ValueError):
        weighted(graph, jnp.ones(4), jnp.ones(3), cfg)
    with pytest.raises(ValueError):
        product(graph, jnp.ones(4), cfg)
    with pytest.raises(ValueError):
        pack([], workers=1, chunk=1)


def test_parameterized_solve_and_differentiable_model():
    from jax.scipy.sparse.linalg import cg

    graph, cfg, x = example()
    x = x.astype(jnp.float64)
    rhs = jnp.array([0.5, -1.0, 2.0], dtype=jnp.float64)
    theta = jnp.array([-0.2, 0.3, 0.8], dtype=jnp.float64)

    def loss(theta):
        diagonal = jnp.exp(theta)

        def apply(v):
            return weighted(graph, diagonal, v, cfg) + 0.7 * v

        solution, _ = cg(apply, rhs, tol=1e-12, maxiter=30)
        prediction = product(graph, theta, cfg)
        return rhs @ solution + jnp.sum(jax.nn.softplus(prediction))

    def reference(theta):
        matrix = x @ jnp.diag(jnp.exp(theta)) @ x.T + 0.7 * jnp.eye(3)
        return rhs @ jnp.linalg.solve(matrix, rhs) + jnp.sum(jax.nn.softplus(x @ theta))

    np.testing.assert_allclose(jax.jit(loss)(theta), reference(theta), rtol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(theta), jax.grad(reference)(theta), rtol=1e-9, atol=1e-9)


def test_nonleading_nested_maps_unused_outputs_and_checkpoint_weights():
    graph, cfg, x = example()
    x = x.astype(jnp.float32)
    u = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)

    def f(v):
        return product(graph, v, cfg)

    mapped = jax.vmap(jax.vmap(f, in_axes=1, out_axes=1))
    np.testing.assert_allclose(jax.jit(mapped)(u), jnp.einsum("ij,bjk->bik", x, u))
    d = jnp.array([0.0, -0.3, 0.8], dtype=jnp.float32)
    values = u[0]

    def loss(d, v):
        return jnp.sum(jax.checkpoint(lambda dd, vv: weighted(graph, dd, vv, cfg))(d, v) ** 2)

    def ref(d, v):
        return jnp.sum((x @ (d[:, None] * (x.T @ v))) ** 2)

    for a, b in zip(jax.grad(loss, (0, 1))(d, values), jax.grad(ref, (0, 1))(d, values)):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)

    def unused(v):
        f(v)
        return jnp.sum(v)

    np.testing.assert_array_equal(jax.grad(unused)(values), jnp.ones_like(values))
