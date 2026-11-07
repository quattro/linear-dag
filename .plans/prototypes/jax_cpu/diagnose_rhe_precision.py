"""Reproduce JAX RHE uncertainty overflow independently of graph traversal."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from linear_dag.association._heritability_jax import _compute_err_variance_vectorized


def experiment(dtype):
    n = 10
    matrix = jnp.diag(jnp.linspace(1, 10, n, dtype=dtype) * 1e9)
    y = jnp.arange(1, n + 1, dtype=dtype)[:, None] / 10
    solutions = jnp.array([[1e-9], [1.0]], dtype=dtype)
    trace = jnp.trace(matrix)
    sqtrace = jnp.sum(matrix * matrix)
    denom = n * sqtrace - trace**2
    result = _compute_err_variance_vectorized(
        lambda x: matrix @ x, y, solutions, sqtrace, trace, 100, jnp.asarray(n, dtype=dtype)
    )
    return dict(
        dtype=str(dtype),
        denom=float(denom),
        denom_squared=float(denom * denom),
        scaled_variances=[np.asarray(result[0] * trace**2).tolist(), np.asarray(result[1] * n**2).tolist()],
        covariance=np.asarray(result[2] * trace * n).tolist(),
    )


if __name__ == "__main__":
    with jax.enable_x64():
        print(json.dumps([experiment(jnp.float32), experiment(jnp.float64)], indent=2))
