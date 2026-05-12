# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg.kernels import pallas_tpu
from linear_dag.core.jaxlinarg.kernels.pure_jax import (
    pure_jax_solve_backward_compressed,
    pure_jax_solve_forward_compressed,
)


def _solve_args(dtype=np.float32) -> tuple:
    indptr = jnp.asarray(np.array([0, 1, 1], dtype=np.int32))
    indices = jnp.asarray(np.array([1], dtype=np.int32))
    data = jnp.asarray(np.ones(1, dtype=dtype))
    src_of_edge = jnp.asarray(np.array([0], dtype=np.int32))
    nonunique_indices = jnp.asarray(np.array([0, 1], dtype=np.int32))
    b = jnp.asarray(np.array([[2.0], [0.0]], dtype=dtype))
    return indptr, indices, data, src_of_edge, nonunique_indices, 0, b


def test_pallas_tpu_availability_requires_tpu_backend(monkeypatch) -> None:
    monkeypatch.setattr(pallas_tpu.jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(pallas_tpu, "pl", object())

    assert not pallas_tpu.is_pallas_tpu_available()


def test_pallas_tpu_availability_requires_pallas_import(monkeypatch) -> None:
    monkeypatch.setattr(pallas_tpu.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(pallas_tpu, "pl", None)

    assert not pallas_tpu.is_pallas_tpu_available()


def test_pallas_tpu_solve_requires_tpu_backend_without_interpret() -> None:
    with pytest.raises(RuntimeError, match="Pallas TPU backend is unavailable"):
        pallas_tpu.pallas_tpu_solve_forward(*_solve_args())


def test_pallas_tpu_forward_and_backward_match_pure_jax_in_interpret_mode() -> None:
    indptr, indices, data, src_of_edge, nonunique_indices, min_index_to_keep, b = _solve_args()
    backward_b = jnp.asarray(np.array([[0.0], [3.0]], dtype=np.float32))

    forward = pallas_tpu.pallas_tpu_solve_forward(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
        interpret=True,
    )
    expected_forward = pure_jax_solve_forward_compressed(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )
    backward = pallas_tpu.pallas_tpu_solve_backward(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        backward_b,
        interpret=True,
    )
    expected_backward = pure_jax_solve_backward_compressed(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        backward_b,
    )

    np.testing.assert_allclose(np.asarray(forward), np.asarray(expected_forward), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(backward), np.asarray(expected_backward), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not jax.config.jax_enable_x64, reason="JAX x64 is disabled")
def test_pallas_tpu_interpret_mode_supports_float64_when_enabled() -> None:
    result = pallas_tpu.pallas_tpu_solve_forward(*_solve_args(np.float64), interpret=True)

    assert result.dtype == jnp.float64
