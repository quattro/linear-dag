# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.kernels import pallas_gpu
from linear_dag.core.jaxlinarg.padding import aligned_length_for_mosaic_gpu_transfer


class _ArrayShape:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)


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


def test_compute_level_schedule_can_reorder_noncontiguous_wavefront_edges() -> None:
    schedule = pallas_gpu.compute_level_schedule(
        np.asarray([0, 1, 2, 3, 3], dtype=np.int32),
        np.asarray([1, 3, 3], dtype=np.int32),
    )

    np.testing.assert_array_equal(schedule.edge_order, np.asarray([0, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(schedule.level_offsets, np.asarray([0, 2, 3], dtype=np.int32))


def test_pallas_gpu_level_scheduled_kernels_use_schedule_in_interpret_mode(monkeypatch) -> None:
    def fail_serial(*args, **kwargs):
        raise AssertionError("level-scheduled path must not delegate to serial Pallas GPU solve")

    monkeypatch.setattr(pallas_gpu, "pallas_gpu_solve_forward", fail_serial)
    monkeypatch.setattr(pallas_gpu, "pallas_gpu_solve_backward", fail_serial)
    indptr = jnp.asarray(np.array([0, 1, 2, 3, 3], dtype=np.int32))
    indices = jnp.asarray(np.array([1, 3, 3], dtype=np.int32))
    data = jnp.asarray(np.ones(3, dtype=np.float32))
    src_of_edge = jnp.asarray(_src_of_edge(np.asarray(indptr)))
    nonunique_indices = jnp.arange(4, dtype=jnp.int32)
    schedule = pallas_gpu.compute_level_schedule(np.asarray(indptr), np.asarray(indices))
    forward_b = jnp.asarray(np.array([[2.0], [0.0], [5.0], [0.0]], dtype=np.float32))
    backward_b = jnp.asarray(np.array([[0.0], [0.0], [0.0], [3.0]], dtype=np.float32))

    forward = pallas_gpu.pallas_gpu_solve_forward_level_scheduled(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        3,
        schedule,
        forward_b,
        interpret=True,
    )
    backward = pallas_gpu.pallas_gpu_solve_backward_level_scheduled(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        3,
        schedule,
        backward_b,
        interpret=True,
    )

    np.testing.assert_allclose(np.asarray(forward), np.array([[0.0], [0.0], [0.0], [7.0]], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(backward), np.array([[3.0], [3.0], [3.0], [3.0]], dtype=np.float32))


def test_mosaic_alignment_helpers_round_lengths_to_128_byte_transfers() -> None:
    assert aligned_length_for_mosaic_gpu_transfer(np.int32, 835) == 864
    assert aligned_length_for_mosaic_gpu_transfer(np.float32, 2008) == 2016
    assert aligned_length_for_mosaic_gpu_transfer(np.float64, 2008) == 2016


def test_mosaic_copy_constraints_reject_unpadded_hpc_shape() -> None:
    support = pallas_gpu.check_pallas_gpu_kernel_support(
        indptr=jnp.zeros((835,), dtype=jnp.int32),
        indices=jnp.zeros((2008,), dtype=jnp.int32),
        data=jnp.zeros((2008,), dtype=jnp.float32),
        src_of_edge=jnp.zeros((2008,), dtype=jnp.int32),
        nonunique_indices=jnp.zeros((834,), dtype=jnp.int32),
        b=jnp.zeros((834, 1), dtype=jnp.float32),
        max_shared_memory_bytes=48 * 1024,
    )

    assert not support.supported
    assert "128-byte aligned transfers" in support.reason


def test_mosaic_copy_constraints_accept_padded_transfer_lengths() -> None:
    n_indptr = aligned_length_for_mosaic_gpu_transfer(np.int32, 835)
    n_edges = aligned_length_for_mosaic_gpu_transfer(np.int32, 2008)
    n_rows = aligned_length_for_mosaic_gpu_transfer(np.float32, 834)
    refs = pallas_gpu._mosaic_visible_refs(
        indptr=jnp.zeros((n_indptr,), dtype=jnp.int32),
        indices=jnp.zeros((n_edges,), dtype=jnp.int32),
        data=jnp.zeros((n_edges,), dtype=jnp.float32),
        src_of_edge=jnp.zeros((n_edges,), dtype=jnp.int32),
        nonunique_indices=jnp.zeros((n_indptr,), dtype=jnp.int32),
        b=jnp.zeros((n_rows, 1), dtype=jnp.float32),
    )

    support = pallas_gpu._check_mosaic_gpu_copy_constraints(refs)

    assert support.supported


def test_scheduled_resource_estimate_does_not_scale_with_total_edges() -> None:
    small = pallas_gpu._estimate_scheduled_kernel_resources(
        indptr=jnp.zeros((32,), dtype=jnp.int32),
        indices=jnp.zeros((32,), dtype=jnp.int32),
        data=jnp.zeros((32,), dtype=jnp.float32),
        src_of_edge=jnp.zeros((32,), dtype=jnp.int32),
        nonunique_indices=jnp.zeros((32,), dtype=jnp.int32),
        b=jnp.zeros((32, 3), dtype=jnp.float32),
    )
    large = pallas_gpu._estimate_scheduled_kernel_resources(
        indptr=jnp.zeros((835,), dtype=jnp.int32),
        indices=jnp.zeros((2008,), dtype=jnp.int32),
        data=jnp.zeros((2008,), dtype=jnp.float32),
        src_of_edge=jnp.zeros((2008,), dtype=jnp.int32),
        nonunique_indices=jnp.zeros((834,), dtype=jnp.int32),
        b=jnp.zeros((834, 3), dtype=jnp.float32),
    )

    assert small.estimated_smem_bytes == large.estimated_smem_bytes
    assert large.estimated_work_items > small.estimated_work_items


def test_pallas_gpu_support_check_rejects_mosaic_lowering_smem_limit() -> None:
    support = pallas_gpu.check_pallas_gpu_kernel_support(
        indptr=jnp.zeros((32,), dtype=jnp.int32),
        indices=jnp.zeros((32,), dtype=jnp.int32),
        data=jnp.zeros((32,), dtype=jnp.float32),
        src_of_edge=jnp.zeros((32,), dtype=jnp.int32),
        nonunique_indices=jnp.zeros((32,), dtype=jnp.int32),
        b=jnp.zeros((32, 3), dtype=jnp.float32),
        max_shared_memory_bytes=64,
    )

    assert not support.supported
    assert "Mosaic lowering shared memory" in support.reason


def test_pallas_gpu_support_check_rejects_large_serial_refs_as_mosaic_lowering_smem() -> None:
    support = pallas_gpu.check_pallas_gpu_kernel_support(
        indptr=_ArrayShape((572384,), np.int32),
        indices=_ArrayShape((4_000_000,), np.int32),
        data=_ArrayShape((4_000_000,), np.float32),
        src_of_edge=_ArrayShape((4_000_000,), np.int32),
        nonunique_indices=_ArrayShape((572384,), np.int32),
        b=_ArrayShape((3200, 1), np.float32),
        kernel_kind="serial",
        max_shared_memory_bytes=48 * 1024,
    )

    assert not support.supported
    assert "Mosaic lowering shared memory" in support.reason


def test_pallas_gpu_support_check_rejects_large_scheduled_refs_as_mosaic_lowering_smem() -> None:
    support = pallas_gpu.check_pallas_gpu_kernel_support(
        indptr=_ArrayShape((572384,), np.int32),
        indices=_ArrayShape((4_000_000,), np.int32),
        data=_ArrayShape((4_000_000,), np.float32),
        src_of_edge=_ArrayShape((4_000_000,), np.int32),
        nonunique_indices=_ArrayShape((572384,), np.int32),
        b=_ArrayShape((3200, 1), np.float32),
        kernel_kind="scheduled",
        max_shared_memory_bytes=48 * 1024,
    )

    assert not support.supported
    assert "Mosaic lowering shared memory" in support.reason


def test_pallas_gpu_support_check_rejects_scheduled_unpadded_transfers() -> None:
    support = pallas_gpu.check_pallas_gpu_kernel_support(
        indptr=jnp.zeros((835,), dtype=jnp.int32),
        indices=jnp.zeros((2008,), dtype=jnp.int32),
        data=jnp.zeros((2008,), dtype=jnp.float32),
        src_of_edge=jnp.zeros((2008,), dtype=jnp.int32),
        nonunique_indices=jnp.zeros((834,), dtype=jnp.int32),
        b=jnp.zeros((834, 3), dtype=jnp.float32),
        kernel_kind="scheduled",
        max_shared_memory_bytes=48 * 1024,
    )

    assert not support.supported
    assert "128-byte aligned transfers" in support.reason


def test_pallas_gpu_support_check_rejects_unsupported_dtypes() -> None:
    support = pallas_gpu.check_pallas_gpu_kernel_support(
        indptr=np.zeros((32,), dtype=np.int64),
        indices=np.zeros((32,), dtype=np.int64),
        data=jnp.zeros((32,), dtype=jnp.float32),
        src_of_edge=np.zeros((32,), dtype=np.int64),
        nonunique_indices=np.zeros((32,), dtype=np.int64),
        b=jnp.zeros((32, 3), dtype=jnp.float32),
    )

    assert not support.supported
    assert "int32" in support.reason


def test_pallas_gpu_forward_falls_back_to_pure_jax_for_unsupported_shape(monkeypatch) -> None:
    monkeypatch.setattr(pallas_gpu, "is_pallas_gpu_available", lambda: True)
    monkeypatch.setattr(
        pallas_gpu,
        "check_pallas_gpu_kernel_support",
        lambda **_kwargs: pallas_gpu.PallasGpuKernelSupport(False, "test unsupported shape"),
    )
    indptr = jnp.asarray(np.array([0, 1, 1], dtype=np.int32))
    indices = jnp.asarray(np.array([1], dtype=np.int32))
    data = jnp.asarray(np.array([2.0], dtype=np.float32))
    src_of_edge = jnp.asarray(np.array([0], dtype=np.int32))
    nonunique_indices = jnp.arange(2, dtype=jnp.int32)
    b = jnp.asarray(np.array([[3.0], [0.0]], dtype=np.float32))
    pallas_gpu.reset_pallas_gpu_fallback_count()

    with pytest.warns(UserWarning, match="falling back to pure JAX"):
        actual = pallas_gpu.pallas_gpu_solve_forward(
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            0,
            b,
        )

    np.testing.assert_allclose(np.asarray(actual), np.array([[3.0], [6.0]], dtype=np.float32))
    assert pallas_gpu.pallas_gpu_fallback_count() == 1


def test_pallas_gpu_forward_uses_kernel_when_shape_is_supported(monkeypatch) -> None:
    monkeypatch.setattr(pallas_gpu, "is_pallas_gpu_available", lambda: True)
    monkeypatch.setattr(
        pallas_gpu,
        "check_pallas_gpu_kernel_support",
        lambda **_kwargs: pallas_gpu.PallasGpuKernelSupport(True, ""),
    )
    calls = []

    def fake_call_kernel(kernel, indptr, indices, data, src_of_edge, nonunique_indices, b, *, name):
        del kernel, indptr, indices, data, src_of_edge, nonunique_indices, name
        calls.append(b)
        return b + 1

    monkeypatch.setattr(pallas_gpu, "_call_kernel", fake_call_kernel)
    b = jnp.zeros((2, 1), dtype=jnp.float32)

    actual = pallas_gpu.pallas_gpu_solve_forward(
        jnp.asarray(np.array([0, 1, 1], dtype=np.int32)),
        jnp.asarray(np.array([1], dtype=np.int32)),
        jnp.asarray(np.array([2.0], dtype=np.float32)),
        jnp.asarray(np.array([0], dtype=np.int32)),
        jnp.arange(2, dtype=jnp.int32),
        0,
        b,
    )

    assert len(calls) == 1
    np.testing.assert_allclose(np.asarray(actual), np.ones((2, 1), dtype=np.float32))


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
