# pattern: Mixed (unavoidable)
# Reason: Algebraic GRM assertions share concrete JAX mesh, compilation,
# device-residency, and HDF5 oracle integration fixtures.

from __future__ import annotations

import inspect

from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.extend import core as jax_core
from jax.sharding import Mesh

from linear_dag.core.jaxlinarg import Backend, grm as grm_module, JaxGRMOperator, JaxLinearARG, JaxParallelOperator
from linear_dag.core.jaxlinarg.ingress import _packed_from_block_arrays
from linear_dag.core.jaxlinarg.packing import LinearARGBlockArrays
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


def _graph_mesh(num_devices: int = 1) -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:num_devices]), ("graph",))


def _two_device_graph_mesh_or_skip() -> Mesh:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip(
            "requires at least two CPU devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import"
        )
    return _graph_mesh(2)


def _jax_block(linearg) -> JaxLinearARG:
    return JaxLinearARG.from_lineararg(
        linearg,
        backend=Backend.PURE_JAX,
        dtype=jnp.float32,
    )


def _block_arrays(linearg, *, dtype: Any = np.float32) -> LinearARGBlockArrays:
    linearg.calculate_nonunique_indices()
    return LinearARGBlockArrays(
        indptr=np.asarray(linearg.A.indptr, dtype=np.int32),
        indices=np.asarray(linearg.A.indices, dtype=np.int32),
        data=np.asarray(linearg.A.data, dtype=dtype),
        variant_indices=np.asarray(linearg.variant_indices, dtype=np.int32),
        flip=np.asarray(linearg.flip, dtype=np.bool_),
        sample_indices=np.asarray(linearg.sample_indices, dtype=np.int32),
        nonunique_indices=np.asarray(linearg.nonunique_indices, dtype=np.int32),
        allele_counts=np.asarray(linearg.allele_counts, dtype=np.int32),
        n_variants=linearg.shape[1],
        n_samples=linearg.shape[0],
    )


def _packed_oracle_operator(oracle_case, *, mesh: Mesh | None = None, dtype: Any = np.float32):
    return _packed_from_block_arrays(
        (_block_arrays(oracle_case.linarg, dtype=dtype),),
        mesh=_graph_mesh() if mesh is None else mesh,
        dtype=dtype,
        allow_excess_padding=True,
    ).operator


def _walk_ir_operations(value: Any):
    operation = getattr(value, "operation", value)
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for nested in block.operations:
                yield from _walk_ir_operations(nested)


def _recursive_array_constant_bytes(closed_jaxpr: jax_core.ClosedJaxpr) -> int:
    total = 0

    def add_constant(constant: Any) -> None:
        nonlocal total
        if isinstance(constant, (jax.Array, np.ndarray)):
            total += int(constant.size * constant.dtype.itemsize)
            return
        lower_val = getattr(jax.typeof(constant), "lower_val", None)
        if lower_val is not None:
            for lowered in lower_val(constant):
                add_constant(lowered)

    def visit(value: Any) -> None:
        if isinstance(value, jax_core.Jaxpr):
            for constant in getattr(value, "consts", ()):
                add_constant(constant)
            for equation in value.eqns:
                visit(equation.params)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(closed_jaxpr)
    return total


def _dense_genotypes(op) -> np.ndarray:
    return np.asarray(op.matmat(np.eye(op.shape[1], dtype=np.float32)))


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


def test_jax_grm_parallel_operator_device_local_matmat_matches_single_device(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
):
    def reject_shard_map(*args, **kwargs):
        raise AssertionError("ragged GRM execution should use explicit device-local state")

    monkeypatch.setattr("linear_dag.core.jaxlinarg.grm.jax.shard_map", reject_shard_map)
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

    grm = JaxGRMOperator(two_device)
    sharded = grm.matmat(y)
    expected = JaxGRMOperator(single_device).matmat(y)

    np.testing.assert_allclose(np.asarray(sharded), np.asarray(expected), rtol=1e-4, atol=1e-4)

    cotangent = jnp.linspace(-1.0, 1.0, y.size, dtype=y.dtype).reshape(y.shape)
    actual_gradient = jax.grad(lambda values: jnp.sum(grm.matmat(values) * cotangent))(y)
    expected_gradient = grm.matmat(cotangent)

    np.testing.assert_allclose(
        np.asarray(actual_gradient),
        np.asarray(expected_gradient),
        rtol=1e-4,
        atol=1e-4,
    )


def test_jax_grm_accepts_private_packed_operator_without_public_annotation_leakage(oracle_case) -> None:
    operator = _packed_oracle_operator(oracle_case)

    grm = JaxGRMOperator(operator, alpha=0.5)

    public_annotations = inspect.get_annotations(JaxGRMOperator, eval_str=False)
    assert grm.operator is operator
    assert "_PackedJaxLinearARG" not in str(public_annotations)
    assert "_hijax" not in str(public_annotations)


@pytest.mark.parametrize(
    ("center", "alpha"),
    [(True, -1.0), (True, 0.5), (False, -1.0)],
)
def test_packed_grm_matches_dense_cython_and_exact_operator(oracle_case, center: bool, alpha: float) -> None:
    packed = _packed_oracle_operator(oracle_case)
    exact = _jax_block(oracle_case.linarg)
    values = jnp.asarray(oracle_case.y, dtype=jnp.float32)
    packed_grm = JaxGRMOperator(packed, center=center, alpha=alpha)

    actual = grm_module._packed_grm_matmat(packed_grm, values)
    dense_expected = _expected_grm_product(oracle_case.linarg, values, center=center, alpha=alpha)
    exact_expected = JaxGRMOperator(exact, center=center, alpha=alpha).matmat(values)

    np.testing.assert_allclose(np.asarray(actual), dense_expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(exact_expected), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(packed_grm.matmat(values)), np.asarray(actual), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(packed_grm.rmatmat(values)), np.asarray(actual), rtol=1e-5, atol=1e-5)


def test_two_device_packed_grm_jvp_and_reverse_gradient_use_symmetric_product(oracle_case) -> None:
    packed = _packed_oracle_operator(oracle_case, mesh=_two_device_graph_mesh_or_skip())
    grm = JaxGRMOperator(packed, center=True, alpha=0.5)
    values = jnp.asarray(oracle_case.y, dtype=jnp.float32)
    tangent = jnp.linspace(-0.5, 0.75, values.size, dtype=values.dtype).reshape(values.shape)
    cotangent = jnp.linspace(-1.0, 1.0, values.size, dtype=values.dtype).reshape(values.shape)

    primal, jvp = jax.jvp(lambda dense: grm_module._packed_grm_matmat(grm, dense), (values,), (tangent,))

    def scalar_loss(explicit_grm, dense, output_cotangent):
        return jnp.sum(grm_module._packed_grm_matmat(explicit_grm, dense) * output_cotangent)

    gradient = jax.jit(jax.grad(scalar_loss, argnums=1))(grm, values, cotangent)
    compiled_loss = jax.jit(scalar_loss)
    epsilon = jnp.asarray(2e-3, dtype=values.dtype)
    finite_difference = (
        compiled_loss(grm, values + epsilon * tangent, cotangent)
        - compiled_loss(grm, values - epsilon * tangent, cotangent)
    ) / (2 * epsilon)

    np.testing.assert_allclose(np.asarray(primal), np.asarray(grm.matmat(values)), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(jvp),
        np.asarray(grm_module._packed_grm_matmat(grm, tangent)),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(gradient),
        np.asarray(grm_module._packed_grm_matmat(grm, cotangent)),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(jnp.vdot(gradient, tangent)),
        np.asarray(finite_difference),
        rtol=2e-2,
        atol=2e-2,
    )


def test_outer_jit_packed_grm_keeps_graph_explicit_resident_and_out_of_collectives(oracle_case) -> None:
    packed = _packed_oracle_operator(oracle_case, mesh=_two_device_graph_mesh_or_skip())
    grm = JaxGRMOperator(packed, center=True, alpha=-1.0)
    values = jnp.asarray(oracle_case.y, dtype=jnp.float32)
    shardings_before = tuple(component.sharding for component in packed.graph.components)
    compiled = jax.jit(grm_module._packed_grm_matmat)

    actual = compiled(grm, values)
    closed_jaxpr = jax.make_jaxpr(compiled)(grm, values)
    stablehlo = compiled.lower(grm, values).compiler_ir("stablehlo")
    collective_text = "\n".join(
        str(operation)
        for operation in _walk_ir_operations(stablehlo)
        if operation.name in {"stablehlo.all_gather", "stablehlo.all_to_all", "stablehlo.collective_broadcast"}
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(grm.matmat(values)), rtol=1e-4, atol=1e-4)
    assert _recursive_array_constant_bytes(closed_jaxpr) == 0
    assert tuple(component.sharding for component in packed.graph.components) == shardings_before
    assert collective_text == ""


def test_packed_grm_blockwise_fallback_is_rejected_as_exact_ragged_only(oracle_case) -> None:
    grm = JaxGRMOperator(_packed_oracle_operator(oracle_case))
    values = jnp.asarray(oracle_case.y, dtype=jnp.float32)

    with pytest.raises(TypeError, match="exact-ragged JaxParallelOperator"):
        grm.matmat_blockwise(values)


def test_packed_grm_zero_frequency_variants_are_inert(oracle_case) -> None:
    block = replace(
        _block_arrays(oracle_case.linarg),
        allele_counts=np.asarray([0] * oracle_case.linarg.shape[1], dtype=np.int32),
    )
    operator = _packed_from_block_arrays((block,), mesh=_graph_mesh()).operator
    values = jnp.asarray(oracle_case.y, dtype=jnp.float32)

    actual = JaxGRMOperator(operator, alpha=-1.0).matmat(values)

    np.testing.assert_array_equal(np.asarray(actual), np.zeros_like(np.asarray(values)))


def test_packed_grm_rejects_bad_rank_and_incompatible_allele_count_shape(oracle_case) -> None:
    operator = _packed_oracle_operator(oracle_case)
    grm = JaxGRMOperator(operator)

    with pytest.raises(ValueError, match="rank 1 or 2"):
        grm.matmat(jnp.ones((operator.n_samples, 1, 1), dtype=jnp.float32))
    with pytest.raises(ValueError, match="allele_counts length must match n_variants"):
        _packed_from_block_arrays(
            (replace(_block_arrays(oracle_case.linarg), allele_counts=np.asarray([1], dtype=np.int32)),),
            mesh=_graph_mesh(),
        )


@pytest.mark.skipif(not jax.config.x64_enabled, reason="requires JAX_ENABLE_X64=1")
def test_packed_grm_preserves_float64_when_x64_is_enabled(oracle_case) -> None:
    operator = _packed_oracle_operator(oracle_case, dtype=np.float64)
    values = jnp.asarray(oracle_case.y, dtype=jnp.float64)

    actual = JaxGRMOperator(operator, alpha=0.5).matmat(values)
    expected = _expected_grm_product(oracle_case.linarg, values, alpha=0.5)

    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-10, atol=1e-10)
