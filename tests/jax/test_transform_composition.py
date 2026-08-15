# pattern: Mixed (unavoidable)
# Reason: Pure nonlinear-loss and derivative assertions share concrete JAX
# mesh, compilation, sharding, rematerialization, and IR inspection.

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.extend import core as jax_core
from jax.sharding import Mesh

from linear_dag.core.jaxlinarg import _hijax as hijax_adapter
from linear_dag.core.jaxlinarg._hijax import _PackedGraphMappingSpec, _PackedGraphType
from linear_dag.core.jaxlinarg.ingress import _packed_from_block_arrays, from_block_arrays
from linear_dag.core.jaxlinarg.packing import LinearARGBlockArrays


def _graph_mesh(num_devices: int) -> Mesh:
    devices = jax.devices("cpu")
    if len(devices) < num_devices:
        pytest.skip(
            f"requires {num_devices} CPU devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import"
        )
    return Mesh(np.asarray(devices[:num_devices]), ("graph",))


def _block(*, dtype: Any = np.float32) -> LinearARGBlockArrays:
    return LinearARGBlockArrays(
        indptr=np.asarray([0, 2, 2, 2, 2, 2], dtype=np.int32),
        indices=np.asarray([3, 4], dtype=np.int32),
        data=np.asarray([1.0, 1.0], dtype=dtype),
        variant_indices=np.asarray([0, 0], dtype=np.int32),
        flip=np.asarray([False, True], dtype=np.bool_),
        sample_indices=np.asarray([4, 3], dtype=np.int32),
        nonunique_indices=np.asarray([0, 1, 1, 2, 3], dtype=np.int32),
        allele_counts=np.asarray([1, 1], dtype=np.int32),
        n_variants=2,
        n_samples=2,
    )


def _empty_block(*, dtype: Any = np.float32) -> LinearARGBlockArrays:
    return LinearARGBlockArrays(
        indptr=np.asarray([0, 0, 0], dtype=np.int32),
        indices=np.asarray([], dtype=np.int32),
        data=np.asarray([], dtype=dtype),
        variant_indices=np.asarray([], dtype=np.int32),
        flip=np.asarray([], dtype=np.bool_),
        sample_indices=np.asarray([1, 0], dtype=np.int32),
        nonunique_indices=np.asarray([0, 1], dtype=np.int32),
        allele_counts=np.asarray([], dtype=np.int32),
        n_variants=0,
        n_samples=2,
    )


def _operator(*blocks: LinearARGBlockArrays, num_devices: int = 1):
    return _packed_from_block_arrays(
        blocks,
        mesh=_graph_mesh(num_devices),
        allow_excess_padding=True,
    ).operator


def _case(operator, columns: int | None):
    theta_shape = (operator.n_variants,) if columns is None else (operator.n_variants, columns)
    target_shape = (operator.n_samples,) if columns is None else (operator.n_samples, columns)
    theta = jnp.linspace(-0.45, 0.75, int(np.prod(theta_shape)), dtype=jnp.float32).reshape(theta_shape)
    target = jnp.linspace(0.35, -0.25, int(np.prod(target_shape)), dtype=jnp.float32).reshape(target_shape)
    tangent = jnp.linspace(0.6, -0.4, theta.size, dtype=theta.dtype).reshape(theta.shape)
    return theta, target, tangent


def _research_loss(theta, operator, target):
    residual = jnp.tanh(operator.matmat(theta)) - target
    return jnp.mean(residual**2) + 1e-3 * jnp.sum(theta**2)


def _dense_matrix(block: LinearARGBlockArrays) -> jax.Array:
    exact = from_block_arrays(block)
    identity = jnp.eye(block.n_variants, dtype=jnp.asarray(block.data).dtype)
    return jnp.asarray(exact.matmat(identity))


def _dense_loss(theta, dense, target):
    residual = jnp.tanh(dense @ theta) - target
    return jnp.mean(residual**2) + 1e-3 * jnp.sum(theta**2)


def _loss_jvp(theta, operator, target, tangent):
    return jax.jvp(lambda dense: _research_loss(dense, operator, target), (theta,), (tangent,))


def _loss_vjp(theta, operator, target):
    value, pullback = jax.vjp(lambda dense: _research_loss(dense, operator, target), theta)
    (gradient,) = pullback(jnp.ones_like(value))
    return value, gradient


def _loss_hvp(theta, operator, target, tangent):
    def dense_gradient(dense):
        return jax.grad(_research_loss, argnums=0)(dense, operator, target)

    return jax.jvp(dense_gradient, (theta,), (tangent,))[1]


def _scan_losses(operator, thetas, targets):
    def body(graph_operator, inputs):
        theta, target = inputs
        return graph_operator, _research_loss(theta, graph_operator, target)

    final_operator, losses = jax.lax.scan(body, operator, (thetas, targets))
    return final_operator, losses


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


def _recursive_graph_widths(closed_jaxpr: jax_core.ClosedJaxpr) -> tuple[int, ...]:
    widths: list[int] = []

    def visit(value: Any) -> None:
        if isinstance(value, jax_core.Jaxpr):
            graph_vars = [
                variable
                for variable in (*value.constvars, *value.invars)
                if isinstance(getattr(variable, "aval", None), _PackedGraphType)
            ]
            if graph_vars:
                widths.append(len(graph_vars))
            for equation in value.eqns:
                graph_inputs = [
                    variable
                    for variable in equation.invars
                    if isinstance(getattr(variable, "aval", None), _PackedGraphType)
                ]
                if graph_inputs:
                    widths.append(len(graph_inputs))
                visit(equation.params)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(closed_jaxpr)
    return tuple(widths)


def _recursive_equation_structure(closed_jaxpr: jax_core.ClosedJaxpr) -> tuple[Any, ...]:
    structures: list[Any] = []

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, jax_core.Jaxpr):
            structures.append((path, tuple(equation.primitive.name for equation in value.eqns)))
            for equation_index, equation in enumerate(value.eqns):
                for name, nested in sorted(equation.params.items()):
                    visit(nested, (*path, equation_index, name))
        elif isinstance(value, dict):
            for name, nested in sorted(value.items()):
                visit(nested, (*path, name))
        elif isinstance(value, (tuple, list)):
            for index, nested in enumerate(value):
                visit(nested, (*path, index))

    visit(closed_jaxpr, ())
    return tuple(structures)


def _walk_ir_operations(value: Any):
    operation = getattr(value, "operation", value)
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for nested in block.operations:
                yield from _walk_ir_operations(nested)


def _stablehlo_operation_count(stablehlo: Any) -> int:
    return sum(operation.name.startswith("stablehlo.") for operation in _walk_ir_operations(stablehlo))


@pytest.mark.parametrize("num_devices", [1, 2], ids=("one-device", "two-device"))
@pytest.mark.parametrize("columns", [None, 2], ids=("rank-one", "multi-column"))
@pytest.mark.parametrize(
    "composition",
    ["jit", "jit-grad", "grad-jit", "value-and-grad", "remat"],
)
def test_nonlinear_loss_supported_compositions_match_dense(
    num_devices: int,
    columns: int | None,
    composition: str,
) -> None:
    block = _block()
    operator = _operator(block, num_devices=num_devices)
    theta, target, _ = _case(operator, columns)
    dense = _dense_matrix(block)
    expected_value, expected_gradient = jax.value_and_grad(_dense_loss, argnums=0)(theta, dense, target)

    if composition == "jit":
        actual = jax.jit(_research_loss)(theta, operator, target)
        np.testing.assert_allclose(actual, expected_value, rtol=1e-5, atol=1e-5)
    elif composition == "jit-grad":
        actual = jax.jit(jax.grad(_research_loss, argnums=0))(theta, operator, target)
        np.testing.assert_allclose(actual, expected_gradient, rtol=2e-5, atol=2e-5)
    elif composition == "grad-jit":
        actual = jax.grad(jax.jit(_research_loss), argnums=0)(theta, operator, target)
        np.testing.assert_allclose(actual, expected_gradient, rtol=2e-5, atol=2e-5)
    elif composition == "value-and-grad":
        actual_value, actual_gradient = jax.jit(jax.value_and_grad(_research_loss, argnums=0))(
            theta,
            operator,
            target,
        )
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-5)
    else:
        rematerialized = jax.remat(_research_loss)
        actual_value, actual_gradient = jax.jit(jax.value_and_grad(rematerialized, argnums=0))(
            theta,
            operator,
            target,
        )
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("num_devices", [1, 2], ids=("one-device", "two-device"))
@pytest.mark.parametrize("columns", [None, 2], ids=("rank-one", "multi-column"))
def test_nonlinear_loss_jvp_vjp_and_hvp_match_dense_and_finite_difference(
    num_devices: int,
    columns: int | None,
) -> None:
    block = _block()
    operator = _operator(block, num_devices=num_devices)
    theta, target, tangent = _case(operator, columns)
    dense = _dense_matrix(block)

    value, actual_jvp = jax.jit(_loss_jvp)(theta, operator, target, tangent)
    vjp_value, actual_vjp = jax.jit(_loss_vjp)(theta, operator, target)
    actual_hvp = jax.jit(_loss_hvp)(theta, operator, target, tangent)
    expected_value, expected_jvp = jax.jvp(lambda value: _dense_loss(value, dense, target), (theta,), (tangent,))
    expected_vjp = jax.grad(_dense_loss, argnums=0)(theta, dense, target)
    expected_hvp = _loss_hvp(theta, _DenseOperator(dense), target, tangent)
    epsilon = jnp.asarray(1e-3, dtype=theta.dtype)
    finite_difference = (
        _research_loss(theta + epsilon * tangent, operator, target)
        - _research_loss(theta - epsilon * tangent, operator, target)
    ) / (2 * epsilon)

    np.testing.assert_allclose(value, expected_value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(vjp_value, expected_value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_jvp, expected_jvp, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_vjp, expected_vjp, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_hvp, expected_hvp, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(actual_jvp, finite_difference, rtol=2e-2, atol=2e-3)


class _DenseOperator:
    """Test-only dense product adapter used solely for independent AD oracles."""

    def __init__(self, matrix: jax.Array):
        self.matrix = matrix

    def matmat(self, values: jax.Array) -> jax.Array:
        return self.matrix @ values


@pytest.mark.parametrize("columns", [None, 2], ids=("rank-one", "multi-column"))
def test_nonlinear_loss_vmap_scan_and_remat_keep_graph_invariant(columns: int | None) -> None:
    operator = _operator(_block(), num_devices=2)
    theta, target, _ = _case(operator, columns)
    theta_batch = jnp.stack((theta, theta + 0.1, theta - 0.2))
    target_batch = jnp.stack((target, target - 0.05, target + 0.15))
    expected = jnp.stack(
        [
            _research_loss(batch_theta, operator, batch_target)
            for batch_theta, batch_target in zip(theta_batch, target_batch)
        ]
    )

    vmapped = jax.jit(jax.vmap(_research_loss, in_axes=(0, None, 0)))(theta_batch, operator, target_batch)
    final_operator, scanned = jax.jit(_scan_losses)(operator, theta_batch, target_batch)
    rematerialized = jax.jit(jax.vmap(jax.remat(_research_loss), in_axes=(0, None, 0)))(
        theta_batch,
        operator,
        target_batch,
    )

    assert jax.typeof(final_operator.graph) == jax.typeof(operator.graph)
    np.testing.assert_allclose(vmapped, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(scanned, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rematerialized, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("transform", ["jit-grad", "grad-jit", "higher-order", "remat"])
def test_transformed_loss_ir_has_no_graph_constants_or_duplicate_graph_residuals(transform: str) -> None:
    operator = _operator(_block(), num_devices=2)
    theta, target, tangent = _case(operator, 2)
    if transform == "jit-grad":
        transformed = jax.jit(jax.grad(_research_loss, argnums=0))
        args = (theta, operator, target)
    elif transform == "grad-jit":
        transformed = jax.grad(jax.jit(_research_loss), argnums=0)
        args = (theta, operator, target)
    elif transform == "higher-order":
        transformed = jax.jit(_loss_hvp)
        args = (theta, operator, target, tangent)
    else:
        transformed = jax.jit(jax.grad(jax.remat(_research_loss), argnums=0))
        args = (theta, operator, target)

    closed_jaxpr = jax.make_jaxpr(transformed)(*args)
    graph_widths = _recursive_graph_widths(closed_jaxpr)

    assert _recursive_array_constant_bytes(closed_jaxpr) == 0
    assert graph_widths
    assert max(graph_widths) == 1


def test_dead_product_is_eliminated_without_graph_constants_or_residuals() -> None:
    operator = _operator(_block(), num_devices=2)
    theta, target, _ = _case(operator, 2)

    def regularization_only(dense, graph_operator, ignored_target):
        del ignored_target
        _ = graph_operator.matmat(dense)
        return 1e-3 * jnp.sum(dense**2)

    transformed = jax.jit(jax.grad(regularization_only, argnums=0))
    closed_jaxpr = jax.make_jaxpr(transformed)(theta, operator, target)
    stablehlo = str(transformed.lower(theta, operator, target).compiler_ir("stablehlo"))

    np.testing.assert_allclose(transformed(theta, operator, target), 2e-3 * theta, rtol=1e-6, atol=1e-6)
    assert _recursive_array_constant_bytes(closed_jaxpr) == 0
    assert "sdy.manual_computation" not in stablehlo


@pytest.mark.parametrize("transform", ["mapped-graph", "differentiate-graph"])
def test_graph_transform_failures_are_actionable(transform: str) -> None:
    operator = _operator(_block(), num_devices=2)
    theta, target, _ = _case(operator, 2)

    with pytest.raises(TypeError, match="opaque graph.*(invariant|differentiat)"):
        if transform == "mapped-graph":
            signature = hijax_adapter._signature_from_graph(operator.graph)

            def mapped_graph_loss(graph):
                prediction = hijax_adapter._bind_matmat_rank2(
                    graph,
                    theta,
                    signature=signature,
                    output_axes=(),
                )
                return jnp.mean((jnp.tanh(prediction) - target) ** 2)

            jax.vmap(
                mapped_graph_loss,
                in_axes=_PackedGraphMappingSpec(mapped=True),
                axis_size=2,
            )(operator.graph)
        else:
            jax.grad(lambda graph_operator: _research_loss(theta, graph_operator, target))(operator)


def test_fixed_capacity_source_block_count_preserves_transformed_program_structure() -> None:
    one_source = _operator(_block(), num_devices=2)
    two_sources = _operator(_block(), _empty_block(), num_devices=2)
    operators = (one_source, two_sources)
    cases = tuple(_case(operator, 2) for operator in operators)
    transformed = jax.jit(jax.grad(_research_loss, argnums=0))
    closed_jaxprs = tuple(
        jax.make_jaxpr(transformed)(theta, operator, target)
        for operator, (theta, target, _tangent) in zip(operators, cases, strict=True)
    )
    stablehlos = tuple(
        transformed.lower(theta, operator, target).compiler_ir("stablehlo")
        for operator, (theta, target, _tangent) in zip(operators, cases, strict=True)
    )

    assert one_source.capacities == two_sources.capacities
    assert jax.typeof(one_source.graph) == jax.typeof(two_sources.graph)
    assert len({_recursive_equation_structure(jaxpr) for jaxpr in closed_jaxprs}) == 1
    assert len({_stablehlo_operation_count(stablehlo) for stablehlo in stablehlos}) == 1
    assert all(_recursive_array_constant_bytes(jaxpr) == 0 for jaxpr in closed_jaxprs)
