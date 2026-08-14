# pattern: Mixed (unavoidable)
# Reason: Pure packed-algebra contract tests share fixtures with concrete JAX
# mesh, sharding, compilation, and device-residency integration tests.

from __future__ import annotations

import inspect

from dataclasses import fields, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.extend import core as jax_core
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy import sparse

from linear_dag.core.jaxlinarg import packed_products as packed_products_module
from linear_dag.core.jaxlinarg.ingress import (
    _packed_from_block_arrays,
    _PackedJaxLinearARG,
    from_block_arrays,
)
from linear_dag.core.jaxlinarg.packed_products import (
    _local_matmat_rank2,
    _local_rmatmat_rank2,
    lineararg_matmat,
    lineararg_rmatmat,
)
from linear_dag.core.jaxlinarg.packing import (
    BLOCK_DESCRIPTOR_FIELDS,
    LinearARGBlockArrays,
    PACKED_COMPONENT_NAMES,
    VALID_LENGTH_FIELDS,
)
from linear_dag.core.lineararg import LinearARG
from tests.jax.bench import test_parallel_benchmarks


def _graph_mesh(num_devices: int = 1) -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:num_devices]), ("graph",))


def _two_device_graph_mesh_or_skip() -> Mesh:
    if len(jax.devices("cpu")) < 2:
        pytest.skip("requires XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import")
    return _graph_mesh(2)


def _block_from_lineararg(linarg: LinearARG, *, dtype: Any = np.float32) -> LinearARGBlockArrays:
    linarg.calculate_nonunique_indices()
    return LinearARGBlockArrays(
        indptr=np.asarray(linarg.A.indptr, dtype=np.int32),
        indices=np.asarray(linarg.A.indices, dtype=np.int32),
        data=np.asarray(linarg.A.data, dtype=dtype),
        variant_indices=np.asarray(linarg.variant_indices, dtype=np.int32),
        flip=np.asarray(linarg.flip, dtype=np.bool_),
        sample_indices=np.asarray(linarg.sample_indices, dtype=np.int32),
        nonunique_indices=np.asarray(linarg.nonunique_indices, dtype=np.int32),
        allele_counts=np.asarray(linarg.allele_counts, dtype=np.int32),
        n_variants=linarg.shape[1],
        n_samples=linarg.shape[0],
    )


def _lineararg_from_block(block: LinearARGBlockArrays) -> LinearARG:
    n_nodes = block.indptr.size - 1
    linarg = LinearARG(
        A=sparse.csc_matrix(
            (np.asarray(block.data, dtype=np.int32), block.indices, block.indptr),
            shape=(n_nodes, n_nodes),
        ),
        variant_indices=np.asarray(block.variant_indices, dtype=np.int32),
        flip=np.asarray(block.flip, dtype=np.bool_),
        n_samples=np.int32(block.n_samples),
        nonunique_indices=np.asarray(block.nonunique_indices, dtype=np.int32),
    )
    if block.allele_counts is not None:
        linarg.set_allele_counts(np.asarray(block.allele_counts, dtype=np.int32))
    return linarg


def _repeated_variant_block(*, dtype: Any = np.float32) -> LinearARGBlockArrays:
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


def _larger_block(*, dtype: Any = np.float32) -> LinearARGBlockArrays:
    return LinearARGBlockArrays(
        indptr=np.asarray([0, 2, 4, 6, 6, 6, 6], dtype=np.int32),
        indices=np.asarray([4, 5, 3, 5, 3, 4], dtype=np.int32),
        data=np.asarray([1.0, 2.0, 1.0, 1.0, 2.0, 1.0], dtype=dtype),
        variant_indices=np.asarray([0, 1, 2], dtype=np.int32),
        flip=np.asarray([True, False, True], dtype=np.bool_),
        sample_indices=np.asarray([5, 4], dtype=np.int32),
        nonunique_indices=np.arange(6, dtype=np.int32),
        allele_counts=np.asarray([1, 1, 1], dtype=np.int32),
        n_variants=3,
        n_samples=2,
    )


def _empty_variant_block(*, dtype: Any = np.float32) -> LinearARGBlockArrays:
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


def _fixed_capacity_different_block_count_operators() -> tuple[_PackedJaxLinearARG, _PackedJaxLinearARG]:
    mesh = _two_device_graph_mesh_or_skip()
    one_block = _packed_from_block_arrays(
        (_repeated_variant_block(),),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    two_blocks = _packed_from_block_arrays(
        (_repeated_variant_block(), _empty_variant_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    return one_block, two_blocks


def _recursive_closed_jaxpr_constant_metrics(closed_jaxpr: jax_core.ClosedJaxpr) -> tuple[int, int]:
    array_constant_bytes = 0
    constvar_count = 0

    def visit(value: Any) -> None:
        nonlocal array_constant_bytes, constvar_count
        if isinstance(value, jax_core.ClosedJaxpr):
            for constant in value.consts:
                if isinstance(constant, (jax.Array, np.ndarray)):
                    array_constant_bytes += int(constant.size * constant.dtype.itemsize)
            visit(value.jaxpr)
        elif isinstance(value, jax_core.Jaxpr):
            constvar_count += len(value.constvars)
            for equation in value.eqns:
                visit(equation.params)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(closed_jaxpr)
    return array_constant_bytes, constvar_count


def _recursive_jaxpr_equation_structure(closed_jaxpr: jax_core.ClosedJaxpr) -> tuple[Any, ...]:
    structures: list[Any] = []

    def visit(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, jax_core.ClosedJaxpr):
            visit(value.jaxpr, path)
        elif isinstance(value, jax_core.Jaxpr):
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


def _ir_operation_names(stablehlo: Any) -> tuple[str, ...]:
    return tuple(operation.name for operation in _walk_ir_operations(stablehlo))


def _stablehlo_operation_count(stablehlo: Any) -> int:
    return sum(name.startswith("stablehlo.") for name in _ir_operation_names(stablehlo))


def _main_graph_operand_attributes(stablehlo: Any) -> tuple[str, ...]:
    main = next(
        operation
        for operation in stablehlo.body.operations
        if operation.operation.name == "func.func" and str(operation.attributes["sym_name"]) == '"main"'
    )
    argument_attributes = main.attributes["arg_attrs"]
    return tuple(str(attributes) for attributes in argument_attributes if "graph" in str(attributes))


def _collective_type_signatures(stablehlo: Any, name: str) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    return tuple(
        (
            tuple(str(operand.type) for operand in operation.operands),
            tuple(str(result.type) for result in operation.results),
        )
        for operation in _walk_ir_operations(stablehlo)
        if operation.name == name
    )


def _replace_carrier_array(
    operator: _PackedJaxLinearARG,
    name: str,
    values: np.ndarray,
) -> _PackedJaxLinearARG:
    components = list(operator.graph.components)
    component_index = PACKED_COMPONENT_NAMES.index(name)
    components[component_index] = jax.device_put(values, getattr(operator, name).sharding)
    return _PackedJaxLinearARG(
        n_samples=operator.n_samples,
        n_variants=operator.n_variants,
        capacities=operator.capacities,
        graph_mesh=operator.graph_mesh,
        graph=replace(operator.graph, components=tuple(components)),
    )


def test_local_and_single_device_products_match_oracle_and_exact_ragged(oracle_case) -> None:
    block = _block_from_lineararg(oracle_case.linarg)
    operator = _packed_from_block_arrays((block,), mesh=_graph_mesh()).operator
    exact = from_block_arrays(block)
    w_rank2 = jnp.asarray(oracle_case.w).reshape(operator.n_variants, -1)
    y_rank2 = jnp.asarray(oracle_case.y).reshape(operator.n_samples, -1)

    local_forward = _local_matmat_rank2(operator, w_rank2)
    local_reverse = _local_rmatmat_rank2(operator, y_rank2)

    np.testing.assert_allclose(np.asarray(local_forward), np.asarray(exact.matmat(w_rank2)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(local_reverse), np.asarray(exact.rmatmat(y_rank2)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(lineararg_matmat(operator, oracle_case.w)),
        oracle_case.Xw,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(lineararg_rmatmat(operator, oracle_case.y)),
        oracle_case.XTy,
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_products_preserve_logical_order_and_repeated_compressed_scatter() -> None:
    blocks = (_repeated_variant_block(), _larger_block())
    operator = _packed_from_block_arrays(blocks, mesh=_graph_mesh(), allow_excess_padding=True).operator
    descriptors = np.asarray(operator.block_descriptors)[0]
    logical_block_column = BLOCK_DESCRIPTOR_FIELDS.index("logical_block_index")
    w = np.arange(operator.n_variants * 3, dtype=np.float32).reshape(operator.n_variants, 3) / 10.0
    y = np.arange(operator.n_samples * 3, dtype=np.float32).reshape(operator.n_samples, 3) / 7.0
    cython_blocks = tuple(_lineararg_from_block(block) for block in blocks)
    exact_blocks = tuple(from_block_arrays(block) for block in blocks)
    offsets = np.cumsum([0, *(block.n_variants for block in blocks)])

    expected_forward_cython = sum(
        block @ w[start:stop] for block, start, stop in zip(cython_blocks, offsets[:-1], offsets[1:], strict=True)
    )
    expected_reverse_cython = np.concatenate([block.T @ y for block in cython_blocks], axis=0)
    expected_forward_exact = sum(
        block.matmat(w[start:stop]) for block, start, stop in zip(exact_blocks, offsets[:-1], offsets[1:], strict=True)
    )
    expected_reverse_exact = jnp.concatenate([block.rmatmat(y) for block in exact_blocks], axis=0)

    assert descriptors[0, logical_block_column] == 1
    np.testing.assert_allclose(np.asarray(lineararg_matmat(operator, w)), expected_forward_cython, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(lineararg_matmat(operator, w)), np.asarray(expected_forward_exact), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(lineararg_rmatmat(operator, y)), expected_reverse_cython, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(lineararg_rmatmat(operator, y)), np.asarray(expected_reverse_exact), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="requires JAX_ENABLE_X64=1")
def test_single_device_float64_products_preserve_dtype_and_numerics() -> None:
    blocks = (_repeated_variant_block(dtype=np.float64), _larger_block(dtype=np.float64))
    operator = _packed_from_block_arrays(blocks, mesh=_graph_mesh(), dtype=np.float64).operator
    exact_blocks = tuple(from_block_arrays(block, dtype=np.float64) for block in blocks)
    w = jnp.arange(operator.n_variants * 2, dtype=jnp.float64).reshape(operator.n_variants, 2) / 10.0
    y = jnp.arange(operator.n_samples * 2, dtype=jnp.float64).reshape(operator.n_samples, 2) / 7.0
    offsets = np.cumsum([0, *(block.n_variants for block in blocks)])

    actual_forward = lineararg_matmat(operator, w)
    actual_reverse = lineararg_rmatmat(operator, y)
    expected_forward = sum(
        block.matmat(w[start:stop]) for block, start, stop in zip(exact_blocks, offsets[:-1], offsets[1:], strict=True)
    )
    expected_reverse = jnp.concatenate([block.rmatmat(y) for block in exact_blocks], axis=0)

    assert actual_forward.dtype == jnp.float64
    assert actual_reverse.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(actual_forward), np.asarray(expected_forward), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(actual_reverse), np.asarray(expected_reverse), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("function", "shape", "match"),
    [
        (lineararg_matmat, (2, 2, 1), "rank 1 or 2"),
        (lineararg_matmat, (3, 1), "leading dimension 2"),
        (lineararg_rmatmat, (2, 2, 1), "rank 1 or 2"),
        (lineararg_rmatmat, (3, 1), "leading dimension 2"),
    ],
)
def test_invalid_operand_shape_fails_before_local_solve(function, shape, match) -> None:
    operator = _packed_from_block_arrays((_repeated_variant_block(),), mesh=_graph_mesh()).operator

    with pytest.raises(ValueError, match=match):
        function(operator, np.ones(shape, dtype=np.float32))


def test_invalid_operand_dtype_fails_before_local_solve() -> None:
    operator = _packed_from_block_arrays((_repeated_variant_block(),), mesh=_graph_mesh()).operator

    with pytest.raises((TypeError, ValueError), match="dtype|convert|not a valid"):
        lineararg_matmat(operator, np.asarray(["left", "right"]))


@pytest.mark.parametrize(
    ("field_name", "mutate", "match"),
    [
        (
            "block_descriptors",
            lambda values: values.at[
                0,
                0,
                BLOCK_DESCRIPTOR_FIELDS.index("edge_length"),
            ].set(np.iinfo(np.int32).max),
            "descriptor edge span",
        ),
        ("variant_indices", lambda values: values.at[0, 0].set(np.iinfo(np.int32).max), "variant indices"),
        ("logical_variant_indices", lambda values: values.at[0, 1].set(0), "logical variant mapping"),
    ],
)
def test_invalid_packed_graph_fails_at_carrier_boundary(field_name, mutate, match) -> None:
    operator = _packed_from_block_arrays((_repeated_variant_block(),), mesh=_graph_mesh()).operator
    values = mutate(jnp.asarray(getattr(operator, field_name)))

    with pytest.raises(ValueError, match=match):
        _replace_carrier_array(operator, field_name, np.asarray(values))


def test_packed_carrier_keeps_fixed_component_count() -> None:
    operator = _packed_from_block_arrays((_repeated_variant_block(),), mesh=_graph_mesh()).operator

    assert tuple(field.name for field in fields(operator) if not field.metadata.get("static")) == ("graph",)
    assert isinstance(operator.indptr.sharding, NamedSharding)
    assert operator.graph_mesh == operator.indptr.sharding.mesh
    assert len(jax.tree_util.tree_leaves(operator)) == 1


def test_shard_map_two_device_products_match_single_device_and_bound_methods() -> None:
    blocks = (_repeated_variant_block(), _larger_block(), _repeated_variant_block())
    mesh = _two_device_graph_mesh_or_skip()
    sharded = _packed_from_block_arrays(blocks, mesh=mesh, allow_excess_padding=True).operator
    single = _packed_from_block_arrays(blocks, mesh=_graph_mesh(), allow_excess_padding=True).operator
    w = jnp.arange(sharded.n_variants * 3, dtype=jnp.float32).reshape(sharded.n_variants, 3) / 10.0
    y = jnp.arange(sharded.n_samples * 3, dtype=jnp.float32).reshape(sharded.n_samples, 3) / 7.0

    expected_forward = lineararg_matmat(single, w)
    expected_reverse = lineararg_rmatmat(single, y)

    np.testing.assert_allclose(
        np.asarray(lineararg_matmat(sharded, w)),
        np.asarray(expected_forward),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(lineararg_rmatmat(sharded, y)), np.asarray(expected_reverse), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(np.asarray(sharded.matmat(w)), np.asarray(expected_forward), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(sharded.rmatmat(y)), np.asarray(expected_reverse), rtol=1e-5, atol=1e-5)


def test_shard_map_graph_fields_retain_one_assigned_local_shard() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block(), _repeated_variant_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    original_residency = {
        name: tuple((shard.device, shard.index) for shard in getattr(operator, name).addressable_shards)
        for name in PACKED_COMPONENT_NAMES
    }
    operator.matmat(jnp.ones((operator.n_variants, 1), dtype=jnp.float32)).block_until_ready()
    operator.rmatmat(jnp.ones((operator.n_samples, 1), dtype=jnp.float32)).block_until_ready()

    for name in PACKED_COMPONENT_NAMES:
        array = getattr(operator, name)
        assert len(array.addressable_shards) == 2
        assert tuple((shard.device, shard.index) for shard in array.addressable_shards) == original_residency[name]
        for shard in array.addressable_shards:
            assert shard.data.shape[0] == 1
            assert shard.data.devices() == {shard.device}


def test_shard_map_forward_uses_replicated_sum_collective() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    w = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)

    stablehlo = str(jax.jit(lineararg_matmat).lower(operator, w).compiler_ir("stablehlo"))

    assert "stablehlo.all_reduce" in stablehlo


def test_shard_map_sample_sharded_forward_uses_compatible_reduce_scatter() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    blocks = (_repeated_variant_block(), _larger_block())
    operator = _packed_from_block_arrays(blocks, mesh=mesh, allow_excess_padding=True).operator
    w = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    requested = NamedSharding(mesh, P("graph", None))
    expected = _packed_from_block_arrays(blocks, mesh=_graph_mesh(), allow_excess_padding=True).operator.matmat(w)

    actual = lineararg_matmat(operator, w, out_sharding=requested)
    stablehlo_module = (
        jax.jit(partial(lineararg_matmat, out_sharding=requested)).lower(operator, w).compiler_ir("stablehlo")
    )
    stablehlo = str(stablehlo_module)

    assert isinstance(actual.sharding, NamedSharding)
    assert actual.sharding.spec == P("graph")
    assert "stablehlo.reduce_scatter" in stablehlo
    assert "stablehlo.all_gather" not in stablehlo
    assert "stablehlo.collective_broadcast" not in stablehlo
    assert _collective_type_signatures(stablehlo_module, "stablehlo.reduce_scatter") == (
        (("tensor<2x2xf32>",), ("tensor<1x2xf32>",)),
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_empty_graph_assignment_is_inert_for_both_shard_map_products() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    block = _repeated_variant_block()
    operator = _packed_from_block_arrays((block,), mesh=mesh, allow_excess_padding=True).operator
    exact = from_block_arrays(block)
    w = jnp.arange(operator.n_variants * 2, dtype=jnp.float32).reshape(operator.n_variants, 2) / 10.0
    y = jnp.arange(operator.n_samples * 2, dtype=jnp.float32).reshape(operator.n_samples, 2) / 7.0

    np.testing.assert_allclose(np.asarray(operator.matmat(w)), np.asarray(exact.matmat(w)), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(operator.rmatmat(y)), np.asarray(exact.rmatmat(y)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method_name", ["compile_matmat", "compile_rmatmat"])
def test_compile_helper_passes_carrier_to_module_level_jit(method_name: str) -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    values = (
        jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
        if method_name == "compile_matmat"
        else jnp.ones((operator.n_samples, 2), dtype=jnp.float32)
    )
    expected = operator.matmat(values) if method_name == "compile_matmat" else operator.rmatmat(values)

    compiled = getattr(operator, method_name)()
    actual = compiled(values)
    lowered = compiled.lower(values)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert lowered.compiler_ir("stablehlo") is not None
    source = inspect.getsource(type(compiled).__call__)
    assert "self.operator, values" in source


def test_compile_and_bound_method_docstrings_disclaim_raw_closure_capture() -> None:
    for name in ("matmat", "rmatmat", "compile_matmat", "compile_rmatmat"):
        docstring = inspect.getdoc(getattr(_PackedJaxLinearARG, name)) or ""
        assert "closure" in docstring.lower()
        assert "memory guarantee" in docstring.lower()


def test_mesh_axis_validation_rejects_wrong_name_before_product_lowering() -> None:
    wrong_mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("devices",))

    with pytest.raises(ValueError, match="graph"):
        _packed_from_block_arrays((_repeated_variant_block(),), mesh=wrong_mesh)


def test_mesh_output_sharding_rejects_wrong_axis_before_product_lowering() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    incompatible = NamedSharding(mesh, P(None, "graph"))

    with pytest.raises(ValueError, match="sample.*leading axis"):
        lineararg_matmat(operator, values, out_sharding=incompatible)


@pytest.mark.parametrize(
    ("operand_columns", "spec"),
    [
        (None, P(None, None)),
        (None, P("graph", None)),
        (2, P(None, None, None)),
        (2, P("graph", None, None)),
    ],
)
@pytest.mark.parametrize("stage", ["eager", "outer_jit_lower", "outer_jit_execute"])
def test_forward_output_sharding_rejects_specs_above_logical_result_rank(
    operand_columns: int | None,
    spec: P,
    stage: str,
) -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    shape = (operator.n_variants,) if operand_columns is None else (operator.n_variants, operand_columns)
    values = jnp.ones(shape, dtype=jnp.float32)
    requested = NamedSharding(mesh, spec)

    with pytest.raises(ValueError, match="logical result rank"):
        if stage == "eager":
            lineararg_matmat(operator, values, out_sharding=requested)
        else:
            compiled = jax.jit(partial(lineararg_matmat, out_sharding=requested))
            if stage == "outer_jit_lower":
                compiled.lower(operator, values)
            else:
                compiled(operator, values)


def test_lineararg_matmat_docstring_documents_output_sharding_contract() -> None:
    docstring = inspect.getdoc(lineararg_matmat) or ""
    contract = " ".join(docstring.split())

    assert "defaults to a result replicated" in contract
    assert "rank-1 result" in contract
    assert "rank-2 result" in contract
    assert "carrier graph mesh" in contract
    assert "divisible" in contract
    assert "ValueError" in contract


def test_mesh_output_sharding_rejects_different_graph_mesh_before_product_lowering() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    reversed_mesh = Mesh(np.asarray(mesh.devices).reshape(-1)[::-1], ("graph",))
    incompatible = NamedSharding(reversed_mesh, P("graph", None))

    with pytest.raises(ValueError, match="carrier graph mesh"):
        lineararg_matmat(
            operator,
            jnp.ones((operator.n_variants, 2), dtype=jnp.float32),
            out_sharding=incompatible,
        )


@pytest.mark.parametrize("stage", ["lower", "execute"])
def test_outer_jit_rejects_different_graph_mesh_for_requested_output(stage: str) -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    reversed_mesh = Mesh(np.asarray(mesh.devices).reshape(-1)[::-1], ("graph",))
    incompatible = NamedSharding(reversed_mesh, P("graph", None))
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    compiled = jax.jit(partial(lineararg_matmat, out_sharding=incompatible))

    with pytest.raises(ValueError, match="carrier graph mesh"):
        if stage == "lower":
            compiled.lower(operator, values)
        else:
            compiled(operator, values)


def test_outer_jit_output_sharding_keeps_graph_arrays_as_zero_constant_operands() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(), _larger_block()),
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    compatible = NamedSharding(mesh, P("graph", None))
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    compiled = jax.jit(partial(lineararg_matmat, out_sharding=compatible))

    closed_jaxpr = jax.make_jaxpr(compiled)(operator, values)

    assert _recursive_closed_jaxpr_constant_metrics(closed_jaxpr) == (0, 0)


@pytest.mark.parametrize("boundary", ["construction", "compilation"])
def test_non_single_host_graph_mesh_is_rejected_at_project_boundary(monkeypatch, boundary: str) -> None:
    mesh = _two_device_graph_mesh_or_skip()
    blocks = (_repeated_variant_block(), _larger_block())
    operator = None
    if boundary == "compilation":
        operator = _packed_from_block_arrays(blocks, mesh=mesh, allow_excess_padding=True).operator
    monkeypatch.setattr(
        packed_products_module,
        "_addressable_device_count",
        lambda _sharding: mesh.size - 1,
    )

    with pytest.raises(ValueError, match="every graph mesh device.*one host"):
        if boundary == "construction":
            _packed_from_block_arrays(blocks, mesh=mesh, allow_excess_padding=True)
        else:
            assert operator is not None
            operator.compile_matmat()


def test_shard_map_reduce_scatter_rejects_indivisible_sample_dimension() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    block = LinearARGBlockArrays(
        indptr=np.asarray([0, 3, 3, 3, 3], dtype=np.int32),
        indices=np.asarray([1, 2, 3], dtype=np.int32),
        data=np.ones(3, dtype=np.float32),
        variant_indices=np.asarray([0], dtype=np.int32),
        flip=np.asarray([False], dtype=np.bool_),
        sample_indices=np.asarray([3, 2, 1], dtype=np.int32),
        nonunique_indices=np.arange(4, dtype=np.int32),
        allele_counts=np.asarray([3], dtype=np.int32),
        n_variants=1,
        n_samples=3,
    )
    operator = _packed_from_block_arrays((block,), mesh=mesh, allow_excess_padding=True).operator
    requested = NamedSharding(mesh, P("graph", None))

    with pytest.raises(ValueError, match="divisible"):
        lineararg_matmat(operator, jnp.ones((1, 1), dtype=jnp.float32), out_sharding=requested)


@pytest.mark.parametrize(
    ("function", "compile_method", "leading_dimension"),
    [
        (lineararg_matmat, "compile_matmat", "n_variants"),
        (lineararg_rmatmat, "compile_rmatmat", "n_samples"),
    ],
)
def test_explicit_and_safe_product_jaxprs_have_no_array_constants(
    function,
    compile_method: str,
    leading_dimension: str,
) -> None:
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(),),
        mesh=_two_device_graph_mesh_or_skip(),
        allow_excess_padding=True,
    ).operator
    values = jnp.ones((getattr(operator, leading_dimension), 2), dtype=jnp.float32)
    compiled = getattr(operator, compile_method)()

    explicit_jaxpr = jax.make_jaxpr(function)(operator, values)
    safe_jaxpr = jax.make_jaxpr(compiled.compiled_function)(compiled.operator, values)

    assert _recursive_closed_jaxpr_constant_metrics(explicit_jaxpr) == (0, 0)
    assert _recursive_closed_jaxpr_constant_metrics(safe_jaxpr) == (0, 0)


def test_closed_over_product_diagnostic_exposes_graph_constants() -> None:
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(),),
        mesh=_two_device_graph_mesh_or_skip(),
        allow_excess_padding=True,
    ).operator
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    expected_bytes = sum(int(getattr(operator, name).nbytes) for name in PACKED_COMPONENT_NAMES)

    closed_jaxpr = jax.make_jaxpr(lambda operand: operator.matmat(operand))(values)
    constant_bytes, constvar_count = _recursive_closed_jaxpr_constant_metrics(closed_jaxpr)

    assert constant_bytes == expected_bytes
    assert constvar_count == len(PACKED_COMPONENT_NAMES)


@pytest.mark.parametrize(
    ("function", "leading_dimension"),
    [
        (lineararg_matmat, "n_variants"),
        (lineararg_rmatmat, "n_samples"),
    ],
)
def test_fixed_capacity_source_block_count_preserves_jaxpr_and_stablehlo_structure(
    function,
    leading_dimension: str,
) -> None:
    operators = _fixed_capacity_different_block_count_operators()
    values = tuple(jnp.ones((getattr(operator, leading_dimension), 2), dtype=jnp.float32) for operator in operators)
    closed_jaxprs = tuple(
        jax.make_jaxpr(function)(operator, operand) for operator, operand in zip(operators, values, strict=True)
    )
    stablehlos = tuple(
        jax.jit(function).lower(operator, operand).compiler_ir("stablehlo")
        for operator, operand in zip(operators, values, strict=True)
    )

    assert operators[0].capacities == operators[1].capacities
    assert operators[0].block_descriptors.shape == operators[1].block_descriptors.shape
    assert jax.tree_util.tree_structure(operators[0]) == jax.tree_util.tree_structure(operators[1])
    block_count_column = VALID_LENGTH_FIELDS.index("block_descriptors")
    assert tuple(int(np.asarray(operator.valid_lengths)[:, block_count_column].sum()) for operator in operators) == (
        1,
        2,
    )
    assert len({_recursive_jaxpr_equation_structure(jaxpr) for jaxpr in closed_jaxprs}) == 1
    assert len({_stablehlo_operation_count(stablehlo) for stablehlo in stablehlos}) == 1
    assert len({_main_graph_operand_attributes(stablehlo) for stablehlo in stablehlos}) == 1
    assert len({_ir_operation_names(stablehlo).count("stablehlo.case") for stablehlo in stablehlos}) == 1


@pytest.mark.parametrize(
    ("function", "leading_dimension", "collective"),
    [
        (lineararg_matmat, "n_variants", "stablehlo.all_reduce"),
        (lineararg_rmatmat, "n_samples", "stablehlo.all_reduce"),
    ],
)
def test_lowered_ir_preserves_graph_sharding_and_collects_only_dense_results(
    function,
    leading_dimension: str,
    collective: str,
) -> None:
    operator = _packed_from_block_arrays(
        (_repeated_variant_block(),),
        mesh=_two_device_graph_mesh_or_skip(),
        allow_excess_padding=True,
    ).operator
    values = jnp.ones((getattr(operator, leading_dimension), 2), dtype=jnp.float32)
    stablehlo = jax.jit(function).lower(operator, values).compiler_ir("stablehlo")
    names = _ir_operation_names(stablehlo)
    graph_operand_attributes = _main_graph_operand_attributes(stablehlo)

    assert len(graph_operand_attributes) == len(PACKED_COMPONENT_NAMES) - 1
    assert all("sdy.sharding" in attributes and '"graph"' in attributes for attributes in graph_operand_attributes)
    assert "sdy.manual_computation" in names
    assert "stablehlo.broadcast_in_dim" in names
    assert "stablehlo.all_gather" not in names
    assert "stablehlo.collective_broadcast" not in names
    assert _collective_type_signatures(stablehlo, collective) == ((("tensor<2x2xf32>",), ("tensor<2x2xf32>",)),)


def test_packed_benchmark_contract_includes_ir_metric_columns() -> None:
    result_fields = {field.name for field in fields(test_parallel_benchmarks.ParallelBenchmarkResult)}
    assert {
        "graph_constant_bytes",
        "graph_operand_count",
        "stablehlo_operation_count",
    } <= result_fields

    packed = _packed_from_block_arrays(
        (_repeated_variant_block(),),
        mesh=_two_device_graph_mesh_or_skip(),
        allow_excess_padding=True,
    )
    operator = packed.operator
    constant_bytes, operand_count, operation_count = test_parallel_benchmarks._packed_ir_metrics(operator)
    values = jax.ShapeDtypeStruct((operator.n_variants, 1), operator.data.dtype)
    closed_jaxpr = jax.make_jaxpr(lineararg_matmat)(operator, values)
    stablehlo = jax.jit(lineararg_matmat).lower(operator, values).compiler_ir("stablehlo")
    result = test_parallel_benchmarks._packed_memory_result(
        packed.diagnostics,
        operator="packed",
        construction_seconds=0.0,
        resident_devices_valid=True,
        graph_constant_bytes=constant_bytes,
        graph_operand_count=operand_count,
        stablehlo_operation_count=operation_count,
    )
    table = test_parallel_benchmarks._format_results_table([result])

    assert constant_bytes == 0
    assert operand_count == len(PACKED_COMPONENT_NAMES) - 1
    assert (constant_bytes, operand_count, operation_count) == (
        _recursive_closed_jaxpr_constant_metrics(closed_jaxpr)[0],
        len(_main_graph_operand_attributes(stablehlo)),
        _stablehlo_operation_count(stablehlo),
    )
    assert "graph constant bytes" in table
    assert "graph operands" in table
    assert "StableHLO ops" in table
