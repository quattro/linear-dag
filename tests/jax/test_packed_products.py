# pattern: Mixed (unavoidable)
# Reason: Pure packed-algebra contract tests share fixtures with concrete JAX
# mesh, sharding, compilation, and device-residency integration tests.

from __future__ import annotations

import inspect

from dataclasses import fields
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from scipy import sparse

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
)
from linear_dag.core.lineararg import LinearARG


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


def _replace_carrier_array(
    operator: _PackedJaxLinearARG,
    name: str,
    values: np.ndarray,
) -> _PackedJaxLinearARG:
    arrays = {
        field.name: getattr(operator, field.name) for field in fields(operator) if not field.metadata.get("static")
    }
    arrays[name] = jax.device_put(values, getattr(operator, name).sharding)
    return _PackedJaxLinearARG(
        n_samples=operator.n_samples,
        n_variants=operator.n_variants,
        capacities=operator.capacities,
        **arrays,
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

    assert tuple(field.name for field in fields(operator) if not field.metadata.get("static")) == PACKED_COMPONENT_NAMES


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

    for name in PACKED_COMPONENT_NAMES:
        array = getattr(operator, name)
        assert len(array.addressable_shards) == 2
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
    stablehlo = str(
        jax.jit(partial(lineararg_matmat, out_sharding=requested)).lower(operator, w).compiler_ir("stablehlo")
    )

    assert isinstance(actual.sharding, NamedSharding)
    assert actual.sharding.spec == P("graph")
    assert "stablehlo.reduce_scatter" in stablehlo
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
