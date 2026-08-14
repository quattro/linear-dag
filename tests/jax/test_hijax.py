# pattern: Mixed (unavoidable)
# Reason: Metadata contract tests read repository configuration and assert its
# pure supported-runtime policy before the private HiJAX adapter is introduced.

from __future__ import annotations

import tomllib

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import linear_dag
import linear_dag.core
import linear_dag.core.jaxlinarg

from linear_dag.core.jaxlinarg._hijax import (
    _graph_pspec_for_type,
    _packed_graph_component,
    _PackedGraphMappingSpec,
    _PackedGraphType,
    _PackedGraphValue,
    _PackedGraphZeroPspec,
    _PackedGraphZeroType,
    _PackedGraphZeroValue,
)
from linear_dag.core.jaxlinarg.ingress import _packed_from_block_arrays
from linear_dag.core.jaxlinarg.packing import LinearARGBlockArrays, PACKED_COMPONENT_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _graph_mesh(num_devices: int = 1) -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:num_devices]), ("graph",))


def _two_device_graph_mesh_or_skip() -> Mesh:
    if len(jax.devices("cpu")) < 2:
        pytest.skip("requires XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import")
    return _graph_mesh(2)


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


def _empty_block() -> LinearARGBlockArrays:
    return LinearARGBlockArrays(
        indptr=np.asarray([0, 0, 0], dtype=np.int32),
        indices=np.asarray([], dtype=np.int32),
        data=np.asarray([], dtype=np.float32),
        variant_indices=np.asarray([], dtype=np.int32),
        flip=np.asarray([], dtype=np.bool_),
        sample_indices=np.asarray([1, 0], dtype=np.int32),
        nonunique_indices=np.asarray([0, 1], dtype=np.int32),
        allele_counts=np.asarray([], dtype=np.int32),
        n_variants=0,
        n_samples=2,
    )


def _operator(*blocks: LinearARGBlockArrays, mesh: Mesh | None = None, dtype: Any = None):
    return _packed_from_block_arrays(
        blocks,
        mesh=_graph_mesh() if mesh is None else mesh,
        dtype=dtype,
        allow_excess_padding=True,
    ).operator


def test_supported_runtime_metadata_and_lock_are_consistent() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]
    build_requires = pyproject["build-system"]["requires"]
    runtime_requires = project["dependencies"]
    supported_python = SpecifierSet(project["requires-python"])

    assert Version("3.11") not in supported_python
    assert all(Version(version) in supported_python for version in ("3.12", "3.13", "3.14"))
    assert Version("3.15") not in supported_python
    assert "Programming Language :: Python :: 3.11" not in project["classifiers"]
    assert pyproject["tool"]["hatch"]["envs"]["all"]["matrix"][0]["python"] == ["3.12", "3.13", "3.14"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py312"

    for requirements in (build_requires, runtime_requires):
        assert "jax==0.11.0" in requirements
        assert "jaxlib==0.11.0" in requirements
        assert "numpy>=2.1" in requirements
        assert "scipy>=1.15" in requirements

    ignored_paths = (_REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "uv.lock" not in ignored_paths

    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
    assert lock["requires-python"] == ">=3.12, <3.15"
    resolved_versions = {package["name"]: package["version"] for package in lock["package"] if "version" in package}
    assert resolved_versions["jax"] == "0.11.0"
    assert resolved_versions["jaxlib"] == "0.11.0"


def test_packed_carrier_has_one_opaque_high_level_graph_leaf() -> None:
    operator = _operator(_block())

    leaves = jax.tree.leaves(operator)
    graph_type = jax.typeof(operator.graph)

    assert leaves == [operator.graph]
    assert isinstance(operator.graph, _PackedGraphValue)
    assert isinstance(graph_type, _PackedGraphType)
    assert len(graph_type.lo_ty()) == len(PACKED_COMPONENT_NAMES)
    assert not hasattr(operator.graph, "source_blocks")
    with pytest.raises(FrozenInstanceError):
        operator.graph.components = ()


def test_packed_graph_lower_raise_round_trip_preserves_order_and_sharding() -> None:
    graph = _operator(_block()).graph
    graph_type = jax.typeof(graph)

    lowered = graph_type.lower_val(graph)
    raised = graph_type.raise_val(*lowered)

    assert tuple(lowered) == graph.components
    assert jax.typeof(raised) == graph_type
    for expected, actual in zip(graph.components, raised.components, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.sharding == expected.sharding


def test_packed_graph_type_identity_tracks_layout_not_source_block_count() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    one_source_type = jax.typeof(_operator(_block(), mesh=mesh).graph)
    two_source_type = jax.typeof(_operator(_block(), _empty_block(), mesh=mesh).graph)
    float16_type = jax.typeof(_operator(_block(dtype=np.float16), mesh=mesh, dtype=np.float16).graph)

    assert one_source_type == two_source_type
    assert hash(one_source_type) == hash(two_source_type)
    assert one_source_type != float16_type

    changed_shape = replace(
        one_source_type,
        metadata=replace(one_source_type.metadata, n_samples=one_source_type.metadata.n_samples + 1),
    )
    changed_capacity_aval = one_source_type.component_types[0].update(
        shape=(one_source_type.component_types[0].shape[0], one_source_type.component_types[0].shape[1] + 1)
    )
    changed_capacity = replace(
        one_source_type,
        component_types=(changed_capacity_aval, *one_source_type.component_types[1:]),
    )
    explicit_mesh = mesh.abstract_mesh.update_axis_types({"graph": AxisType.Explicit})
    changed_component_sharding = NamedSharding(explicit_mesh, P("graph", None))
    changed_sharding_aval = one_source_type.component_types[0].update(sharding=changed_component_sharding)
    changed_sharding = replace(
        one_source_type,
        component_types=(changed_sharding_aval, *one_source_type.component_types[1:]),
    )

    assert one_source_type != changed_shape
    assert one_source_type != changed_capacity
    assert one_source_type != changed_sharding
    assert len({one_source_type, changed_shape, changed_capacity, changed_sharding}) == 4


def test_graph_mapping_and_sharding_specs_enforce_invariant_graph_axis() -> None:
    mesh = _two_device_graph_mesh_or_skip()
    graph_type = jax.typeof(_operator(_block(), mesh=mesh).graph)
    invariant = _PackedGraphMappingSpec(mapped=False)
    mapped = _PackedGraphMappingSpec(mapped=True)
    graph_spec = _graph_pspec_for_type(graph_type)

    assert graph_type.dec_rank(None, invariant) == graph_type
    assert graph_type.inc_rank(None, invariant) == graph_type
    assert graph_type.leading_axis_spec() == mapped
    with pytest.raises(TypeError, match="opaque graph.*invariant"):
        graph_type.dec_rank(2, mapped)
    with pytest.raises(TypeError, match="opaque graph.*invariant"):
        graph_type.inc_rank(2, mapped)

    lowered_specs = graph_spec.to_lo()
    assert len(lowered_specs) == len(graph_type.component_types)
    assert all(spec[0] == "graph" for spec in lowered_specs)
    assert isinstance(graph_spec.to_tangent_spec(), _PackedGraphZeroPspec)
    assert isinstance(graph_spec.to_ct_spec(), _PackedGraphZeroPspec)
    local_type = graph_type.shard(mesh, frozenset({"graph"}), True, graph_spec)
    restored_type = local_type.unshard(mesh, True, graph_spec)
    assert all(component_type.shape[0] == 1 for component_type in local_type.component_types)
    assert tuple(component_type.shape for component_type in restored_type.component_types) == tuple(
        component_type.shape for component_type in graph_type.component_types
    )
    assert tuple(component_type.dtype for component_type in restored_type.component_types) == tuple(
        component_type.dtype for component_type in graph_type.component_types
    )
    assert restored_type.metadata == graph_type.metadata

    graph = _operator(_block(), mesh=mesh).graph
    sharded_identity = jax.jit(
        jax.shard_map(
            lambda value: value,
            mesh=mesh,
            in_specs=graph_spec,
            out_specs=graph_spec,
            axis_names={"graph"},
            check_vma=True,
        )
    )
    identity_result = sharded_identity(graph)
    assert jax.typeof(identity_result) == graph_type
    for expected, actual in zip(graph.components, identity_result.components, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    wrong_mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("wrong",))
    with pytest.raises(ValueError, match='mesh.*"graph"'):
        graph_type.shard(wrong_mesh, frozenset({"wrong"}), True, graph_spec)


def test_graph_zero_contract_has_no_array_payload_and_adds_inertly() -> None:
    graph_type = jax.typeof(_operator(_block()).graph)
    zero_type = graph_type.to_tangent_aval()
    zero = zero_type.vspace_zero()

    assert isinstance(zero_type, _PackedGraphZeroType)
    assert isinstance(zero, _PackedGraphZeroValue)
    assert zero_type.lo_ty() == []
    assert zero_type.lower_val(zero) == []
    assert zero_type.raise_val() == zero
    assert zero_type.vspace_add(zero, zero) == zero
    assert jax.typeof(zero) == zero_type

    with pytest.raises(TypeError, match="opaque graph.*differentiat"):
        jax.jvp(
            lambda graph: _packed_graph_component(graph, 0),
            (operator_graph := _operator(_block()).graph,),
            (operator_graph,),
        )


def test_public_exports_and_annotations_do_not_expose_hijax() -> None:
    forbidden = ("_hijax", "HiType", "HiPspec", "MappingSpec", "Primitive")
    modules = (linear_dag, linear_dag.core, linear_dag.core.jaxlinarg)

    for module in modules:
        exported_names = tuple(getattr(module, "__all__", ()))
        annotations = tuple(str(annotation) for annotation in getattr(module, "__annotations__", {}).values())
        public_values = tuple(name for name in vars(module) if not name.startswith("_"))
        visible = (*exported_names, *annotations, *public_values)
        assert not any(token in item for token in forbidden for item in visible)
