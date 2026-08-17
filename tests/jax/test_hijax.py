# pattern: Mixed (unavoidable)
# Reason: Metadata contract tests read repository configuration and assert its
# pure supported-runtime policy before the private HiJAX adapter is introduced.

from __future__ import annotations

import inspect
import subprocess
import sys
import tomllib

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax._src import ad_util, core
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import linear_dag
import linear_dag.core
import linear_dag.core.jaxlinarg

from linear_dag.core.jaxlinarg import _hijax as hijax_adapter
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
from linear_dag.core.jaxlinarg.kernels import ffi_cpu
from linear_dag.core.jaxlinarg.packing import LinearARGBlockArrays, PACKED_COMPONENT_NAMES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_HIJAX_TOKENS = ("_hijax", "HiType", "HiPspec", "MappingSpec", "Primitive")


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


def _operator(
    *blocks: LinearARGBlockArrays,
    mesh: Mesh | None = None,
    dtype: Any = None,
    backend: linear_dag.Backend = linear_dag.Backend.PURE_JAX,
):
    return _packed_from_block_arrays(
        blocks,
        mesh=_graph_mesh() if mesh is None else mesh,
        dtype=dtype,
        backend=backend,
        allow_excess_padding=True,
    ).operator


def _assert_hijax_free(label: str, value: Any) -> None:
    text = str(value)
    leaked_token = next((token for token in _FORBIDDEN_HIJAX_TOKENS if token in text), None)
    assert leaked_token is None, f"{label} exposed forbidden HiJAX token {leaked_token}: {text}"


def _inspectable_callable(descriptor: Any) -> Any | None:
    if isinstance(descriptor, (classmethod, staticmethod)):
        return descriptor.__func__
    if isinstance(descriptor, property):
        return descriptor.fget
    return descriptor if callable(descriptor) else None


def _assert_callable_is_hijax_free(label: str, value: Any) -> None:
    for surface_name, surface in (
        ("module", getattr(value, "__module__", "")),
        ("qualname", getattr(value, "__qualname__", "")),
        ("annotations", inspect.get_annotations(value, eval_str=False)),
        ("docstring", inspect.getdoc(value) or ""),
    ):
        _assert_hijax_free(f"{label} {surface_name}", surface)
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        pass
    else:
        _assert_hijax_free(f"{label} signature", signature)
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        pass
    else:
        _assert_hijax_free(f"{label} source", source)


def _assert_public_module_is_hijax_free(module: ModuleType) -> None:
    declared_exports = getattr(module, "__all__", None)
    names = (
        tuple(declared_exports)
        if declared_exports is not None
        else tuple(name for name in vars(module) if not name.startswith("_"))
    )
    assert names, f"{module.__name__} has no inspectable public objects"
    _assert_hijax_free(f"{module.__name__} annotations", getattr(module, "__annotations__", {}))

    for name in names:
        value = getattr(module, name)
        _assert_hijax_free(f"{module.__name__}.{name} name", name)
        _assert_hijax_free(
            f"{module.__name__}.{name} type",
            f"{type(value).__module__}.{type(value).__qualname__}",
        )
        if not callable(value):
            continue
        _assert_callable_is_hijax_free(f"{module.__name__}.{name}", value)
        if inspect.isclass(value):
            for method_name, descriptor in vars(value).items():
                if method_name.startswith("_"):
                    continue
                method = _inspectable_callable(descriptor)
                if method is not None:
                    _assert_callable_is_hijax_free(f"{module.__name__}.{name}.{method_name}", method)


def _assert_pytree_leaves_are_hijax_free(value: Any) -> None:
    leaves = jax.tree.leaves(value)
    assert leaves, f"{type(value).__qualname__} unexpectedly has no PyTree leaves"
    for index, leaf in enumerate(leaves):
        _assert_hijax_free(
            f"{type(value).__qualname__} PyTree leaf {index}",
            (
                f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                repr(jax.typeof(leaf)),
            ),
        )


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
    modules = (linear_dag, linear_dag.core, linear_dag.core.jaxlinarg)

    for module in modules:
        _assert_public_module_is_hijax_free(module)


def test_public_isolation_audit_detects_leaked_annotation_without_dunder_all() -> None:
    synthetic_module = ModuleType("synthetic_public_module")

    def leaked(graph: Any) -> None:
        del graph

    leaked.__module__ = synthetic_module.__name__
    leaked.__annotations__["graph"] = "HiType"
    setattr(synthetic_module, "leaked", leaked)

    assert not hasattr(synthetic_module, "__all__")
    with pytest.raises(AssertionError, match="HiType"):
        _assert_public_module_is_hijax_free(synthetic_module)


def test_exact_public_jax_operator_pytrees_do_not_expose_hijax() -> None:
    arrays = _block()
    block = linear_dag.JaxLinearARG.from_lineararg_arrays(
        indptr=arrays.indptr,
        indices=arrays.indices,
        data=arrays.data,
        variant_indices=arrays.variant_indices,
        flip=arrays.flip,
        sample_indices=arrays.sample_indices,
        nonunique_indices=arrays.nonunique_indices,
        allele_counts=arrays.allele_counts,
        n_variants=arrays.n_variants,
        n_samples=arrays.n_samples,
        backend=linear_dag.Backend.PURE_JAX,
    )
    parallel = linear_dag.JaxParallelOperator(
        blocks=(block,),
        variant_offsets=(0, block.n_variants),
        mesh=Mesh(np.asarray(jax.devices("cpu")[:1]), ("blocks",)),
        block_ranges=((0, 1),),
        backend=linear_dag.Backend.PURE_JAX,
    )
    grm = linear_dag.JaxGRMOperator(block)

    assert type(block) is linear_dag.JaxLinearARG
    assert type(parallel) is linear_dag.JaxParallelOperator
    assert type(grm) is linear_dag.JaxGRMOperator
    for operator in (block, parallel, grm):
        _assert_pytree_leaves_are_hijax_free(operator)


def test_missing_hijax_module_failure_is_actionable_and_chained_in_isolated_import() -> None:
    script = """
import importlib.abc
import sys

class BlockHiJAX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        del path, target
        if fullname == "jax.experimental.hijax":
            raise ModuleNotFoundError("blocked whole HiJAX module")
        return None

sys.meta_path.insert(0, BlockHiJAX())
try:
    import linear_dag.core.jaxlinarg._hijax
except ImportError as error:
    if error.__cause__ is None:
        raise SystemExit("missing chained import cause")
    print(error)
    print(f"cause={type(error.__cause__).__name__}: {error.__cause__}")
else:
    raise SystemExit("missing HiJAX module was accepted")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "JAX/JAXlib 0.11.0" in completed.stdout
    assert "jax.experimental.hijax" in completed.stdout
    assert "cause=ModuleNotFoundError: blocked whole HiJAX module" in completed.stdout


def test_hijax_compatibility_failure_is_actionable_in_isolated_import() -> None:
    script = """
import jax.experimental.hijax as hijax

delattr(hijax, "jvp_from_lin")
try:
    import linear_dag.core.jaxlinarg._hijax
except ImportError as error:
    print(error)
else:
    raise SystemExit("missing HiJAX symbol was accepted")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "JAX/JAXlib 0.11.0" in completed.stdout
    assert "jvp_from_lin" in completed.stdout


@pytest.mark.parametrize(
    ("mutation", "qualified_name"),
    [
        ("hijax.HiPrimitive.bind = lambda self, arg: None", "HiPrimitive.bind"),
        ("hijax.VJPHiPrimitive.__call__ = lambda self, arg: None", "VJPHiPrimitive.__call__"),
    ],
)
def test_hijax_compatibility_rejects_changed_binder_signatures_in_isolated_import(
    mutation: str,
    qualified_name: str,
) -> None:
    script = f"""
import jax.experimental.hijax as hijax

{mutation}
try:
    import linear_dag.core.jaxlinarg._hijax
except ImportError as error:
    print(error)
else:
    raise SystemExit("changed HiJAX binder signature was accepted")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "JAX/JAXlib 0.11.0" in completed.stdout
    assert qualified_name in completed.stdout


@pytest.mark.parametrize(
    ("method", "companion", "leading_dimension"),
    [
        ("matmat", "rmatmat", "n_variants"),
        ("rmatmat", "matmat", "n_samples"),
    ],
)
def test_packed_product_jvp_and_vjp_use_paired_products(
    method: str,
    companion: str,
    leading_dimension: str,
) -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.arange(getattr(operator, leading_dimension) * 2, dtype=jnp.float32).reshape(-1, 2) / 5
    tangent = jnp.flip(values, axis=0) + 0.25

    primal, primal_tangent = jax.jvp(
        lambda operand: getattr(operator, method)(operand),
        (values,),
        (tangent,),
    )
    expected_primal = getattr(operator, method)(values)
    expected_tangent = getattr(operator, method)(tangent)
    cotangent = jnp.arange(primal.size, dtype=primal.dtype).reshape(primal.shape) / 7
    _, pullback = jax.vjp(lambda operand: getattr(operator, method)(operand), values)
    (actual_cotangent,) = pullback(cotangent)
    expected_cotangent = getattr(operator, companion)(cotangent)

    np.testing.assert_allclose(primal, expected_primal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(primal_tangent, expected_tangent, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_cotangent, expected_cotangent, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("method", "companion", "leading_dimension"),
    [
        ("matmat", "rmatmat", "n_variants"),
        ("rmatmat", "matmat", "n_samples"),
    ],
)
def test_packed_product_linearize_transpose_and_finite_difference(
    method: str,
    companion: str,
    leading_dimension: str,
) -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.arange(getattr(operator, leading_dimension), dtype=jnp.float32) / 3
    tangent = jnp.asarray([0.75, -0.5], dtype=jnp.float32)
    primal, linearized = jax.linearize(lambda operand: getattr(operator, method)(operand), values)
    expected_tangent = getattr(operator, method)(tangent)
    cotangent = jnp.asarray([0.25, -1.0], dtype=jnp.float32)
    (transposed,) = jax.linear_transpose(
        lambda operand: getattr(operator, method)(operand),
        values,
    )(cotangent)
    expected_transposed = getattr(operator, companion)(cotangent)

    def scalar_loss(operand):
        return jnp.vdot(getattr(operator, method)(operand), cotangent)

    epsilon = jnp.asarray(1e-3, dtype=values.dtype)
    finite_difference = (scalar_loss(values + epsilon * tangent) - scalar_loss(values - epsilon * tangent)) / (
        2 * epsilon
    )

    np.testing.assert_allclose(primal, getattr(operator, method)(values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(linearized(tangent), expected_tangent, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(transposed, expected_transposed, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(finite_difference, jnp.vdot(expected_transposed, tangent), rtol=2e-3, atol=2e-3)


def test_packed_product_composes_with_jit_grad_value_and_grad_and_second_order() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.asarray([0.25, -0.75], dtype=jnp.float32)

    def loss(graph_operator, operand):
        product = graph_operator.matmat(operand)
        return 0.5 * jnp.vdot(product, product)

    expected_gradient = operator.rmatmat(operator.matmat(values))
    eager_value, eager_gradient = jax.value_and_grad(loss, argnums=1)(operator, values)
    jit_value, jit_gradient = jax.jit(jax.value_and_grad(loss, argnums=1))(operator, values)
    grad_jit = jax.grad(jax.jit(loss), argnums=1)(operator, values)

    def dense_gradient(operand):
        return jax.grad(loss, argnums=1)(operator, operand)

    hessian_vector = jax.jvp(dense_gradient, (values,), (jnp.ones_like(values),))[1]
    expected_hessian_vector = operator.rmatmat(operator.matmat(jnp.ones_like(values)))

    np.testing.assert_allclose(eager_gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jit_gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad_jit, expected_gradient, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(hessian_vector, expected_hessian_vector, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jit_value, eager_value, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("method", "leading_dimension"),
    [
        ("matmat", "n_variants"),
        ("rmatmat", "n_samples"),
    ],
)
@pytest.mark.parametrize(
    ("right_hand_sides", "mapped_axis"),
    [
        (None, 0),
        (None, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
)
def test_packed_product_dense_vmap_fuses_vector_and_matrix_batches(
    method: str,
    leading_dimension: str,
    right_hand_sides: int | None,
    mapped_axis: int,
) -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    logical_shape = (getattr(operator, leading_dimension),)
    if right_hand_sides is not None:
        logical_shape += (right_hand_sides,)
    batch_size = 3
    batch_first = (
        jnp.arange(batch_size * np.prod(logical_shape), dtype=jnp.float32).reshape(
            batch_size,
            *logical_shape,
        )
        / 5
    )
    values = jnp.moveaxis(batch_first, 0, mapped_axis)
    product = getattr(operator, method)

    actual = jax.vmap(product, in_axes=mapped_axis, out_axes=mapped_axis)(values)
    expected = jnp.stack([product(value) for value in batch_first], axis=mapped_axis)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_packed_product_invariant_graph_scan_remat_and_dce() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.arange(6, dtype=jnp.float32).reshape(3, operator.n_variants) / 4

    def body(graph_operator, operand):
        return graph_operator, graph_operator.matmat(operand)

    final_operator, scanned = jax.lax.scan(body, operator, values)
    rematerialized = jax.jit(jax.remat(lambda graph_operator, operand: graph_operator.matmat(operand)))(
        operator,
        values[0],
    )
    dce_ir = str(
        jax.jit(lambda graph_operator, operand: (graph_operator.matmat(operand), operand)[1])
        .lower(operator, values[0])
        .compiler_ir(dialect="stablehlo")
    )

    assert jax.typeof(final_operator.graph) == jax.typeof(operator.graph)
    np.testing.assert_allclose(scanned, jnp.stack([operator.matmat(value) for value in values]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rematerialized, operator.matmat(values[0]), rtol=1e-6, atol=1e-6)
    assert "sdy.manual_computation" not in dce_ir


def test_packed_ffi_backend_composes_through_dense_transform_matrix() -> None:
    ffi_cpu.is_ffi_cpu_available.cache_clear()
    ffi_cpu.is_ffi_cpu_packed_available.cache_clear()
    operator = _operator(
        _block(),
        mesh=_two_device_graph_mesh_or_skip(),
        backend=linear_dag.Backend.FFI_CPU,
    )
    pure = _operator(
        _block(),
        mesh=operator.graph_mesh,
        backend=linear_dag.Backend.PURE_JAX,
    )
    values = jnp.asarray([0.25, -0.75], dtype=jnp.float32)
    tangent = jnp.asarray([-0.5, 1.25], dtype=jnp.float32)

    def product(graph_operator, operand):
        return graph_operator.matmat(operand)

    def loss(graph_operator, operand):
        result = product(graph_operator, operand)
        return jnp.sum(jnp.tanh(result) ** 2)

    jit_result = jax.jit(product)(operator, values)
    primal, jvp_result = jax.jvp(lambda operand: product(operator, operand), (values,), (tangent,))
    _, pullback = jax.vjp(lambda operand: product(operator, operand), values)
    (vjp_result,) = pullback(tangent)
    gradient = jax.jit(jax.grad(loss, argnums=1))(operator, values)
    hvp = jax.jvp(lambda operand: jax.grad(loss, argnums=1)(operator, operand), (values,), (tangent,))[1]
    batched = jax.vmap(lambda operand: product(operator, operand))(jnp.stack((values, tangent)))
    _, scanned = jax.lax.scan(
        lambda graph_operator, operand: (graph_operator, product(graph_operator, operand)),
        operator,
        jnp.stack((values, tangent)),
    )
    rematerialized = jax.jit(jax.remat(product))(operator, values)

    np.testing.assert_allclose(jit_result, product(pure, values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(primal, product(pure, values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jvp_result, product(pure, tangent), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(vjp_result, pure.rmatmat(tangent), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gradient, jax.grad(loss, argnums=1)(pure, values), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        hvp,
        jax.jvp(lambda operand: jax.grad(loss, argnums=1)(pure, operand), (values,), (tangent,))[1],
        rtol=1e-6,
        atol=1e-6,
    )
    expected_batch = jnp.stack((product(pure, values), product(pure, tangent)))
    np.testing.assert_allclose(batched, expected_batch, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(scanned, expected_batch, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rematerialized, product(pure, values), rtol=1e-6, atol=1e-6)

    high_jaxpr = str(jax.make_jaxpr(product)(operator, values))
    lowered = str(jax.jit(product).lower(operator, values).compiler_ir("stablehlo"))
    gradient_ir = str(jax.jit(jax.grad(loss, argnums=1)).lower(operator, values).compiler_ir("stablehlo"))
    assert "ffi_call" not in high_jaxpr
    assert "'backend': 'ffi_cpu'" in high_jaxpr
    assert "linear_dag_jaxlinarg_packed_solve_forward_f32" in lowered
    assert "linear_dag_jaxlinarg_packed_solve_forward_f32" in gradient_ir
    assert "linear_dag_jaxlinarg_packed_solve_backward_f32" in gradient_ir


def test_packed_product_symbolic_zero_dense_tangent_has_output_type() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 1), dtype=jnp.float32)
    primitive = hijax_adapter._matmat_primitive(operator.graph, values)
    graph_zero = jax.typeof(operator.graph).vspace_zero()
    dense_zero = ad_util.Zero(jax.typeof(values).to_tangent_aval())

    _, residual, output_nonzero = primitive.lin((False, False), operator.graph, values)
    result = primitive.linearized(residual, graph_zero, dense_zero)

    assert residual is operator.graph
    assert output_nonzero is False
    assert isinstance(result, ad_util.Zero)
    assert result.aval == primitive.out_aval.to_tangent_aval()


def test_packed_product_actual_jvp_preserves_symbolic_zero_product_path() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 1), dtype=jnp.float32)
    tangent = jnp.full_like(values, 2.0)

    def forward(operand):
        return operator.matmat(jax.lax.stop_gradient(operand)), operand

    (_, primal_passthrough), (product_tangent, passthrough_tangent) = jax.jvp(
        forward,
        (values,),
        (tangent,),
    )

    np.testing.assert_array_equal(product_tangent, jnp.zeros_like(operator.matmat(values)))
    np.testing.assert_array_equal(primal_passthrough, values)
    np.testing.assert_array_equal(passthrough_tangent, tangent)


def test_packed_product_actual_vjp_accepts_symbolic_zero_output_cotangent() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 1), dtype=jnp.float32)
    output, pullback = jax.vjp(lambda operand: (operator.matmat(operand), operand), values)
    product_zero = ad_util.Zero(jax.typeof(output[0]).to_ct_aval())
    passthrough_cotangent = jnp.full_like(values, 3.0)

    (actual_cotangent,) = pullback((product_zero, passthrough_cotangent))

    np.testing.assert_array_equal(actual_cotangent, passthrough_cotangent)


def test_packed_product_reverse_symbolic_zero_remains_symbolic() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    primitive = hijax_adapter._matmat_primitive(operator.graph, values)
    output_zero = ad_util.Zero(primitive.out_aval.to_ct_aval())
    _, residual, output_nonzero = primitive.vjp_fwd((False, True), operator.graph, values)

    graph_cotangent, dense_cotangent = primitive.vjp_bwd_retval(residual, output_zero)
    _, inactive_residual, inactive_output_nonzero = primitive.vjp_fwd((False, False), operator.graph, values)
    inactive_graph_cotangent, inactive_dense_cotangent = primitive.vjp_bwd_retval(
        inactive_residual,
        output_zero,
    )

    assert residual is operator.graph
    assert inactive_residual is operator.graph
    assert output_nonzero is True
    assert isinstance(graph_cotangent, _PackedGraphZeroValue)
    assert isinstance(dense_cotangent, ad_util.Zero)
    assert dense_cotangent.aval == primitive.in_avals[1].to_ct_aval()
    assert inactive_output_nonzero is False
    assert isinstance(inactive_graph_cotangent, _PackedGraphZeroValue)
    assert isinstance(inactive_dense_cotangent, ad_util.Zero)
    assert inactive_dense_cotangent.aval == primitive.in_avals[1].to_ct_aval()


def test_packed_product_identity_includes_dense_and_graph_abstract_signatures() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    graph_type = jax.typeof(operator.graph)
    signature = hijax_adapter._signature_from_graph(operator.graph)
    narrow_values = jnp.ones((operator.n_variants, 1), dtype=jnp.float32)
    wide_values = jnp.ones((operator.n_variants, 3), dtype=jnp.float32)

    def primitive_for(candidate_graph_type, candidate_values):
        dense_type = (
            candidate_values if isinstance(candidate_values, core.ShapedArray) else jax.typeof(candidate_values)
        )
        return hijax_adapter._PackedMatmatPrimitive(
            candidate_graph_type,
            dense_type,
            n_samples=signature.n_samples,
            n_variants=signature.n_variants,
            capacities=signature.capacities,
            data_dtype=signature.data_dtype,
            output_axes=(),
        )

    def replace_component(index, component_type):
        components = list(graph_type.component_types)
        components[index] = component_type
        return replace(graph_type, component_types=tuple(components))

    narrow = primitive_for(graph_type, narrow_values)
    narrow_copy = primitive_for(
        replace(graph_type, component_types=tuple(component.update() for component in graph_type.component_types)),
        jax.typeof(narrow_values).update(),
    )
    wide = primitive_for(graph_type, wide_values)
    weak = primitive_for(graph_type, jax.typeof(narrow_values).update(weak_type=True))
    changed_component = graph_type.component_types[1].update(
        shape=(graph_type.component_types[1].shape[0], graph_type.component_types[1].shape[1] + 1)
    )
    changed_shape = primitive_for(replace_component(1, changed_component), narrow_values)
    changed_dtype = primitive_for(
        replace_component(1, graph_type.component_types[1].update(dtype=jnp.float32)),
        narrow_values,
    )
    changed_sharding = primitive_for(
        replace_component(
            1,
            graph_type.component_types[1].update(
                sharding=NamedSharding(
                    graph_type.component_types[1].sharding.mesh.update_axis_types({"graph": AxisType.Explicit}),
                    P("graph", None),
                )
            ),
        ),
        narrow_values,
    )

    assert narrow == narrow_copy
    assert hash(narrow) == hash(narrow_copy)
    assert narrow != wide
    assert hash(narrow) != hash(wide)
    assert narrow != weak
    assert hash(narrow) != hash(weak)
    assert len({narrow, changed_shape, changed_dtype, changed_sharding}) == 4
    assert "dense_abstract_signature" in narrow.params
    assert "graph_abstract_signature" in narrow.params
    assert _contains_only_normalized_signature_values(narrow.params["dense_abstract_signature"])
    assert _contains_only_normalized_signature_values(narrow.params["graph_abstract_signature"])


def test_abstract_signatures_cover_every_shaped_array_identity_field() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    graph_type = jax.typeof(operator.graph)
    dense_type = jax.typeof(jnp.ones((operator.n_variants, 1), dtype=jnp.float32))
    explicit_mesh = graph_type.component_types[0].sharding.mesh.update_axis_types({"graph": AxisType.Explicit})
    manual_mesh = graph_type.component_types[0].sharding.mesh.update_axis_types({"graph": AxisType.Manual})

    dense_candidates = (
        dense_type.update(shape=(dense_type.shape[0], dense_type.shape[1] + 1)),
        dense_type.update(dtype=jnp.float16),
        dense_type.update(weak_type=True),
        dense_type.update(sharding=NamedSharding(explicit_mesh, P(None, None))),
        dense_type.update(
            sharding=NamedSharding(manual_mesh, P(None, None)),
            manual_axis_type=core.ManualAxisType(varying={"graph"}),
        ),
        dense_type.update(memory_space=core.MemorySpace.Host),
    )
    dense_signature = hijax_adapter._dense_abstract_signature(dense_type)

    def graph_signature_with(component_type):
        components = list(graph_type.component_types)
        components[0] = component_type
        return hijax_adapter._graph_abstract_signature(replace(graph_type, component_types=tuple(components)))

    component_type = graph_type.component_types[0]
    graph_candidates = (
        component_type.update(shape=(component_type.shape[0], component_type.shape[1] + 1)),
        component_type.update(dtype=jnp.float32),
        component_type.update(weak_type=True),
        component_type.update(sharding=NamedSharding(explicit_mesh, P("graph", None))),
        component_type.update(
            sharding=NamedSharding(manual_mesh, P("graph", None)),
            manual_axis_type=core.ManualAxisType(varying={"graph"}),
        ),
        component_type.update(memory_space=core.MemorySpace.Host),
    )
    graph_signature = hijax_adapter._graph_abstract_signature(graph_type)

    assert (
        len({dense_signature, *(hijax_adapter._dense_abstract_signature(candidate) for candidate in dense_candidates)})
        == 7
    )
    assert len({graph_signature, *(graph_signature_with(candidate) for candidate in graph_candidates)}) == 7
    assert _contains_only_normalized_signature_values(dense_signature)
    assert _contains_only_normalized_signature_values(graph_signature)


def test_packed_product_linearized_jaxpr_has_graph_as_only_residual_operand() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 3), dtype=jnp.float32)
    _, linearized = jax.linearize(operator.matmat, values)

    closed_jaxpr = jax.make_jaxpr(linearized)(jnp.ones_like(values))
    product_equation = next(
        equation
        for equation in closed_jaxpr.jaxpr.eqns
        if type(equation.params.get("_prim")).__name__ == "_PackedMatmatPrimitive"
    )

    assert len(closed_jaxpr.jaxpr.constvars) == 1
    assert isinstance(closed_jaxpr.jaxpr.constvars[0].aval, _PackedGraphType)
    assert len(closed_jaxpr.jaxpr.invars) == 1
    assert product_equation.invars[0] is closed_jaxpr.jaxpr.constvars[0]
    assert len(product_equation.invars) == 2


@pytest.mark.parametrize(
    ("method", "companion_type"),
    [
        ("matmat", "_PackedRmatmatPrimitive"),
        ("rmatmat", "_PackedMatmatPrimitive"),
    ],
)
def test_packed_product_differentiated_jaxpr_uses_companion_primitive(
    method: str,
    companion_type: str,
) -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    leading_dimension = operator.n_variants if method == "matmat" else operator.n_samples
    values = jnp.ones((leading_dimension, 2), dtype=jnp.float32)

    differentiated = jax.make_jaxpr(jax.grad(lambda operand: jnp.sum(getattr(operator, method)(operand))))(values)
    primitive_types = [
        type(equation.params["_prim"]).__name__
        for equation in differentiated.jaxpr.eqns
        if equation.primitive.name == "call_hi_primitive"
    ]

    assert companion_type in primitive_types


def test_packed_product_vjp_residual_contains_one_graph_and_no_dense_primal() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 3), dtype=jnp.float32)

    _, pullback = jax.vjp(lambda operand: operator.matmat(operand), values)

    assert len(pullback.args_res) == 1
    assert type(pullback.args_res[0]).__name__ == "NotNeeded"
    assert len(pullback.opaque_residuals) == 1
    assert pullback.opaque_residuals[0] is operator.graph
    assert len(pullback.jaxpr.constvars) == 1
    assert isinstance(pullback.jaxpr.constvars[0].aval, _PackedGraphType)
    assert all(residual is not values for residual in pullback.opaque_residuals)


@pytest.mark.parametrize(
    "transform",
    ["jvp", "vjp", "grad", "jacfwd", "jacrev", "linear_transpose"],
)
def test_packed_product_rejects_every_graph_differentiation_transform_actionably(transform: str) -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 1), dtype=jnp.float32)
    signature = hijax_adapter._signature_from_graph(operator.graph)

    def product(graph):
        return hijax_adapter._bind_matmat_rank2(
            graph,
            values,
            signature=signature,
            output_axes=(),
        )

    def differentiate_graph() -> None:
        if transform == "jvp":
            jax.jvp(product, (operator.graph,), (operator.graph,))
        elif transform == "vjp":
            jax.vjp(product, operator.graph)
        elif transform == "grad":
            jax.grad(lambda graph: jnp.sum(product(graph)))(operator.graph)
        elif transform == "jacfwd":
            jax.jacfwd(product)(operator.graph)
        elif transform == "jacrev":
            jax.jacrev(product)(operator.graph)
        else:
            jax.linear_transpose(product, operator.graph)(jnp.ones((operator.n_samples, 1), dtype=jnp.float32))

    with pytest.raises(TypeError, match="opaque graph.*invariant"):
        differentiate_graph()


def test_packed_product_rejects_graph_batching() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 1), dtype=jnp.float32)
    signature = hijax_adapter._signature_from_graph(operator.graph)

    with pytest.raises(TypeError, match="opaque graph.*invariant"):
        jax.vmap(
            lambda graph: hijax_adapter._bind_matmat_rank2(
                graph,
                values,
                signature=signature,
                output_axes=(),
            ),
            in_axes=hijax_adapter._PackedGraphMappingSpec(mapped=True),
            axis_size=2,
        )(operator.graph)


def test_packed_product_high_level_jaxpr_has_one_project_product_equation() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)

    closed_jaxpr = jax.make_jaxpr(lambda graph_operator, operand: graph_operator.matmat(operand))(operator, values)
    product_equations = [
        equation
        for equation in closed_jaxpr.jaxpr.eqns
        if equation.primitive.name == "call_hi_primitive"
        and type(equation.params["_prim"]).__name__ == "_PackedMatmatPrimitive"
    ]

    assert len(product_equations) == 1
    primitive = product_equations[0].params["_prim"]
    assert all(isinstance(value, (str, int, tuple, np.dtype)) for value in primitive.params.values())
    assert not any(isinstance(value, (_PackedGraphValue, jax.Array, Mesh)) for value in primitive.params.values())


def test_project_binders_keep_graph_arrays_out_of_params_and_closures() -> None:
    operator = _operator(_block(), mesh=_two_device_graph_mesh_or_skip())
    values = jnp.ones((operator.n_variants, 2), dtype=jnp.float32)
    closed_jaxpr = jax.make_jaxpr(lambda graph_operator, operand: graph_operator.matmat(operand))(operator, values)
    equation = next(
        equation
        for equation in closed_jaxpr.jaxpr.eqns
        if type(equation.params.get("_prim")).__name__ == "_PackedMatmatPrimitive"
    )
    primitive = equation.params["_prim"]

    assert len(equation.invars) == 2
    assert isinstance(equation.invars[0].aval, _PackedGraphType)
    assert set(primitive.params) == {
        "n_samples",
        "n_variants",
        "capacities",
        "data_dtype",
        "backend",
        "output_axes",
        "dense_abstract_signature",
        "graph_abstract_signature",
    }
    assert all(_is_hashable(value) for value in primitive.params.values())
    assert not any(isinstance(value, (_PackedGraphValue, jax.Array, Mesh)) for value in primitive.params.values())
    for binder in (hijax_adapter._bind_matmat_rank2, hijax_adapter._bind_rmatmat_rank2):
        assert binder.__closure__ is None
        assert "graph" in inspect.signature(binder).parameters
        assert "values" in inspect.signature(binder).parameters


def _is_hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _contains_only_normalized_signature_values(value: Any) -> bool:
    if isinstance(value, tuple):
        return all(_contains_only_normalized_signature_values(item) for item in value)
    return value is None or isinstance(value, (bool, int, str))
