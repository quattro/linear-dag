# pattern: Mixed (unavoidable)
# Reason: This private compatibility boundary owns experimental JAX type,
# lowering, mapping, sharding, and registration hooks while numerical work
# remains delegated to project-owned packed product functions.

"""Private HiJAX representation for opaque packed LinearARG graph state."""

from __future__ import annotations

import inspect

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib

from jax._src import ad_util
from jax._src.interpreters import ad
from jax.experimental import hijax
from jax.sharding import NamedSharding, PartitionSpec as P

from .packing import _PackedGraphLogicalMetadata, PACKED_COMPONENT_NAMES

_OPAQUE_GRAPH_GUIDANCE = "packed LinearARG opaque graph state must be used as an invariant operand"
_SUPPORTED_JAX_VERSION = "0.11.0"
_REQUIRED_HIJAX_SIGNATURES = {
    "HiType": "()",
    "HiType.lo_ty": "(self)",
    "HiType.lower_val": "(self, hi_val)",
    "HiType.raise_val": "(self, *lo_vals)",
    "HiType.to_tangent_aval": "(self)",
    "HiType.to_ct_aval": "(self)",
    "HiType.vspace_zero": "(self)",
    "HiType.vspace_add": "(self, x, y)",
    "HiType.dec_rank": "(self, size, spec)",
    "HiType.inc_rank": "(self, size, spec)",
    "HiType.leading_axis_spec": "(self)",
    "HiType.shard": "(self, mesh, manual_axes, check_vma, spec)",
    "HiType.unshard": "(self, mesh, check_vma, spec)",
    "HiType.nospec": "(self, mesh, check_vma, all_names)",
    "HiType.str_short": "(self, short_dtypes=False, mesh_axis_types=False)",
    "HiPspec": "()",
    "HiPspec.to_lo": "(self)",
    "HiPspec.to_tangent_spec": "(self)",
    "HiPspec.to_ct_spec": "(self)",
    "MappingSpec": "()",
    "HiPrimitive": "(name)",
    "HiPrimitive.bind": "(self, *args, **params)",
    "HiPrimitive.abstract_eval": "(self, *arg_avals, **params)",
    "HiPrimitive.to_lojax": "(self, *lotypes_wrapped_in_hitypes, **params)",
    "HiPrimitive.jvp": "(self, primals, tangents, **params)",
    "HiPrimitive.transpose": "(self, *args, **params)",
    "VJPHiPrimitive": "()",
    "VJPHiPrimitive.__call__": "(self, *args)",
    "VJPHiPrimitive.expand": "(self, *args)",
    "VJPHiPrimitive.lin": "(self, nzs_in, *primals)",
    "VJPHiPrimitive.linearized": "(self, residuals, *tangents)",
    "VJPHiPrimitive.vjp_fwd": "(self, nzs_in, /, *args)",
    "VJPHiPrimitive.vjp_bwd_retval": "(self, res, outgrad, /)",
    "VJPHiPrimitive.transpose": "(self, out_ct, *maybe_accums)",
    "VJPHiPrimitive.batch": "(self, axis_data, args, dims)",
    "jvp_from_lin": "(self, primals, tangents)",
    "register_hitype": "(val_cls, typeof_fn)",
}


class _PackedGraphDType:
    """Sentinel rejected by dtype consumers but accepted by JAX type probes."""

    def __repr__(self) -> str:
        return f"{_OPAQUE_GRAPH_GUIDANCE}; graph differentiation is not supported"


_PACKED_GRAPH_DTYPE = _PackedGraphDType()


def _assert_hijax_compatibility() -> None:
    """Fail once, at the adapter boundary, for unsupported HiJAX surfaces."""
    incompatibilities = []
    if jax.__version__ != _SUPPORTED_JAX_VERSION or jaxlib.__version__ != _SUPPORTED_JAX_VERSION:
        incompatibilities.append(f"found JAX/JAXlib {jax.__version__}/{jaxlib.__version__}")
    for qualified_name, expected_signature in _REQUIRED_HIJAX_SIGNATURES.items():
        target: Any = hijax
        try:
            for name in qualified_name.split("."):
                target = getattr(target, name)
        except AttributeError:
            incompatibilities.append(f"missing {qualified_name}")
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            incompatibilities.append(f"non-inspectable {qualified_name}")
            continue
        signature = signature.replace(
            parameters=[
                parameter.replace(annotation=inspect.Signature.empty) for parameter in signature.parameters.values()
            ],
            return_annotation=inspect.Signature.empty,
        )
        if str(signature) != expected_signature:
            incompatibilities.append(f"{qualified_name}{signature} != {expected_signature}")
    if incompatibilities:
        details = "; ".join(incompatibilities)
        raise ImportError(
            "linear_dag's private packed adapter supports exactly JAX/JAXlib 0.11.0; "
            f"incompatible jax.experimental.hijax surface: {details}"
        )


_assert_hijax_compatibility()


@dataclass(frozen=True, slots=True)
class _PackedGraphValue:
    """One immutable high-level value backed by fixed ordered graph arrays."""

    components: tuple[Any, ...]
    metadata: _PackedGraphLogicalMetadata

    def __post_init__(self) -> None:
        if len(self.components) != len(PACKED_COMPONENT_NAMES):
            raise ValueError(f"packed graph must contain {len(PACKED_COMPONENT_NAMES)} ordered components")
        if not isinstance(self.metadata, _PackedGraphLogicalMetadata):
            raise TypeError("packed graph metadata must use the compact logical metadata contract")


@dataclass(frozen=True, slots=True)
class _PackedGraphZeroValue:
    """Inert symbolic graph tangent/cotangent with no graph array payload."""

    metadata: _PackedGraphLogicalMetadata


@dataclass(frozen=True, slots=True)
class _PackedGraphMappingSpec(hijax.MappingSpec):
    """Mapping intent for an opaque graph value."""

    mapped: bool


@dataclass(frozen=True, slots=True)
class _PackedGraphZeroPspec(hijax.HiPspec):
    """Sharding contract for the array-free graph zero value."""

    def to_lo(self) -> tuple[P, ...]:
        return ()

    def to_tangent_spec(self) -> _PackedGraphZeroPspec:
        return self

    def to_ct_spec(self) -> _PackedGraphZeroPspec:
        return self


@dataclass(frozen=True, slots=True)
class _PackedGraphPspec(hijax.HiPspec):
    """Fixed per-component graph-axis sharding specifications."""

    component_specs: tuple[P, ...]

    def to_lo(self) -> tuple[P, ...]:
        return self.component_specs

    def to_tangent_spec(self) -> _PackedGraphZeroPspec:
        return _PackedGraphZeroPspec()

    def to_ct_spec(self) -> _PackedGraphZeroPspec:
        return _PackedGraphZeroPspec()


@dataclass(frozen=True, slots=True)
class _PackedGraphZeroType(hijax.HiType):
    """Array-free vector-space type for inactive graph tangents."""

    metadata: _PackedGraphLogicalMetadata

    def lo_ty(self) -> list[Any]:
        return []

    def lower_val(self, hi_val: _PackedGraphZeroValue) -> list[Any]:
        if not isinstance(hi_val, _PackedGraphZeroValue) or hi_val.metadata != self.metadata:
            raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; only its graph-zero tangent is valid")
        return []

    def raise_val(self, *lo_vals: Any) -> _PackedGraphZeroValue:
        if lo_vals:
            raise TypeError("graph-zero values must not carry lowered array payloads")
        return _PackedGraphZeroValue(self.metadata)

    def to_tangent_aval(self) -> _PackedGraphZeroType:
        return self

    def to_ct_aval(self) -> _PackedGraphZeroType:
        return self

    def vspace_zero(self) -> _PackedGraphZeroValue:
        return self.raise_val()

    def vspace_add(
        self,
        x: _PackedGraphZeroValue,
        y: _PackedGraphZeroValue,
    ) -> _PackedGraphZeroValue:
        self.lower_val(x)
        self.lower_val(y)
        return self.raise_val()

    def dec_rank(self, size: int | None, spec: hijax.MappingSpec) -> _PackedGraphZeroType:
        _validate_invariant_mapping(spec)
        return self

    def inc_rank(self, size: int | None, spec: hijax.MappingSpec) -> _PackedGraphZeroType:
        _validate_invariant_mapping(spec)
        return self

    def leading_axis_spec(self) -> _PackedGraphMappingSpec:
        return _PackedGraphMappingSpec(mapped=True)

    def shard(
        self,
        mesh: Any,
        manual_axes: frozenset[Any],
        check_vma: bool,
        spec: hijax.HiPspec,
    ) -> _PackedGraphZeroType:
        del manual_axes, check_vma
        _require_graph_mesh(mesh)
        if not isinstance(spec, _PackedGraphZeroPspec):
            raise TypeError("graph-zero shard_map specs must use the private graph-zero contract")
        return self

    def unshard(
        self,
        mesh: Any,
        check_vma: bool,
        spec: hijax.HiPspec,
    ) -> _PackedGraphZeroType:
        del check_vma
        _require_graph_mesh(mesh)
        if not isinstance(spec, _PackedGraphZeroPspec):
            raise TypeError("graph-zero shard_map specs must use the private graph-zero contract")
        return self

    def nospec(
        self,
        mesh: Any,
        check_vma: bool,
        all_names: tuple[Any, ...],
    ) -> _PackedGraphZeroPspec:
        del check_vma, all_names
        _require_graph_mesh(mesh)
        return _PackedGraphZeroPspec()

    def str_short(self, short_dtypes: bool = False, mesh_axis_types: bool = False) -> str:
        del short_dtypes, mesh_axis_types
        return "opaque-packed-graph-zero"


@dataclass(frozen=True, slots=True)
class _PackedGraphType(hijax.HiType):
    """Hashable high-level type lowering to the fixed packed component order."""

    component_types: tuple[Any, ...]
    metadata: _PackedGraphLogicalMetadata

    def __post_init__(self) -> None:
        if len(self.component_types) != len(PACKED_COMPONENT_NAMES):
            raise ValueError(f"packed graph type must contain {len(PACKED_COMPONENT_NAMES)} component types")

    @property
    def dtype(self) -> _PackedGraphDType:
        """Return a sentinel that makes generic dtype consumers fail actionably."""
        return _PACKED_GRAPH_DTYPE

    def lo_ty(self) -> list[Any]:
        return list(self.component_types)

    def lower_val(self, hi_val: _PackedGraphValue) -> list[Any]:
        if not isinstance(hi_val, _PackedGraphValue):
            raise TypeError("packed graph lowering requires the private opaque graph value")
        if hi_val.metadata != self.metadata or len(hi_val.components) != len(self.component_types):
            raise TypeError("packed graph value does not match its compact logical type metadata")
        observed_types = tuple(jax.typeof(component) for component in hi_val.components)
        if observed_types != self.component_types:
            raise TypeError("packed graph component shapes, dtypes, or shardings do not match its high-level type")
        return list(hi_val.components)

    def raise_val(self, *lo_vals: Any) -> _PackedGraphValue:
        value = _PackedGraphValue(tuple(lo_vals), self.metadata)
        if tuple(jax.typeof(component) for component in value.components) != self.component_types:
            raise TypeError("lowered packed graph components do not match the fixed high-level component types")
        return value

    def to_tangent_aval(self) -> _PackedGraphZeroType:
        return _PackedGraphZeroType(self.metadata)

    def to_ct_aval(self) -> _PackedGraphZeroType:
        return self.to_tangent_aval()

    def vspace_zero(self) -> _PackedGraphZeroValue:
        return self.to_tangent_aval().vspace_zero()

    def dec_rank(self, size: int | None, spec: hijax.MappingSpec) -> _PackedGraphType:
        _validate_invariant_mapping(spec)
        return self

    def inc_rank(self, size: int | None, spec: hijax.MappingSpec) -> _PackedGraphType:
        _validate_invariant_mapping(spec)
        return self

    def leading_axis_spec(self) -> _PackedGraphMappingSpec:
        return _PackedGraphMappingSpec(mapped=True)

    def shard(
        self,
        mesh: Any,
        manual_axes: frozenset[Any],
        check_vma: bool,
        spec: hijax.HiPspec,
    ) -> _PackedGraphType:
        _require_graph_mesh(mesh)
        graph_spec = _validate_graph_pspec(spec, self)
        component_types = tuple(
            component_type.shard(mesh, manual_axes, check_vma, component_spec)
            for component_type, component_spec in zip(
                self.component_types,
                graph_spec.component_specs,
                strict=True,
            )
        )
        return _PackedGraphType(component_types, self.metadata)

    def unshard(
        self,
        mesh: Any,
        check_vma: bool,
        spec: hijax.HiPspec,
    ) -> _PackedGraphType:
        _require_graph_mesh(mesh)
        graph_spec = _validate_graph_pspec(spec, self)
        component_types = tuple(
            component_type.unshard(mesh, check_vma, component_spec)
            for component_type, component_spec in zip(
                self.component_types,
                graph_spec.component_specs,
                strict=True,
            )
        )
        return _PackedGraphType(component_types, self.metadata)

    def nospec(
        self,
        mesh: Any,
        check_vma: bool,
        all_names: tuple[Any, ...],
    ) -> _PackedGraphPspec:
        _require_graph_mesh(mesh)
        return _PackedGraphPspec(
            tuple(component_type.nospec(mesh, check_vma, all_names) for component_type in self.component_types)
        )

    def str_short(self, short_dtypes: bool = False, mesh_axis_types: bool = False) -> str:
        del short_dtypes, mesh_axis_types
        return f"opaque-packed-graph[{self.metadata.n_samples},{self.metadata.n_variants}]"


def _packed_graph_type(value: _PackedGraphValue) -> _PackedGraphType:
    return _PackedGraphType(
        component_types=tuple(jax.typeof(component) for component in value.components),
        metadata=value.metadata,
    )


def _packed_graph_zero_type(value: _PackedGraphZeroValue) -> _PackedGraphZeroType:
    return _PackedGraphZeroType(value.metadata)


def _validate_invariant_mapping(spec: hijax.MappingSpec) -> None:
    if not isinstance(spec, _PackedGraphMappingSpec) or spec.mapped:
        raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; mapped graph axes are not supported")


def _require_graph_mesh(mesh: Any) -> None:
    if "graph" not in tuple(mesh.axis_names):
        raise ValueError('packed graph shard_map mesh must contain the dedicated "graph" axis')


def _graph_pspec_for_type(graph_type: _PackedGraphType) -> _PackedGraphPspec:
    if not isinstance(graph_type, _PackedGraphType):
        raise TypeError("graph sharding specs require the private packed graph high-level type")
    return _PackedGraphPspec(
        tuple(P("graph", *([None] * (len(component_type.shape) - 1))) for component_type in graph_type.component_types)
    )


def _validate_graph_pspec(spec: hijax.HiPspec, graph_type: _PackedGraphType) -> _PackedGraphPspec:
    if not isinstance(spec, _PackedGraphPspec):
        raise TypeError("packed graph shard_map specs must use the private graph sharding contract")
    expected = _graph_pspec_for_type(graph_type)
    if spec != expected:
        raise ValueError("packed graph shard_map specs must shard every lowered component on the graph axis")
    return spec


def _shape_signature(shape: tuple[Any, ...]) -> tuple[int | str, ...]:
    return tuple(int(dimension) if isinstance(dimension, int) else str(dimension) for dimension in shape)


def _partition_spec_signature(spec: P) -> tuple[Any, ...]:
    def normalize_axis(axis: Any) -> Any:
        if axis is None:
            return None
        if isinstance(axis, tuple):
            return tuple(normalize_axis(item) for item in axis)
        return str(axis)

    return tuple(normalize_axis(axis) for axis in spec)


def _axis_names_signature(axis_names: Any) -> tuple[str, ...]:
    return tuple(sorted(str(axis_name) for axis_name in axis_names))


def _mesh_signature(mesh: Any) -> tuple[Any, ...]:
    abstract_device = getattr(mesh, "abstract_device", None)
    abstract_device_signature = (
        ()
        if abstract_device is None
        else (
            str(abstract_device.device_kind),
            None if abstract_device.num_cores is None else int(abstract_device.num_cores),
            str(abstract_device.platform),
        )
    )
    try:
        devices = tuple(
            (
                str(device.platform),
                int(device.process_index),
                int(device.id),
                str(device.device_kind),
            )
            for device in mesh.devices.flat
        )
    except (AttributeError, ValueError):
        devices = ()
    return (
        tuple(str(name) for name in mesh.axis_names),
        tuple((str(name), int(size)) for name, size in mesh.shape.items()),
        tuple(str(axis_type) for axis_type in mesh.axis_types),
        abstract_device_signature,
        devices,
    )


def _sharding_signature(sharding: Any) -> tuple[Any, ...]:
    if isinstance(sharding, NamedSharding):
        logical_device_ids = getattr(sharding, "_logical_device_ids", None)
        if logical_device_ids is not None:
            logical_device_ids = tuple(int(device_id) for device_id in logical_device_ids.flat)
        return (
            "NamedSharding",
            _mesh_signature(sharding.mesh),
            _partition_spec_signature(sharding.spec),
            str(sharding.memory_kind),
            logical_device_ids,
        )
    return (
        f"{type(sharding).__module__}.{type(sharding).__qualname__}",
        str(sharding),
    )


def _manual_axis_type_signature(manual_axis_type: Any) -> tuple[Any, ...]:
    unreduced_kind = manual_axis_type.unreduced_kind
    return (
        "ManualAxisType",
        _axis_names_signature(manual_axis_type.varying),
        _axis_names_signature(manual_axis_type.unreduced),
        _axis_names_signature(manual_axis_type.reduced),
        None
        if unreduced_kind is None
        else (f"{type(unreduced_kind).__module__}.{type(unreduced_kind).__qualname__}", unreduced_kind.name),
    )


def _memory_space_signature(memory_space: Any) -> tuple[str, str]:
    return (f"{type(memory_space).__module__}.{type(memory_space).__qualname__}", memory_space.name)


def _shaped_array_signature(array_type: Any) -> tuple[Any, ...]:
    return (
        ("shape", _shape_signature(array_type.shape)),
        ("dtype", str(array_type.dtype)),
        ("weak_type", bool(array_type.weak_type)),
        ("sharding", _sharding_signature(array_type.sharding)),
        ("manual_axis_type", _manual_axis_type_signature(array_type.manual_axis_type)),
        ("memory_space", _memory_space_signature(array_type.memory_space)),
    )


def _dense_abstract_signature(dense_type: Any) -> tuple[Any, ...]:
    return ("dense", *_shaped_array_signature(dense_type))


def _graph_abstract_signature(graph_type: _PackedGraphType) -> tuple[Any, ...]:
    return tuple(
        (
            "component",
            *_shaped_array_signature(component_type),
        )
        for component_type in graph_type.component_types
    )


class _PackedProductPrimitive(hijax.VJPHiPrimitive):
    """Shared handwritten transform contract for one packed linear product."""

    _direction: str
    n_samples: int
    n_variants: int
    capacities: tuple[int, ...]
    data_dtype: str
    output_axes: tuple[Any, ...]

    def __init__(
        self,
        graph_type: _PackedGraphType,
        dense_type: Any,
        *,
        n_samples: int,
        n_variants: int,
        capacities: tuple[int, ...],
        data_dtype: str,
        output_axes: tuple[Any, ...],
    ) -> None:
        if not isinstance(graph_type, _PackedGraphType):
            raise TypeError("packed product primitives require the private opaque graph type")
        if len(dense_type.shape) != 2:
            raise TypeError("packed product primitive dense operands must have rank two")
        expected_rows = n_variants if self._direction == "matmat" else n_samples
        output_rows = n_samples if self._direction == "matmat" else n_variants
        if dense_type.shape[0] != expected_rows:
            raise ValueError(f"packed {self._direction} expected leading dimension {expected_rows}")
        if dense_type.dtype != graph_type.component_types[2].dtype:
            raise TypeError("packed product dense operand dtype must match packed graph data")
        if self._direction == "rmatmat" and output_axes:
            raise ValueError("packed transpose products must return replicated logical variants")
        if output_axes not in ((), ("graph",)):
            raise ValueError("packed product output axes must be replicated or graph-sharded")
        output_spec = P(*output_axes, *([None] * (2 - len(output_axes))))
        graph_mesh = graph_type.component_types[0].sharding.mesh
        self.in_avals = (graph_type, dense_type)
        self.out_aval = dense_type.update(
            shape=(output_rows, dense_type.shape[1]),
            sharding=NamedSharding(graph_mesh, output_spec),
        )
        self.params = {
            "n_samples": int(n_samples),
            "n_variants": int(n_variants),
            "capacities": tuple(int(value) for value in capacities),
            "data_dtype": str(data_dtype),
            "output_axes": tuple(output_axes),
            "dense_abstract_signature": _dense_abstract_signature(dense_type),
            "graph_abstract_signature": _graph_abstract_signature(graph_type),
        }
        super().__init__()

    def _signature(self) -> Any:
        from .packed_products import _PackedProductSignature

        return _PackedProductSignature(
            n_samples=self.n_samples,
            n_variants=self.n_variants,
            capacities=self.capacities,
            data_dtype=self.data_dtype,
        )

    def expand(self, *args: Any) -> Any:
        from .packed_products import _lineararg_matmat_graph, _lineararg_rmatmat_graph

        graph, values = args
        if self._direction == "matmat":
            return _lineararg_matmat_graph(
                graph,
                values,
                signature=self._signature(),
                output_axes=self.output_axes,
            )
        return _lineararg_rmatmat_graph(graph, values, signature=self._signature())

    def lin(self, nzs_in: Any, *primals: Any) -> tuple[Any, Any, bool]:
        graph, values = primals
        _reject_nonzero_graph_input(nzs_in[0])
        dense_nonzero = bool(nzs_in[1])
        return self(graph, values), graph, dense_nonzero

    def linearized(self, residuals: Any, *tangents: Any) -> Any:
        graph = residuals
        graph_tangent, values_tangent = tangents
        _validate_graph_zero_tangent(graph_tangent)
        if isinstance(values_tangent, ad_util.Zero):
            return ad_util.Zero(self.out_aval.to_tangent_aval())
        return self(graph, values_tangent)

    jvp = hijax.jvp_from_lin

    def vjp_fwd(self, nzs_in: Any, /, *args: Any) -> tuple[Any, Any, bool]:
        graph, values = args
        _reject_nonzero_graph_input(nzs_in[0])
        dense_nonzero = bool(nzs_in[1])
        return self(graph, values), graph, dense_nonzero

    def vjp_bwd_retval(self, graph: _PackedGraphValue, output_cotangent: Any):
        graph_cotangent = _PackedGraphZeroValue(graph.metadata)
        if isinstance(output_cotangent, ad_util.Zero):
            dense_cotangent = ad_util.Zero(self.in_avals[1].to_ct_aval())
        else:
            dense_cotangent = self._bind_companion(graph, output_cotangent)
        return graph_cotangent, dense_cotangent

    def transpose(self, out_ct: Any, *maybe_accumulators: Any) -> None:
        graph, values_accumulator = maybe_accumulators
        if isinstance(graph, ad.GradAccum):
            raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; graph differentiation is not supported")
        if isinstance(values_accumulator, ad.GradAccum):
            if isinstance(out_ct, ad_util.Zero):
                values_accumulator.accum(ad_util.Zero(self.in_avals[1].to_ct_aval()))
            else:
                values_accumulator.accum(self._bind_companion(graph, out_ct))

    def _bind_companion(self, graph: _PackedGraphValue, values: Any) -> Any:
        signature = self._signature()
        if self._direction == "matmat":
            return _bind_rmatmat_rank2(graph, values, signature=signature)
        return _bind_matmat_rank2(graph, values, signature=signature, output_axes=())

    def batch(self, axis_data: Any, args: tuple[Any, Any], dims: tuple[Any, Any]):
        graph, values = args
        graph_dim, dense_dim = dims
        if not _is_invariant_graph_mapping(graph_dim):
            raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; mapped graph axes are not supported")
        if dense_dim is None:
            return self(graph, values), None
        if not isinstance(dense_dim, int):
            raise TypeError("packed product dense batching requires one integer mapped axis")
        batch_size = axis_data.size
        right_hand_sides = self.in_avals[1].shape[1]
        moved = jnp.moveaxis(values, dense_dim, -1)
        fused = moved.reshape(self.in_avals[1].shape[0], right_hand_sides * batch_size)
        fused_output = self._bind_same(graph, fused)
        restored = fused_output.reshape(self.out_aval.shape[0], right_hand_sides, batch_size)
        return jnp.moveaxis(restored, -1, dense_dim), dense_dim

    def _bind_same(self, graph: _PackedGraphValue, values: Any) -> Any:
        signature = self._signature()
        if self._direction == "matmat":
            return _bind_matmat_rank2(graph, values, signature=signature, output_axes=self.output_axes)
        return _bind_rmatmat_rank2(graph, values, signature=signature)


class _PackedMatmatPrimitive(_PackedProductPrimitive):
    """Private high-level primitive for packed forward products."""

    _direction = "matmat"


class _PackedRmatmatPrimitive(_PackedProductPrimitive):
    """Private high-level primitive for packed transpose products."""

    _direction = "rmatmat"


def _reject_nonzero_graph_input(nonzero: bool) -> None:
    if nonzero:
        raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; graph differentiation is not supported")


def _validate_graph_zero_tangent(tangent: Any) -> None:
    if not isinstance(tangent, (_PackedGraphZeroValue, ad_util.Zero)):
        raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; graph differentiation is not supported")


def _is_invariant_graph_mapping(spec: Any) -> bool:
    return spec is None or isinstance(spec, _PackedGraphMappingSpec) and not spec.mapped


def _matmat_primitive(
    graph: _PackedGraphValue,
    values: Any,
    *,
    signature: Any | None = None,
    output_axes: tuple[Any, ...] = (),
) -> _PackedMatmatPrimitive:
    signature = _signature_from_graph(graph) if signature is None else signature
    return _PackedMatmatPrimitive(
        jax.typeof(graph),
        jax.typeof(values),
        n_samples=signature.n_samples,
        n_variants=signature.n_variants,
        capacities=signature.capacities,
        data_dtype=signature.data_dtype,
        output_axes=output_axes,
    )


def _rmatmat_primitive(
    graph: _PackedGraphValue,
    values: Any,
    *,
    signature: Any | None = None,
) -> _PackedRmatmatPrimitive:
    signature = _signature_from_graph(graph) if signature is None else signature
    return _PackedRmatmatPrimitive(
        jax.typeof(graph),
        jax.typeof(values),
        n_samples=signature.n_samples,
        n_variants=signature.n_variants,
        capacities=signature.capacities,
        data_dtype=signature.data_dtype,
        output_axes=(),
    )


def _signature_from_graph(graph: _PackedGraphValue):
    from .packed_products import _PackedProductSignature

    graph_type = jax.typeof(graph)
    return _PackedProductSignature(
        n_samples=graph.metadata.n_samples,
        n_variants=graph.metadata.n_variants,
        capacities=graph.metadata.capacities,
        data_dtype=str(graph_type.component_types[2].dtype),
    )


def _bind_matmat_rank2(
    graph: _PackedGraphValue,
    values: Any,
    *,
    signature: Any,
    output_axes: tuple[Any, ...],
) -> Any:
    """Project-owned private binder for one rank-two forward product."""
    return _matmat_primitive(graph, values, signature=signature, output_axes=output_axes)(graph, values)


def _bind_rmatmat_rank2(graph: _PackedGraphValue, values: Any, *, signature: Any) -> Any:
    """Project-owned private binder for one rank-two transpose product."""
    return _rmatmat_primitive(graph, values, signature=signature)(graph, values)


class _PackedGraphComponentPrimitive(hijax.HiPrimitive):
    """Lower one named component without exposing the graph as a PyTree."""

    def abstract_eval(self, graph_type: _PackedGraphType, *, index: int):
        if not isinstance(graph_type, _PackedGraphType):
            raise TypeError("packed graph component access requires the private opaque graph value")
        return graph_type.component_types[index], set()

    def to_lojax(self, graph: _PackedGraphValue, *, index: int):
        return graph.components[index]

    def jvp(self, primals: Any, tangents: Any, **params: Any):
        del primals, tangents, params
        raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; graph differentiation is not supported")

    def transpose(self, out_bar: Any, graph: Any, *, index: int):
        del out_bar, graph, index
        raise TypeError(f"{_OPAQUE_GRAPH_GUIDANCE}; graph differentiation is not supported")


_packed_graph_component_p = _PackedGraphComponentPrimitive("packed_graph_component")


def _packed_graph_component(graph: Any, index: int) -> Any:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(PACKED_COMPONENT_NAMES):
        raise IndexError("packed graph component index is outside the fixed component order")
    return _packed_graph_component_p.bind(graph, index=index)


hijax.register_hitype(_PackedGraphValue, _packed_graph_type)
hijax.register_hitype(_PackedGraphZeroValue, _packed_graph_zero_type)
