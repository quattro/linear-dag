# pattern: Mixed (unavoidable)
# Reason: This private compatibility boundary owns experimental JAX type,
# lowering, mapping, sharding, and registration hooks while numerical work
# remains delegated to project-owned packed product functions.

"""Private HiJAX representation for opaque packed LinearARG graph state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from jax.experimental import hijax
from jax.sharding import PartitionSpec as P

from .packing import _PackedGraphLogicalMetadata, PACKED_COMPONENT_NAMES

_OPAQUE_GRAPH_GUIDANCE = "packed LinearARG opaque graph state must be used as an invariant operand"


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
