# pattern: Mixed (unavoidable)
# Reason: The opt-in benchmark shell shares pure metric extraction and table
# formatting with the executable benchmark so reported IR facts stay aligned.

from __future__ import annotations

import json
import statistics
import time

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from jax.extend import core as jax_core
from jax.sharding import Mesh, NamedSharding

from linear_dag.core.jaxlinarg import Backend, JaxParallelOperator
from linear_dag.core.jaxlinarg.ingress import _packed_from_hdf5, _PackedJaxLinearARG
from linear_dag.core.jaxlinarg.kernels import ffi_cpu
from linear_dag.core.jaxlinarg.packed_products import lineararg_matmat, lineararg_rmatmat
from linear_dag.core.jaxlinarg.packing import GRAPH_FIELD_NAMES, PACKED_COMPONENT_NAMES
from linear_dag.core.parallel_processing import ParallelOperator
from tests.jax.bench._promotion import (
    build_promotion_evidence,
    compute_dataset_fingerprint,
    git_commit,
    is_git_dirty,
    make_record,
    PerformanceMetrics,
    Representation,
    TimedPhase,
    TimingPhase,
    write_evidence_fragment,
)

MIN_SAMPLE_SECONDS = 0.005
WARMUP_ITERATIONS = 2
TIMED_ITERATIONS = 9


@dataclass(frozen=True)
class ParallelBenchmarkResult:
    operator: str
    operation: str
    k: int | None
    median_seconds: float
    ratio_to_parallel_operator: float | None
    resident_graph_bytes: int | None = None
    max_device_graph_bytes: int | None = None
    canonical_graph_bytes: int | None = None
    padded_graph_bytes: int | None = None
    descriptor_bytes: int | None = None
    padding_ratio: float | None = None
    staging_bytes: int | None = None
    component_count: int | None = None
    pytree_leaf_count: int | None = None
    resident_devices_valid: bool | None = None
    graph_constant_bytes: int | None = None
    graph_operand_count: int | None = None
    stablehlo_operation_count: int | None = None
    requested_backend: Backend | None = None
    resolved_backend: Backend | None = None
    ffi_cpu_built: bool | None = None
    ffi_cpu_exact_available: bool | None = None
    ffi_cpu_packed_available: bool | None = None
    ffi_cpu_blas_enabled: bool | None = None
    ffi_cpu_blas_backend: str | None = None
    ffi_cpu_native_tuning: bool | None = None

    def __post_init__(self) -> None:
        if "ffi_cpu" in self.operator.lower() and self.resolved_backend is not Backend.FFI_CPU:
            observed = None if self.resolved_backend is None else self.resolved_backend.value
            raise ValueError(f"FFI-labeled benchmark row must have resolved backend ffi_cpu; observed {observed}")


@dataclass(frozen=True)
class _PromotionProductMeasurement:
    operation: str
    k: int
    phase: str
    seconds: float | None
    null_reason: str | None
    dtype: str
    requested_backend: str | None
    resolved_backend: str | None
    device_count: int
    metrics: PerformanceMetrics
    numeric_passed: bool = True


def _ffi_build_metadata() -> dict[str, Any]:
    return {
        "ffi_cpu_built": ffi_cpu.is_ffi_cpu_built(),
        "ffi_cpu_exact_available": ffi_cpu.is_ffi_cpu_available(),
        "ffi_cpu_packed_available": ffi_cpu.is_ffi_cpu_packed_available(),
        "ffi_cpu_blas_enabled": ffi_cpu.is_ffi_cpu_blas_enabled(),
        "ffi_cpu_blas_backend": ffi_cpu.ffi_cpu_blas_backend(),
        "ffi_cpu_native_tuning": ffi_cpu.is_ffi_cpu_native_tuning_enabled(),
    }


def test_benchmark_rows_record_requested_resolved_and_native_backend_configuration(monkeypatch):
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_built", lambda: True)
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_available", lambda: True)
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_packed_available", lambda: False)
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_blas_enabled", lambda: True)
    monkeypatch.setattr(ffi_cpu, "ffi_cpu_blas_backend", lambda: "openblas")
    monkeypatch.setattr(ffi_cpu, "is_ffi_cpu_native_tuning_enabled", lambda: True)

    result = ParallelBenchmarkResult(
        operator="jax_parallel_ffi_cpu_1_device",
        operation="matmat",
        k=4,
        median_seconds=0.1,
        ratio_to_parallel_operator=1.0,
        requested_backend=Backend.FFI_CPU,
        resolved_backend=Backend.FFI_CPU,
        **_ffi_build_metadata(),
    )
    table = _format_results_table([result])

    assert result.requested_backend is Backend.FFI_CPU
    assert result.resolved_backend is Backend.FFI_CPU
    assert result.ffi_cpu_built is True
    assert result.ffi_cpu_exact_available is True
    assert result.ffi_cpu_packed_available is False
    assert result.ffi_cpu_blas_enabled is True
    assert result.ffi_cpu_blas_backend == "openblas"
    assert result.ffi_cpu_native_tuning is True
    assert "requested backend" in table
    assert "resolved backend" in table
    assert "openblas" in table


def test_ffi_labeled_benchmark_row_rejects_non_ffi_resolution():
    with pytest.raises(ValueError, match="FFI-labeled benchmark row.*resolved.*ffi_cpu"):
        ParallelBenchmarkResult(
            operator="jax_parallel_ffi_cpu_1_device",
            operation="matmat",
            k=4,
            median_seconds=0.1,
            ratio_to_parallel_operator=1.0,
            requested_backend=Backend.FFI_CPU,
            resolved_backend=Backend.PURE_JAX,
        )


def test_graph_operand_count_uses_explicit_carrier_count_without_manual_computation() -> None:
    class Operation:
        name = "builtin.module"
        regions = ()

    assert _stablehlo_graph_operand_count(Operation()) == len(PACKED_COMPONENT_NAMES) - 1


def test_jax_parallel_operator_benchmark(
    request: pytest.FixtureRequest,
    linarg_h5_path,
    linarg_block_metadata,
    linarg_benchmark_k_values: tuple[int, ...],
    linarg_parallel_processes: int,
):
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")

    num_processes = min(linarg_parallel_processes, linarg_block_metadata.height)
    process_results = _time_parallel_operator(
        linarg_h5_path,
        linarg_block_metadata,
        num_processes=num_processes,
        k_values=linarg_benchmark_k_values,
    )
    baselines = {}
    for result in process_results:
        if result.k is None:
            continue
        baselines[(result.operation, result.k)] = result.median_seconds
    results = list(process_results)

    for config in _jax_parallel_configs(linarg_block_metadata):
        results.extend(
            _time_jax_parallel_operator(
                linarg_h5_path,
                linarg_block_metadata,
                config=config,
                baselines=baselines,
                k_values=linarg_benchmark_k_values,
            )
        )

    explicit_production_path = request.config.getoption("--linarg-h5-path") is not None
    packed_result = _benchmark_packed_ingress(
        linarg_h5_path,
        linarg_block_metadata,
        production_gate=explicit_production_path,
    )
    results.append(packed_result)
    if explicit_production_path:
        failures = _packed_gate_failures(packed_result)
        if failures:
            pytest.fail("packed production memory gate failed:\n- " + "\n- ".join(failures))

    _print_results(results)


def test_promotion_numpy_cython_product_child(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_benchmark_k_values: tuple[int, ...],
    linarg_parallel_processes: int,
) -> None:
    _run_promotion_product_child(
        request,
        representation=Representation.NUMPY_CYTHON,
        path=linarg_h5_path,
        block_metadata=linarg_block_metadata,
        k_values=linarg_benchmark_k_values,
        parallel_processes=linarg_parallel_processes,
    )


def test_promotion_exact_product_child(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_benchmark_k_values: tuple[int, ...],
    linarg_parallel_processes: int,
) -> None:
    _run_promotion_product_child(
        request,
        representation=Representation.RETAINED_EXACT_RAGGED,
        path=linarg_h5_path,
        block_metadata=linarg_block_metadata,
        k_values=linarg_benchmark_k_values,
        parallel_processes=linarg_parallel_processes,
    )


def test_promotion_packed_product_child(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_benchmark_k_values: tuple[int, ...],
    linarg_parallel_processes: int,
) -> None:
    _run_promotion_product_child(
        request,
        representation=Representation.PACKED_CANDIDATE,
        path=linarg_h5_path,
        block_metadata=linarg_block_metadata,
        k_values=linarg_benchmark_k_values,
        parallel_processes=linarg_parallel_processes,
    )


def _benchmark_packed_ingress(
    linarg_h5_path: Any,
    linarg_block_metadata: pl.DataFrame,
    *,
    production_gate: bool,
) -> ParallelBenchmarkResult:
    cpu_devices = tuple(_devices_for_backend("cpu"))
    if not cpu_devices:
        pytest.skip("packed ingress benchmark requires at least one JAX device")
    if production_gate and len(cpu_devices) < 2:
        pytest.fail(
            "the packed production residency gate requires two CPU devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import"
        )
    num_devices = 2 if production_gate else min(2, len(cpu_devices))
    mesh = Mesh(np.asarray(cpu_devices[:num_devices]), ("graph",))
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list())

    start = time.perf_counter()
    packed = _packed_from_hdf5(
        linarg_h5_path,
        block_names,
        mesh=mesh,
        allow_excess_padding=True,
    )
    construction_seconds = time.perf_counter() - start
    op = packed.operator
    for name in PACKED_COMPONENT_NAMES:
        getattr(op, name).block_until_ready()

    observed_graph_bytes = _graph_bytes_by_device(op)
    if sum(observed_graph_bytes.values()) != sum(packed.diagnostics.final_graph_bytes_by_device):
        pytest.fail("packed ingress diagnostics do not match observed graph residency")
    graph_constant_bytes, graph_operand_count, stablehlo_operation_count = _packed_ir_metrics(op)
    return _packed_memory_result(
        packed.diagnostics,
        operator=f"packed_jax_lineararg_{num_devices}_device",
        construction_seconds=construction_seconds,
        resident_devices_valid=_packed_fields_have_expected_residency(op),
        graph_constant_bytes=graph_constant_bytes,
        graph_operand_count=graph_operand_count,
        stablehlo_operation_count=stablehlo_operation_count,
        requested_backend=Backend.PURE_JAX,
        resolved_backend=op.backend,
    )


def _packed_memory_result(
    diagnostics: Any,
    *,
    operator: str,
    construction_seconds: float,
    resident_devices_valid: bool,
    graph_constant_bytes: int | None = None,
    graph_operand_count: int | None = None,
    stablehlo_operation_count: int | None = None,
    requested_backend: Backend = Backend.PURE_JAX,
    resolved_backend: Backend = Backend.PURE_JAX,
) -> ParallelBenchmarkResult:
    """Convert packed ingress diagnostics into the shared benchmark record."""
    final_graph_bytes = tuple(int(value) for value in diagnostics.final_graph_bytes_by_device)
    return ParallelBenchmarkResult(
        operator=operator,
        operation="ingress",
        k=None,
        median_seconds=construction_seconds,
        ratio_to_parallel_operator=None,
        resident_graph_bytes=sum(final_graph_bytes),
        max_device_graph_bytes=max(final_graph_bytes, default=0),
        canonical_graph_bytes=int(diagnostics.canonical_graph_bytes),
        padded_graph_bytes=int(diagnostics.padded_graph_bytes),
        descriptor_bytes=int(diagnostics.descriptor_bytes),
        padding_ratio=float(diagnostics.padding_ratio),
        staging_bytes=int(diagnostics.staging_bytes),
        component_count=int(diagnostics.component_count),
        pytree_leaf_count=int(diagnostics.pytree_leaf_count),
        resident_devices_valid=resident_devices_valid,
        graph_constant_bytes=graph_constant_bytes,
        graph_operand_count=graph_operand_count,
        stablehlo_operation_count=stablehlo_operation_count,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        **_ffi_build_metadata(),
    )


def _packed_ir_metrics(op: _PackedJaxLinearARG) -> tuple[int, int, int]:
    """Return explicit forward-program constant, operand, and operation counts."""
    values = jax.ShapeDtypeStruct((op.n_variants, 1), op.data.dtype)
    closed_jaxpr = jax.make_jaxpr(lineararg_matmat)(op, values)
    stablehlo = jax.jit(lineararg_matmat).lower(op, values).compiler_ir("stablehlo")
    return (
        _closed_jaxpr_array_constant_bytes(closed_jaxpr),
        _stablehlo_graph_operand_count(stablehlo),
        _stablehlo_operation_count(stablehlo),
    )


def _closed_jaxpr_array_constant_bytes(closed_jaxpr: jax_core.ClosedJaxpr) -> int:
    total = 0

    def add_constant(constant: Any) -> None:
        nonlocal total
        if isinstance(constant, (jax.Array, np.ndarray)):
            total += int(constant.size * constant.dtype.itemsize)
            return
        abstract_value = jax.typeof(constant)
        lower_val = getattr(abstract_value, "lower_val", None)
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


def _walk_ir_operations(value: Any):
    operation = getattr(value, "operation", value)
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for nested in block.operations:
                yield from _walk_ir_operations(nested)


def _stablehlo_graph_operand_count(stablehlo: Any) -> int:
    try:
        return len(_stablehlo_graph_operand_attributes(stablehlo))
    except StopIteration:
        # Single-device lowering removes the sharding wrapper. The explicit
        # packed carrier contract still fixes the graph operand count.
        return len(PACKED_COMPONENT_NAMES) - 1


def _stablehlo_graph_operand_attributes(stablehlo: Any) -> tuple[str, ...]:
    """Return graph-sharded input entries from JAX 0.11 manual computation."""
    manual_computation = next(
        operation for operation in _walk_ir_operations(stablehlo) if operation.name == "sdy.manual_computation"
    )
    serialized = str(manual_computation.attributes["in_shardings"])
    prefix = "#sdy.sharding_per_value<["
    suffix = "]>"
    if not serialized.startswith(prefix) or not serialized.endswith(suffix):
        raise ValueError("unexpected JAX 0.11 sdy.manual_computation input sharding format")
    entries = serialized[len(prefix) : -len(suffix)].split(">, <")
    normalized = tuple(
        f"#sdy.sharding<{entry.removeprefix('<').removesuffix('>')}>" for entry in entries if '"graph"' in entry
    )
    return normalized


def _stablehlo_operation_count(stablehlo: Any) -> int:
    return sum(operation.name.startswith("stablehlo.") for operation in _walk_ir_operations(stablehlo))


def _packed_gate_failures(result: ParallelBenchmarkResult) -> tuple[str, ...]:
    """Return every failed production packed-memory gate."""
    if result.canonical_graph_bytes is None or result.padding_ratio is None:
        raise ValueError("packed gate requires canonical bytes and a padding ratio")
    if result.max_device_graph_bytes is None:
        raise ValueError("packed gate requires maximum device graph residency")

    failures = []
    if result.padding_ratio > 1.25:
        failures.append(f"packed padding ratio {result.padding_ratio:.6f} exceeds 1.250000")
    residency_limit = 0.65 * result.canonical_graph_bytes
    if result.max_device_graph_bytes > residency_limit:
        failures.append(
            f"maximum device graph residency {result.max_device_graph_bytes} exceeds "
            f"0.65 * canonical graph bytes ({residency_limit:.3f})"
        )
    if result.resident_devices_valid is not True:
        failures.append("one or more packed fields has an unexpected resident device or shard index")
    return tuple(failures)


def _time_parallel_operator(
    linarg_h5_path,
    linarg_block_metadata: pl.DataFrame,
    *,
    num_processes: int,
    k_values: tuple[int, ...],
) -> list[ParallelBenchmarkResult]:
    with ParallelOperator.from_hdf5(
        str(linarg_h5_path),
        num_processes=num_processes,
        max_num_traits=max(k_values),
        block_metadata=linarg_block_metadata,
    ) as op:
        variant_inputs, sample_inputs = _benchmark_inputs(op.shape, k_values=k_values)
        results = []
        for k, matrix in variant_inputs.items():
            results.append(
                ParallelBenchmarkResult(
                    f"parallel_operator_{num_processes}_processes",
                    "matmat",
                    k,
                    _time_call(lambda matrix=matrix: op.matmat(matrix)),
                    1.0,
                    requested_backend=None,
                    resolved_backend=None,
                    **_ffi_build_metadata(),
                )
            )
        for k, matrix in sample_inputs.items():
            results.append(
                ParallelBenchmarkResult(
                    f"parallel_operator_{num_processes}_processes",
                    "rmatmat",
                    k,
                    _time_call(lambda matrix=matrix: op.rmatmat(matrix)),
                    1.0,
                    requested_backend=None,
                    resolved_backend=None,
                    **_ffi_build_metadata(),
                )
            )
        return results


def _time_jax_parallel_operator(
    linarg_h5_path,
    linarg_block_metadata: pl.DataFrame,
    *,
    config: "JaxParallelBenchmarkConfig",
    baselines: dict[tuple[str, int], float],
    k_values: tuple[int, ...],
) -> list[ParallelBenchmarkResult]:
    with jax.default_device(config.devices[0]):
        op = JaxParallelOperator.from_hdf5(
            linarg_h5_path,
            mesh=config.mesh,
            block_metadata=linarg_block_metadata,
            backend=config.backend,
        )
    variant_inputs, sample_inputs = _benchmark_inputs(op.shape, k_values=k_values)
    graph_bytes_by_device = _graph_bytes_by_device(op)
    resident_graph_bytes = sum(graph_bytes_by_device.values())
    max_device_graph_bytes = max(graph_bytes_by_device.values(), default=0)
    results = []
    for k, matrix in variant_inputs.items():
        with jax.default_device(config.devices[0]):
            jax_matrix = jnp.asarray(matrix)
        # The operator owns cached device-local range executables. Wrapping the
        # bound method in another JIT would capture all graph arrays as constants
        # and defeat the placement contract this benchmark is intended to test.
        runtime = _time_call(lambda matrix=jax_matrix, op=op: op.matmat(matrix), block_until_ready=True)
        results.append(
            ParallelBenchmarkResult(
                _result_operator_name(config),
                "matmat",
                k,
                runtime,
                runtime / baselines[("matmat", k)],
                resident_graph_bytes,
                max_device_graph_bytes,
                requested_backend=config.backend,
                resolved_backend=op.blocks[0].backend,
                **_ffi_build_metadata(),
            )
        )
    for k, matrix in sample_inputs.items():
        with jax.default_device(config.devices[0]):
            jax_matrix = jnp.asarray(matrix)
        # `JaxParallelOperator.rmatmat` uses cached per-range JIT functions and
        # concatenates exact-size variant outputs. Benchmark that path directly:
        # an outer JIT would collapse the host-side launch pattern we are trying
        # to compare against ParallelOperator's worker writes.
        runtime = _time_call(lambda matrix=jax_matrix, op=op: op.rmatmat(matrix), block_until_ready=True)
        results.append(
            ParallelBenchmarkResult(
                _result_operator_name(config),
                "rmatmat",
                k,
                runtime,
                runtime / baselines[("rmatmat", k)],
                resident_graph_bytes,
                max_device_graph_bytes,
                requested_backend=config.backend,
                resolved_backend=op.blocks[0].backend,
                **_ffi_build_metadata(),
            )
        )
    return results


def _promotion_backends(representation: Representation) -> tuple[Backend, ...]:
    backends = [Backend.PURE_JAX]
    if representation is Representation.PACKED_CANDIDATE:
        ffi_available = ffi_cpu.is_ffi_cpu_packed_available()
    else:
        ffi_available = ffi_cpu.is_ffi_cpu_available()
    if ffi_available:
        backends.append(Backend.FFI_CPU)
    return tuple(backends)


def _promotion_device_counts(block_metadata: pl.DataFrame) -> tuple[int, ...]:
    maximum = min(len(_devices_for_backend("cpu")), block_metadata.height)
    if maximum < 1:
        pytest.skip("promotion products require a CPU JAX device")
    return (1, 2) if maximum >= 2 else (1,)


def _time_promotion_numpy_products(
    path: Path,
    block_metadata: pl.DataFrame,
    *,
    k_values: tuple[int, ...],
    num_processes: int,
) -> list[_PromotionProductMeasurement]:
    start = time.perf_counter()
    with ParallelOperator.from_hdf5(
        str(path),
        num_processes=num_processes,
        max_num_traits=max(k_values),
        block_metadata=block_metadata,
    ) as operator:
        construction_seconds = time.perf_counter() - start
        variant_inputs, sample_inputs = _benchmark_inputs(operator.shape, k_values=k_values)
        measurements = []
        for operation, inputs in (("matmat", variant_inputs), ("rmatmat", sample_inputs)):
            call_method = getattr(operator, operation)
            for k, values in inputs.items():
                start = time.perf_counter()
                output = call_method(values)
                first_seconds = time.perf_counter() - start
                warm_seconds = _time_call(lambda call_method=call_method, values=values: call_method(values))
                numeric_passed = bool(np.all(np.isfinite(output)))
                common: dict[str, Any] = dict(
                    operation=operation,
                    k=k,
                    dtype=str(values.dtype),
                    requested_backend=None,
                    resolved_backend=None,
                    device_count=num_processes,
                    metrics=PerformanceMetrics(),
                    numeric_passed=numeric_passed,
                )
                measurements.extend(
                    (
                        _PromotionProductMeasurement(
                            phase=TimingPhase.CONSTRUCTION.value,
                            seconds=construction_seconds,
                            null_reason=None,
                            **common,
                        ),
                        _PromotionProductMeasurement(
                            phase=TimingPhase.LOWERING.value,
                            seconds=None,
                            null_reason="NumPy/Cython does not expose a JAX lowering phase",
                            **common,
                        ),
                        _PromotionProductMeasurement(
                            phase=TimingPhase.COMPILATION.value,
                            seconds=None,
                            null_reason="NumPy/Cython does not expose a separate compilation phase",
                            **common,
                        ),
                        _PromotionProductMeasurement(
                            phase=TimingPhase.FIRST_EXECUTION.value,
                            seconds=first_seconds,
                            null_reason=None,
                            **common,
                        ),
                        _PromotionProductMeasurement(
                            phase=TimingPhase.WARM_EXECUTION.value,
                            seconds=warm_seconds,
                            null_reason=None,
                            **common,
                        ),
                    )
                )
    return measurements


def _time_promotion_exact_products(
    path: Path,
    block_metadata: pl.DataFrame,
    *,
    k_values: tuple[int, ...],
) -> list[_PromotionProductMeasurement]:
    measurements = []
    cpu_devices = tuple(_devices_for_backend("cpu"))
    for backend in _promotion_backends(Representation.RETAINED_EXACT_RAGGED):
        for device_count in _promotion_device_counts(block_metadata):
            devices = cpu_devices[:device_count]
            mesh = Mesh(np.asarray(devices), ("blocks",))
            with jax.default_device(devices[0]):
                start = time.perf_counter()
                operator = JaxParallelOperator.from_hdf5(
                    path,
                    mesh=mesh,
                    block_metadata=block_metadata,
                    backend=backend,
                )
                construction_seconds = time.perf_counter() - start
            variant_inputs, sample_inputs = _benchmark_inputs(operator.shape, k_values=k_values)
            resolved_backend = operator.blocks[0].backend.value
            graph_bytes = _graph_bytes_by_device(operator)
            metrics = PerformanceMetrics(
                resident_graph_bytes=sum(graph_bytes.values()),
                max_device_graph_bytes=max(graph_bytes.values(), default=0),
            )
            for operation, inputs in (("matmat", variant_inputs), ("rmatmat", sample_inputs)):
                call_method = getattr(operator, operation)
                for k, host_values in inputs.items():
                    with jax.default_device(devices[0]):
                        values = jnp.asarray(host_values)
                    start = time.perf_counter()
                    output = call_method(values)
                    output.block_until_ready()
                    first_seconds = time.perf_counter() - start
                    warm_seconds = _time_call(
                        lambda call_method=call_method, values=values: call_method(values),
                        block_until_ready=True,
                    )
                    numeric_passed = bool(np.all(np.isfinite(np.asarray(output))))
                    common: dict[str, Any] = dict(
                        operation=operation,
                        k=k,
                        dtype=str(values.dtype),
                        requested_backend=backend.value,
                        resolved_backend=resolved_backend,
                        device_count=device_count,
                        metrics=metrics,
                        numeric_passed=numeric_passed,
                    )
                    measurements.extend(
                        (
                            _PromotionProductMeasurement(
                                phase=TimingPhase.CONSTRUCTION.value,
                                seconds=construction_seconds,
                                null_reason=None,
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.LOWERING.value,
                                seconds=None,
                                null_reason="exact-ragged dispatch lowers cached per-range programs during first call",
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.COMPILATION.value,
                                seconds=None,
                                null_reason="exact-ragged per-range compilation is included in first execution",
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.FIRST_EXECUTION.value,
                                seconds=first_seconds,
                                null_reason=None,
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.WARM_EXECUTION.value,
                                seconds=warm_seconds,
                                null_reason=None,
                                **common,
                            ),
                        )
                    )
    return measurements


def _time_promotion_packed_products(
    path: Path,
    block_metadata: pl.DataFrame,
    *,
    k_values: tuple[int, ...],
) -> list[_PromotionProductMeasurement]:
    measurements = []
    cpu_devices = tuple(_devices_for_backend("cpu"))
    block_names = tuple(block_metadata.get_column("block_name").to_list())
    max_padding_ratio = None if path.parent.name == "testdata" else 1.25
    for backend in _promotion_backends(Representation.PACKED_CANDIDATE):
        for device_count in _promotion_device_counts(block_metadata):
            devices = cpu_devices[:device_count]
            mesh = Mesh(np.asarray(devices), ("graph",))
            start = time.perf_counter()
            packed = _packed_from_hdf5(
                path,
                block_names,
                mesh=mesh,
                backend=backend,
                max_padding_ratio=max_padding_ratio,
            )
            construction_seconds = time.perf_counter() - start
            operator = packed.operator
            graph_bytes = _graph_bytes_by_device(operator)
            diagnostics = packed.diagnostics
            for operation, logical_size in (("matmat", operator.n_variants), ("rmatmat", operator.n_samples)):
                function = lineararg_matmat if operation == "matmat" else lineararg_rmatmat
                output_size = operator.n_samples if operation == "matmat" else operator.n_variants
                for k in k_values:
                    rng = np.random.default_rng(20260506 + k + (0 if operation == "matmat" else 1000))
                    values = jnp.asarray(rng.normal(size=(logical_size, k)).astype(np.float32))
                    start = time.perf_counter()
                    lowered = jax.jit(function).lower(operator, values)
                    lowering_seconds = time.perf_counter() - start
                    stablehlo = lowered.compiler_ir("stablehlo")
                    start = time.perf_counter()
                    compiled = lowered.compile()
                    compilation_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    output = compiled(operator, values)
                    output.block_until_ready()
                    first_seconds = time.perf_counter() - start
                    warm_seconds = _time_call(
                        lambda compiled=compiled, operator=operator, values=values: compiled(operator, values),
                        block_until_ready=True,
                    )
                    closed_jaxpr = jax.make_jaxpr(function)(operator, values)
                    metrics = PerformanceMetrics(
                        canonical_graph_bytes=int(diagnostics.canonical_graph_bytes),
                        padded_graph_bytes=int(diagnostics.padded_graph_bytes),
                        descriptor_bytes=int(diagnostics.descriptor_bytes),
                        resident_graph_bytes=sum(graph_bytes.values()),
                        max_device_graph_bytes=max(graph_bytes.values(), default=0),
                        staging_bytes=int(diagnostics.staging_bytes),
                        component_count=int(diagnostics.component_count),
                        pytree_leaf_count=int(diagnostics.pytree_leaf_count),
                        graph_constant_bytes=_closed_jaxpr_array_constant_bytes(closed_jaxpr),
                        graph_operand_count=_stablehlo_graph_operand_count(stablehlo),
                        stablehlo_bytes=len(str(stablehlo).encode("utf-8")),
                        stablehlo_operation_count=_stablehlo_operation_count(stablehlo),
                        logical_collective_bytes=(
                            output_size * k * np.dtype(values.dtype).itemsize if device_count > 1 else 0
                        ),
                    )
                    common: dict[str, Any] = dict(
                        operation=operation,
                        k=k,
                        dtype=str(values.dtype),
                        requested_backend=backend.value,
                        resolved_backend=operator.backend.value,
                        device_count=device_count,
                        metrics=metrics,
                        numeric_passed=bool(np.all(np.isfinite(np.asarray(output)))),
                    )
                    measurements.extend(
                        (
                            _PromotionProductMeasurement(
                                phase=TimingPhase.CONSTRUCTION.value,
                                seconds=construction_seconds,
                                null_reason=None,
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.LOWERING.value,
                                seconds=lowering_seconds,
                                null_reason=None,
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.COMPILATION.value,
                                seconds=compilation_seconds,
                                null_reason=None,
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.FIRST_EXECUTION.value,
                                seconds=first_seconds,
                                null_reason=None,
                                **common,
                            ),
                            _PromotionProductMeasurement(
                                phase=TimingPhase.WARM_EXECUTION.value,
                                seconds=warm_seconds,
                                null_reason=None,
                                **common,
                            ),
                        )
                    )
    return measurements


def _run_promotion_product_child(
    request: pytest.FixtureRequest,
    *,
    representation: Representation,
    path: Path,
    block_metadata: pl.DataFrame,
    k_values: tuple[int, ...],
    parallel_processes: int,
) -> None:
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")
    output_path = request.config.getoption("--jax-promotion-output")
    if output_path is None:
        pytest.fail("promotion child requires --jax-promotion-output PATH")
    if representation is Representation.NUMPY_CYTHON:
        measurements = _time_promotion_numpy_products(
            path,
            block_metadata,
            k_values=k_values,
            num_processes=min(parallel_processes, block_metadata.height),
        )
    elif representation is Representation.RETAINED_EXACT_RAGGED:
        measurements = _time_promotion_exact_products(path, block_metadata, k_values=k_values)
    else:
        measurements = _time_promotion_packed_products(path, block_metadata, k_values=k_values)

    fingerprint = compute_dataset_fingerprint(path)
    candidate = git_commit()
    dirty = is_git_dirty()
    platform_label = request.config.getoption("--platform-label")
    cache_policy = request.config.getoption("--cache-policy")
    records = tuple(
        make_record(
            platform_label=platform_label,
            cache_label=cache_policy,
            candidate_commit=candidate,
            dataset=fingerprint,
            representation=representation.value,
            operation=measurement.operation,
            phase=measurement.phase,
            workload_size=measurement.k,
            dtype=measurement.dtype,
            requested_backend=measurement.requested_backend,
            resolved_backend=measurement.resolved_backend,
            device_count=measurement.device_count,
            timed=TimedPhase(
                phase=measurement.phase,
                seconds=measurement.seconds,
                null_reason=measurement.null_reason,
            ),
            metric=measurement.metrics,
            numeric_passed=measurement.numeric_passed,
            notes=json.dumps(
                {
                    "ffi_build_config": _ffi_build_metadata(),
                    "historical_ir_counterexample": "genoio@c271a9a",
                },
                sort_keys=True,
            ),
            dirty_worktree=dirty,
        )
        for measurement in measurements
    )
    evidence = build_promotion_evidence(
        cache_label=cache_policy,
        platform_label=platform_label,
        records=records,
        candidate_commit=candidate,
        dataset=fingerprint,
    )
    write_evidence_fragment(output_path, evidence)


def _benchmark_inputs(
    shape: tuple[int, int],
    *,
    k_values: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(20260506)
    n_samples, n_variants = shape
    variant_inputs = {k: rng.normal(size=(n_variants, k)).astype(np.float32) for k in k_values}
    sample_inputs = {k: rng.normal(size=(n_samples, k)).astype(np.float32) for k in k_values}
    return variant_inputs, sample_inputs


def _graph_bytes_by_device(op: JaxParallelOperator | _PackedJaxLinearARG) -> dict[str, int]:
    resident_bytes: dict[str, int] = {}
    if isinstance(op, _PackedJaxLinearARG):
        arrays = tuple(getattr(op, name) for name in GRAPH_FIELD_NAMES)
    else:
        arrays = tuple(
            leaf for block in op.blocks for leaf in jax.tree_util.tree_leaves(block) if isinstance(leaf, jax.Array)
        )
    for array in arrays:
        array.block_until_ready()
        for shard in array.addressable_shards:
            shard.data.block_until_ready()
            device = str(shard.device)
            resident_bytes[device] = resident_bytes.get(device, 0) + int(shard.data.on_device_size_in_bytes())
    return resident_bytes


def _packed_fields_have_expected_residency(op: _PackedJaxLinearARG) -> bool:
    for name in PACKED_COMPONENT_NAMES:
        array = getattr(op, name)
        array.block_until_ready()
        if not isinstance(array.sharding, NamedSharding):
            return False
        if array.sharding.mesh.axis_names != ("graph",) or array.sharding.spec[0] != "graph":
            return False
        expected_indices = array.sharding.addressable_devices_indices_map(array.shape)
        observed_shards = {shard.device: shard for shard in array.addressable_shards}
        if set(observed_shards) != set(expected_indices):
            return False
        for device, expected_index in expected_indices.items():
            shard = observed_shards[device]
            if shard.index != expected_index or shard.data.devices() != {device}:
                return False
            if shard.data.on_device_size_in_bytes() != shard.data.nbytes:
                return False
    return True


@dataclass(frozen=True)
class JaxParallelBenchmarkConfig:
    name: str
    backend: Backend
    devices: tuple[jax.Device, ...]

    @property
    def mesh(self) -> Mesh:
        return Mesh(np.asarray(self.devices), ("blocks",))


def _jax_parallel_configs(linarg_block_metadata: pl.DataFrame) -> list[JaxParallelBenchmarkConfig]:
    cpu_devices = tuple(_devices_for_backend("cpu"))
    if not cpu_devices:
        pytest.skip("JAX parallel benchmark requires at least one JAX device")

    configs = []
    configs.extend(_configs_for_backend(Backend.PURE_JAX, "pure_jax_cpu", cpu_devices, linarg_block_metadata))
    if ffi_cpu.is_ffi_cpu_available():
        configs.extend(_configs_for_backend(Backend.FFI_CPU, "ffi_cpu", cpu_devices, linarg_block_metadata))
    return configs


def _configs_for_backend(
    backend: Backend,
    backend_name: str,
    devices: tuple[jax.Device, ...],
    linarg_block_metadata: pl.DataFrame,
) -> list[JaxParallelBenchmarkConfig]:
    if not devices:
        return []

    configs = [
        JaxParallelBenchmarkConfig(
            f"jax_parallel_{backend_name}_1_device",
            backend,
            devices[:1],
        )
    ]
    sharded_device_count = min(len(devices), linarg_block_metadata.height)
    if sharded_device_count > 1:
        configs.append(
            JaxParallelBenchmarkConfig(
                f"jax_parallel_{backend_name}_{sharded_device_count}_device_sharded",
                backend,
                devices[:sharded_device_count],
            )
        )
    return configs


def _devices_for_backend(backend: str) -> list[jax.Device]:
    try:
        return list(jax.devices(backend))
    except RuntimeError:
        return []


def _result_operator_name(config: JaxParallelBenchmarkConfig) -> str:
    return config.name


def _time_call(call: Callable[[], Any], *, block_until_ready: bool = False) -> float:
    for _ in range(WARMUP_ITERATIONS):
        _call_repeated(call, repetitions=1, block_until_ready=block_until_ready)

    repetitions = _calibrate_repetitions(call, block_until_ready=block_until_ready)

    timings = []
    for _ in range(TIMED_ITERATIONS):
        start = time.perf_counter()
        _call_repeated(call, repetitions=repetitions, block_until_ready=block_until_ready)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings) / repetitions


def _calibrate_repetitions(call: Callable[[], Any], *, block_until_ready: bool) -> int:
    repetitions = 1
    while True:
        start = time.perf_counter()
        _call_repeated(call, repetitions=repetitions, block_until_ready=block_until_ready)
        elapsed = time.perf_counter() - start
        if elapsed >= MIN_SAMPLE_SECONDS:
            return repetitions
        repetitions *= 2


def _call_repeated(call: Callable[[], Any], *, repetitions: int, block_until_ready: bool) -> None:
    for _ in range(repetitions):
        result = call()
        if block_until_ready:
            result.block_until_ready()


def _print_results(results: list[ParallelBenchmarkResult]) -> None:
    print("\n" + _format_results_table(results))


def _format_results_table(results: list[ParallelBenchmarkResult]) -> str:
    lines = [
        "| operator | operation | k | median seconds | ratio to ParallelOperator "
        "| requested backend | resolved backend | FFI built | exact FFI | packed FFI | BLAS enabled | BLAS backend "
        "| native tuning "
        "| canonical graph MiB | padded graph MiB | descriptor MiB | padding ratio "
        "| resident graph MiB | max device graph MiB | staging MiB | components | PyTree leaves "
        "| graph constant bytes | graph operands | StableHLO ops |",
        "|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: (item.operation, item.operator, item.k)):
        ratio = "" if result.ratio_to_parallel_operator is None else f"{result.ratio_to_parallel_operator:.3f}"
        k = "" if result.k is None else str(result.k)
        canonical_mib = _format_mib(result.canonical_graph_bytes)
        padded_mib = _format_mib(result.padded_graph_bytes)
        descriptor_mib = _format_mib(result.descriptor_bytes)
        padding_ratio = "" if result.padding_ratio is None else f"{result.padding_ratio:.3f}"
        resident_mib = _format_mib(result.resident_graph_bytes)
        max_device_mib = _format_mib(result.max_device_graph_bytes)
        staging_mib = _format_mib(result.staging_bytes)
        component_count = "" if result.component_count is None else str(result.component_count)
        pytree_leaf_count = "" if result.pytree_leaf_count is None else str(result.pytree_leaf_count)
        graph_constant_bytes = "" if result.graph_constant_bytes is None else str(result.graph_constant_bytes)
        graph_operand_count = "" if result.graph_operand_count is None else str(result.graph_operand_count)
        stablehlo_operation_count = (
            "" if result.stablehlo_operation_count is None else str(result.stablehlo_operation_count)
        )
        requested_backend = "" if result.requested_backend is None else result.requested_backend.value
        resolved_backend = "" if result.resolved_backend is None else result.resolved_backend.value
        ffi_cpu_built = "" if result.ffi_cpu_built is None else str(result.ffi_cpu_built)
        ffi_cpu_exact_available = "" if result.ffi_cpu_exact_available is None else str(result.ffi_cpu_exact_available)
        ffi_cpu_packed_available = (
            "" if result.ffi_cpu_packed_available is None else str(result.ffi_cpu_packed_available)
        )
        ffi_cpu_blas_enabled = "" if result.ffi_cpu_blas_enabled is None else str(result.ffi_cpu_blas_enabled)
        ffi_cpu_native_tuning = "" if result.ffi_cpu_native_tuning is None else str(result.ffi_cpu_native_tuning)
        ffi_cpu_blas_backend = "" if result.ffi_cpu_blas_backend is None else result.ffi_cpu_blas_backend
        lines.append(
            f"| {result.operator} | {result.operation} | {k} | "
            f"{result.median_seconds:.6f} | {ratio} | {requested_backend} | {resolved_backend} | "
            f"{ffi_cpu_built} | {ffi_cpu_exact_available} | {ffi_cpu_packed_available} | "
            f"{ffi_cpu_blas_enabled} | {ffi_cpu_blas_backend} | {ffi_cpu_native_tuning} | "
            f"{canonical_mib} | {padded_mib} | {descriptor_mib} | "
            f"{padding_ratio} | {resident_mib} | {max_device_mib} | {staging_mib} | "
            f"{component_count} | {pytree_leaf_count} | {graph_constant_bytes} | "
            f"{graph_operand_count} | {stablehlo_operation_count} |"
        )
    return "\n".join(lines)


def _format_mib(value: int | None) -> str:
    return "" if value is None else f"{value / 2**20:.3f}"
