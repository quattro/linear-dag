# pattern: Imperative Shell

from __future__ import annotations

import statistics
import time

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from jax.sharding import Mesh, NamedSharding

from linear_dag.core.jaxlinarg import Backend, JaxParallelOperator
from linear_dag.core.jaxlinarg.ingress import _packed_from_hdf5, _PackedJaxLinearARG
from linear_dag.core.jaxlinarg.kernels import ffi_cpu
from linear_dag.core.jaxlinarg.packing import GRAPH_FIELD_NAMES, PACKED_COMPONENT_NAMES
from linear_dag.core.parallel_processing import ParallelOperator

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
    op = _packed_from_hdf5(
        linarg_h5_path,
        block_names,
        mesh=mesh,
        allow_excess_padding=True,
    )
    construction_seconds = time.perf_counter() - start
    for name in PACKED_COMPONENT_NAMES:
        getattr(op, name).block_until_ready()

    observed_graph_bytes = _graph_bytes_by_device(op)
    if sum(observed_graph_bytes.values()) != sum(op.diagnostics.final_graph_bytes_by_device):
        pytest.fail("packed ingress diagnostics do not match observed graph residency")
    return _packed_memory_result(
        op.diagnostics,
        operator=f"packed_jax_lineararg_{num_devices}_device",
        construction_seconds=construction_seconds,
        resident_devices_valid=_packed_fields_have_expected_residency(op),
    )


def _packed_memory_result(
    diagnostics: Any,
    *,
    operator: str,
    construction_seconds: float,
    resident_devices_valid: bool,
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
    )


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
            )
        )
    return results


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
        "| canonical graph MiB | padded graph MiB | descriptor MiB | padding ratio "
        "| resident graph MiB | max device graph MiB | staging MiB | components | PyTree leaves |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
        lines.append(
            f"| {result.operator} | {result.operation} | {k} | "
            f"{result.median_seconds:.6f} | {ratio} | {canonical_mib} | {padded_mib} | {descriptor_mib} | "
            f"{padding_ratio} | {resident_mib} | {max_device_mib} | {staging_mib} | "
            f"{component_count} | {pytree_leaf_count} |"
        )
    return "\n".join(lines)


def _format_mib(value: int | None) -> str:
    return "" if value is None else f"{value / 2**20:.3f}"
