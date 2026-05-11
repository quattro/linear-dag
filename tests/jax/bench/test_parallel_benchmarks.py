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

from jax.sharding import Mesh

from linear_dag.core.jaxlinarg import Backend, JaxParallelOperator
from linear_dag.core.parallel_processing import ParallelOperator

K_VALUES = (1, 8, 64)
MIN_SAMPLE_SECONDS = 0.005
WARMUP_ITERATIONS = 2
TIMED_ITERATIONS = 9


@dataclass(frozen=True)
class ParallelBenchmarkResult:
    operator: str
    operation: str
    k: int
    median_seconds: float
    ratio_to_parallel_operator: float | None


def test_jax_parallel_operator_benchmark(request: pytest.FixtureRequest, linarg_h5_path, linarg_block_metadata):
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")

    num_processes = min(2, linarg_block_metadata.height)
    process_results = _time_parallel_operator(
        linarg_h5_path,
        linarg_block_metadata,
        num_processes=num_processes,
    )
    baselines = {(result.operation, result.k): result.median_seconds for result in process_results}
    results = list(process_results)

    for name, mesh in _jax_parallel_meshes(linarg_block_metadata):
        results.extend(
            _time_jax_parallel_operator(
                linarg_h5_path,
                linarg_block_metadata,
                operator_name=name,
                mesh=mesh,
                baselines=baselines,
            )
        )

    _print_results(results)


def _time_parallel_operator(
    linarg_h5_path,
    linarg_block_metadata: pl.DataFrame,
    *,
    num_processes: int,
) -> list[ParallelBenchmarkResult]:
    with ParallelOperator.from_hdf5(
        str(linarg_h5_path),
        num_processes=num_processes,
        max_num_traits=max(K_VALUES),
        block_metadata=linarg_block_metadata,
    ) as op:
        variant_inputs, sample_inputs = _benchmark_inputs(op.shape)
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
    operator_name: str,
    mesh: Mesh,
    baselines: dict[tuple[str, int], float],
) -> list[ParallelBenchmarkResult]:
    op = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=mesh,
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    variant_inputs, sample_inputs = _benchmark_inputs(op.shape)
    results = []
    for k, matrix in variant_inputs.items():
        jax_matrix = jnp.asarray(matrix)
        matmat = jax.jit(lambda values: op.matmat(values)).lower(jax_matrix).compile()
        runtime = _time_call(lambda matrix=jax_matrix, matmat=matmat: matmat(matrix), block_until_ready=True)
        results.append(
            ParallelBenchmarkResult(
                operator_name,
                "matmat",
                k,
                runtime,
                runtime / baselines[("matmat", k)],
            )
        )
    for k, matrix in sample_inputs.items():
        jax_matrix = jnp.asarray(matrix)
        rmatmat = jax.jit(lambda values: op.rmatmat(values)).lower(jax_matrix).compile()
        runtime = _time_call(lambda matrix=jax_matrix, rmatmat=rmatmat: rmatmat(matrix), block_until_ready=True)
        results.append(
            ParallelBenchmarkResult(
                operator_name,
                "rmatmat",
                k,
                runtime,
                runtime / baselines[("rmatmat", k)],
            )
        )
    return results


def _benchmark_inputs(shape: tuple[int, int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(20260506)
    n_samples, n_variants = shape
    variant_inputs = {k: rng.normal(size=(n_variants, k)).astype(np.float32) for k in K_VALUES}
    sample_inputs = {k: rng.normal(size=(n_samples, k)).astype(np.float32) for k in K_VALUES}
    return variant_inputs, sample_inputs


def _jax_parallel_meshes(linarg_block_metadata: pl.DataFrame) -> list[tuple[str, Mesh]]:
    devices = jax.devices()
    if not devices:
        pytest.skip("JAX parallel benchmark requires at least one JAX device")

    meshes = [("jax_parallel_pure_jax_1_device", Mesh(np.asarray(devices[:1]), ("blocks",)))]
    sharded_device_count = min(len(devices), linarg_block_metadata.height)
    if sharded_device_count > 1:
        meshes.append(
            (
                f"jax_parallel_pure_jax_{sharded_device_count}_device_sharded",
                Mesh(np.asarray(devices[:sharded_device_count]), ("blocks",)),
            )
        )
    return meshes


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
    print("\n| operator | operation | k | median seconds | ratio to ParallelOperator |")
    print("|---|---|---:|---:|---:|")
    for result in sorted(results, key=lambda item: (item.operation, item.operator, item.k)):
        ratio = "" if result.ratio_to_parallel_operator is None else f"{result.ratio_to_parallel_operator:.3f}"
        print(f"| {result.operator} | {result.operation} | {result.k} | {result.median_seconds:.6f} | {ratio} |")
