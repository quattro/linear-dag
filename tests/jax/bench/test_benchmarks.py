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
import pytest

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.kernels import ffi_cpu
from linear_dag.core.lineararg import LinearARG

MIN_SAMPLE_SECONDS = 0.005
WARMUP_ITERATIONS = 2
TIMED_ITERATIONS = 9


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    k: int
    median_seconds: float
    ratio_to_cython: float | None


def test_jax_lineararg_benchmark_gates(
    request: pytest.FixtureRequest,
    linarg_h5_path,
    first_block_name,
    linarg_benchmark_k_values: tuple[int, ...],
):
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")

    linarg = LinearARG.read(linarg_h5_path, block=first_block_name, load_metadata=True)
    inputs = _benchmark_inputs(linarg, k_values=linarg_benchmark_k_values)

    cython_results = {k: _time_call(lambda matrix=matrix: linarg._matmat(matrix)) for k, matrix in inputs.items()}
    results = [BenchmarkResult("cython", k, runtime, 1.0) for k, runtime in cython_results.items()]

    pure_jax_results = _time_backend(linarg, Backend.PURE_JAX, inputs, cython_results)
    results.extend(pure_jax_results)
    gate_checks = []
    if 1 in linarg_benchmark_k_values:
        gate_checks.append(
            lambda: _assert_ratio(
                pure_jax_results,
                backend="pure_jax",
                k=1,
                threshold=2.0,
                criterion="jaxlinarg.AC8.1",
            )
        )

    if ffi_cpu.is_ffi_cpu_available():
        ffi_results = _time_backend(linarg, Backend.FFI_CPU, inputs, cython_results)
        results.extend(ffi_results)
        if 1 in linarg_benchmark_k_values:
            gate_checks.append(
                lambda: _assert_ratio(
                    ffi_results,
                    backend="ffi_cpu",
                    k=1,
                    threshold=2.5,
                    criterion="jaxlinarg.AC8.2",
                )
            )
        large_k = max(linarg_benchmark_k_values)
        if large_k >= 64 and cython_results[large_k] >= 0.01:
            # Tiny fixtures run in microseconds, where framework overhead is
            # larger than the solve. Only enforce the large-k FFI speed gate
            # for workloads big enough to measure the BLAS-backed kernel.
            gate_checks.append(
                lambda k=large_k: _assert_ratio(
                    ffi_results,
                    backend="ffi_cpu",
                    k=k,
                    threshold=1.0,
                    criterion="jaxlinarg.AC8.3",
                )
            )

    _print_results(results)
    for gate_check in gate_checks:
        gate_check()


def _benchmark_inputs(linarg: LinearARG, *, k_values: tuple[int, ...]) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(20260506)
    return {k: rng.normal(size=(linarg.shape[1], k)).astype(np.float32) for k in k_values}


def _time_backend(
    linarg: LinearARG,
    backend: Backend,
    inputs: dict[int, np.ndarray],
    cython_results: dict[int, float],
) -> list[BenchmarkResult]:
    op = JaxLinearARG.from_lineararg(linarg, backend=backend, dtype=jnp.float32)
    results = []
    for k, matrix in inputs.items():
        jax_matrix = jnp.asarray(matrix)
        matmat = jax.jit(lambda values: op.matmat(values)).lower(jax_matrix).compile()
        runtime = _time_call(lambda matrix=jax_matrix, matmat=matmat: matmat(matrix), block_until_ready=True)
        results.append(
            BenchmarkResult(
                str(op.backend),
                k,
                runtime,
                runtime / cython_results[k],
            )
        )
    return results


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


def _assert_ratio(
    results: list[BenchmarkResult],
    *,
    backend: str,
    k: int,
    threshold: float,
    criterion: str,
) -> None:
    result = _find_result(results, backend=backend, k=k)
    assert result.ratio_to_cython is not None
    assert result.ratio_to_cython <= threshold, (
        f"{criterion} failed: {backend} / cython at k={k} was "
        f"{result.ratio_to_cython:.3f}, expected <= {threshold:.3f}"
    )


def _find_result(results: list[BenchmarkResult], *, backend: str, k: int) -> BenchmarkResult:
    for result in results:
        if result.backend == backend and result.k == k:
            return result
    raise AssertionError(f"missing benchmark result for backend={backend!r}, k={k}")


def _print_results(results: list[BenchmarkResult]) -> None:
    print("\n| backend | k | median seconds | ratio |")
    print("|---|---:|---:|---:|")
    for result in sorted(results, key=lambda item: (item.backend, item.k)):
        ratio = "" if result.ratio_to_cython is None else f"{result.ratio_to_cython:.3f}"
        print(f"| {result.backend} | {result.k} | {result.median_seconds:.6f} | {ratio} |")
