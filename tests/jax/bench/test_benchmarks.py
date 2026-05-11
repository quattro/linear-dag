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
from linear_dag.core.jaxlinarg.kernels import ffi_cpu, pallas_gpu
from linear_dag.core.lineararg import LinearARG

K_VALUES = (1, 8, 64)
WARMUP_ITERATIONS = 1
TIMED_ITERATIONS = 3


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    k: int
    median_seconds: float
    ratio_to_cython: float | None


def test_jax_lineararg_benchmark_gates(request: pytest.FixtureRequest, linarg_h5_path, first_block_name):
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")

    linarg = LinearARG.read(linarg_h5_path, block=first_block_name, load_metadata=True)
    inputs = _benchmark_inputs(linarg)

    cython_results = {k: _time_call(lambda matrix=matrix: linarg._matmat(matrix)) for k, matrix in inputs.items()}
    results = [BenchmarkResult("cython", k, runtime, 1.0) for k, runtime in cython_results.items()]

    pure_jax_results = _time_backend(linarg, Backend.PURE_JAX, inputs, cython_results)
    results.extend(pure_jax_results)
    _assert_ratio(
        pure_jax_results,
        backend="pure_jax",
        k=1,
        threshold=2.0,
        criterion="jaxlinarg.AC8.1",
    )

    if ffi_cpu.is_ffi_cpu_available():
        ffi_results = _time_backend(linarg, Backend.FFI_CPU, inputs, cython_results)
        results.extend(ffi_results)
        _assert_ratio(
            ffi_results,
            backend="ffi_cpu",
            k=1,
            threshold=2.0,
            criterion="jaxlinarg.AC8.2",
        )

    if pallas_gpu.is_pallas_gpu_available():
        pallas_results = _time_backend(linarg, Backend.PALLAS_GPU, inputs, cython_results)
        results.extend(pallas_results)
        pure_jax_cpu_results = _time_pure_jax_cpu(linarg, inputs)
        results.extend(pure_jax_cpu_results)
        _assert_gpu_speedup(pallas_results, pure_jax_cpu_results, k=8, criterion="jaxlinarg.AC7.1")
        _assert_gpu_speedup(pallas_results, pure_jax_cpu_results, k=64, criterion="jaxlinarg.AC7.2")

    _print_results(results)


def _benchmark_inputs(linarg: LinearARG) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(20260506)
    return {k: rng.normal(size=(linarg.shape[1], k)).astype(np.float32) for k in K_VALUES}


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


def _time_pure_jax_cpu(linarg: LinearARG, inputs: dict[int, np.ndarray]) -> list[BenchmarkResult]:
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        pytest.skip("GPU benchmark requires a CPU device for PURE_JAX_CPU baseline")
    with jax.default_device(cpu_devices[0]):
        op = JaxLinearARG.from_lineararg(linarg, backend=Backend.PURE_JAX, dtype=jnp.float32)
        results = []
        for k, matrix in inputs.items():
            jax_matrix = jnp.asarray(matrix)
            matmat = jax.jit(lambda values: op.matmat(values)).lower(jax_matrix).compile()
            results.append(
                BenchmarkResult(
                    "pure_jax_cpu",
                    k,
                    _time_call(lambda matrix=jax_matrix, matmat=matmat: matmat(matrix), block_until_ready=True),
                    None,
                )
            )
        return results


def _time_call(call: Callable[[], Any], *, block_until_ready: bool = False) -> float:
    for _ in range(WARMUP_ITERATIONS):
        result = call()
        if block_until_ready:
            result.block_until_ready()

    timings = []
    for _ in range(TIMED_ITERATIONS):
        start = time.perf_counter()
        result = call()
        if block_until_ready:
            result.block_until_ready()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


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


def _assert_gpu_speedup(
    pallas_results: list[BenchmarkResult],
    pure_jax_cpu_results: list[BenchmarkResult],
    *,
    k: int,
    criterion: str,
) -> None:
    pallas = _find_result(pallas_results, backend="pallas_gpu", k=k)
    pure_jax_cpu = _find_result(pure_jax_cpu_results, backend="pure_jax_cpu", k=k)
    ratio = pallas.median_seconds / pure_jax_cpu.median_seconds
    assert ratio < 1.0, f"{criterion} failed: pallas_gpu / pure_jax_cpu at k={k} was " f"{ratio:.3f}, expected < 1.000"


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
