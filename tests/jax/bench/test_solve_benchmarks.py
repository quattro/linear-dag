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
from linear_dag.core.solve import spsolve_backward_triangular_matmat, spsolve_forward_triangular_matmat

MIN_SAMPLE_SECONDS = 0.005
WARMUP_ITERATIONS = 2
TIMED_ITERATIONS = 9


@dataclass(frozen=True)
class SolveBenchmarkResult:
    backend: str
    direction: str
    k: int
    median_seconds: float
    ratio_to_cython: float | None


def test_jax_ffi_cpu_solve_benchmark(
    request: pytest.FixtureRequest,
    linarg_h5_path,
    first_block_name,
    linarg_benchmark_k_values: tuple[int, ...],
):
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")
    if not ffi_cpu.is_ffi_cpu_available():
        pytest.skip("CPU FFI backend is unavailable")

    linarg = LinearARG.read(linarg_h5_path, block=first_block_name, load_metadata=True)
    linarg.calculate_nonunique_indices()
    jax_op = JaxLinearARG.from_lineararg(linarg, backend=Backend.FFI_CPU, dtype=jnp.float32)

    results = []
    for direction in ("forward", "backward"):
        cython_results = _time_cython_solves(linarg, direction=direction, k_values=linarg_benchmark_k_values)
        results.extend(
            SolveBenchmarkResult("cython_solve", direction, k, runtime, 1.0) for k, runtime in cython_results.items()
        )
        results.extend(
            _time_ffi_solves(
                jax_op,
                direction=direction,
                k_values=linarg_benchmark_k_values,
                cython_results=cython_results,
            )
        )

    _print_results(results)


def _time_cython_solves(
    linarg: LinearARG,
    *,
    direction: str,
    k_values: tuple[int, ...],
) -> dict[int, float]:
    solve = _cython_solve(direction)
    min_index_to_keep = int(linarg.sample_indices[-1])
    results = {}
    for k in k_values:
        # The zero state stays zero after every solve, so repeated timing can
        # reuse one mutable buffer while still executing the full edge loop.
        state = np.zeros((k, int(linarg.num_nonunique_indices)), dtype=np.float32, order="F")
        results[k] = _time_call(
            lambda state=state, solve=solve: solve(
                linarg.A,
                state,
                linarg.nonunique_indices,
                min_index_to_keep,
            )
        )
    return results


def _time_ffi_solves(
    op: JaxLinearARG,
    *,
    direction: str,
    k_values: tuple[int, ...],
    cython_results: dict[int, float],
) -> list[SolveBenchmarkResult]:
    solve = ffi_cpu.ffi_cpu_solve_forward if direction == "forward" else ffi_cpu.ffi_cpu_solve_backward
    results = []
    for k in k_values:
        state = jnp.zeros((op.n_nonunique_indices, k), dtype=op.dtype)
        compiled = (
            jax.jit(
                lambda values, solve=solve: solve(
                    op.indptr,
                    op.indices,
                    op.data,
                    op.nonunique_indices,
                    op.min_index_to_keep,
                    values,
                ),
                donate_argnums=(0,),
            )
            .lower(state)
            .compile()
        )
        reusable_state = [state]

        def call_solve(compiled=compiled, reusable_state=reusable_state):
            # Donation lets XLA pass the mutable state buffer to the custom call
            # without an immutable-input defensive copy. The zero-state solve
            # returns another reusable zero buffer for the next repetition.
            reusable_state[0] = compiled(reusable_state[0])
            return reusable_state[0]

        runtime = _time_call(call_solve, block_until_ready=True)
        results.append(
            SolveBenchmarkResult(
                "ffi_cpu_solve",
                direction,
                k,
                runtime,
                runtime / cython_results[k],
            )
        )
    return results


def _cython_solve(direction: str) -> Callable[..., None]:
    if direction == "forward":
        return spsolve_forward_triangular_matmat
    if direction == "backward":
        return spsolve_backward_triangular_matmat
    raise ValueError(f"unknown solve direction: {direction}")


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


def _print_results(results: list[SolveBenchmarkResult]) -> None:
    print("\n| backend | direction | k | median seconds | ratio |")
    print("|---|---|---:|---:|---:|")
    for result in sorted(results, key=lambda item: (item.direction, item.backend, item.k)):
        ratio = "" if result.ratio_to_cython is None else f"{result.ratio_to_cython:.3f}"
        print(f"| {result.backend} | {result.direction} | {result.k} | {result.median_seconds:.6f} | {ratio} |")
