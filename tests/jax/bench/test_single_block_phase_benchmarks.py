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
from linear_dag.core.solve import (
    add_at,
    spsolve_backward_triangular_matmat,
    spsolve_forward_triangular_matmat,
)

MIN_SAMPLE_SECONDS = 0.005
WARMUP_ITERATIONS = 2
TIMED_ITERATIONS = 9


@dataclass(frozen=True)
class PhaseBenchmarkResult:
    backend: str
    operation: str
    phase: str
    k: int
    median_seconds: float
    ratio_to_cython: float | None


def test_single_block_matmat_phase_benchmark(
    request: pytest.FixtureRequest,
    linarg_h5_path,
    first_block_name,
    linarg_benchmark_k_values: tuple[int, ...],
) -> None:
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")
    if not ffi_cpu.is_ffi_cpu_available():
        pytest.skip("CPU FFI backend is unavailable")

    linarg = LinearARG.read(linarg_h5_path, block=first_block_name, load_metadata=True)
    linarg.calculate_nonunique_indices()
    op = JaxLinearARG.from_lineararg(linarg, backend=Backend.FFI_CPU, dtype=jnp.float32)
    variant_inputs, sample_inputs = _benchmark_inputs(linarg, k_values=linarg_benchmark_k_values)

    results: list[PhaseBenchmarkResult] = []
    for operation, inputs in (("matmat", variant_inputs), ("rmatmat", sample_inputs)):
        cython_results = _time_cython_phases(linarg, operation=operation, inputs=inputs)
        for (phase, k), runtime in cython_results.items():
            results.append(PhaseBenchmarkResult("cython", operation, phase, k, runtime, 1.0))
        results.extend(_time_jax_ffi_phases(op, operation=operation, inputs=inputs, cython_results=cython_results))

    _print_results(results)


def _benchmark_inputs(
    linarg: LinearARG,
    *,
    k_values: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    rng = np.random.default_rng(20260514)
    variant_inputs = {k: rng.normal(size=(linarg.shape[1], k)).astype(np.float32) for k in k_values}
    sample_inputs = {k: rng.normal(size=(linarg.shape[0], k)).astype(np.float32) for k in k_values}
    return variant_inputs, sample_inputs


def _time_cython_phases(
    linarg: LinearARG,
    *,
    operation: str,
    inputs: dict[int, np.ndarray],
) -> dict[tuple[str, int], float]:
    results = {}
    min_index_to_keep = int(linarg.sample_indices[-1])
    solve = spsolve_forward_triangular_matmat if operation == "matmat" else spsolve_backward_triangular_matmat

    for k, matrix in inputs.items():
        if operation == "matmat":
            input_state = _cython_matmat_input_state(linarg, matrix)
            solved_state = input_state.copy(order="F")
            solve_phase = "forward_solve"
            output_phase = "sample_output"

            def output_call(state=solved_state, matrix=matrix):
                return _cython_matmat_output(linarg, state, matrix)

            def input_call(matrix=matrix):
                return _cython_matmat_input_state(linarg, matrix)

        else:
            input_state = _cython_rmatmat_input_state(linarg, matrix)
            solved_state = input_state.copy(order="F")
            solve_phase = "backward_solve"
            output_phase = "variant_output"

            def output_call(state=solved_state, matrix=matrix):
                return _cython_rmatmat_output(linarg, state, matrix)

            def input_call(matrix=matrix):
                return _cython_rmatmat_input_state(linarg, matrix)

        solve(linarg.A, solved_state, linarg.nonunique_indices, min_index_to_keep)
        zero_state = np.zeros((k, int(linarg.num_nonunique_indices)), dtype=np.float32, order="F")

        results[("input_state", k)] = _time_call(input_call)
        results[(solve_phase, k)] = _time_call(
            lambda state=zero_state, solve=solve: solve(
                linarg.A,
                state,
                linarg.nonunique_indices,
                min_index_to_keep,
            )
        )
        results[(output_phase, k)] = _time_call(output_call)

    return results


def _cython_matmat_input_state(linarg: LinearARG, matrix: np.ndarray) -> np.ndarray:
    state = np.zeros((matrix.shape[1], int(linarg.num_nonunique_indices)), dtype=matrix.dtype, order="F")
    if np.any(linarg.flip):
        values = (matrix.T * (-1) ** linarg.flip.reshape(1, -1)).astype(matrix.dtype)
    else:
        values = matrix.T
    add_at(state, linarg.nonunique_indices[linarg.variant_indices], values)
    return state


def _cython_rmatmat_input_state(linarg: LinearARG, matrix: np.ndarray) -> np.ndarray:
    state = np.zeros((matrix.shape[1], int(linarg.num_nonunique_indices)), dtype=matrix.dtype, order="F")
    state[:, linarg.nonunique_indices[linarg.sample_indices]] = matrix.T
    return state


def _cython_matmat_output(linarg: LinearARG, solved_state: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    sample_nonunique_indices = linarg.nonunique_indices[linarg.sample_indices]
    return solved_state[:, sample_nonunique_indices].T + np.sum(matrix[linarg.flip], axis=0)


def _cython_rmatmat_output(linarg: LinearARG, solved_state: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    variant_nonunique_indices = linarg.nonunique_indices[linarg.variant_indices]
    values = solved_state[:, variant_nonunique_indices]
    if np.any(linarg.flip):
        values[:, linarg.flip] = np.sum(matrix, axis=0)[:, np.newaxis] - values[:, linarg.flip]
    return values.T


def _time_jax_ffi_phases(
    op: JaxLinearARG,
    *,
    operation: str,
    inputs: dict[int, np.ndarray],
    cython_results: dict[tuple[str, int], float],
) -> list[PhaseBenchmarkResult]:
    results = []
    for k, matrix in inputs.items():
        jax_matrix = jnp.asarray(matrix)
        if operation == "matmat":
            input_phase = "input_state"
            solve_phase = "forward_solve"
            output_phase = "sample_output"
            solve = ffi_cpu.ffi_cpu_solve_forward

            def input_fun(values, op=op):
                return _jax_matmat_input_state(op, values)

            def output_fun(state, values, op=op):
                return _jax_matmat_output(op, state, values)

        else:
            input_phase = "input_state"
            solve_phase = "backward_solve"
            output_phase = "variant_output"
            solve = ffi_cpu.ffi_cpu_solve_backward

            def input_fun(values, op=op):
                return _jax_rmatmat_input_state(op, values)

            def output_fun(state, values, op=op):
                return _jax_rmatmat_output(op, state, values)

        input_call = jax.jit(input_fun).lower(jax_matrix).compile()
        runtime = _time_call(lambda matrix=jax_matrix, call=input_call: call(matrix), block_until_ready=True)
        results.append(_jax_result(operation, input_phase, k, runtime, cython_results))

        zero_state = jnp.zeros((op.n_nonunique_indices, k), dtype=op.dtype)
        solve_call = (
            jax.jit(
                lambda state, solve=solve, op=op: solve(
                    op.indptr,
                    op.indices,
                    op.data,
                    op.nonunique_indices,
                    op.min_index_to_keep,
                    state,
                ),
                donate_argnums=(0,),
            )
            .lower(zero_state)
            .compile()
        )
        reusable_state = [zero_state]

        def call_solve(call=solve_call, reusable_state=reusable_state):
            reusable_state[0] = call(reusable_state[0])
            return reusable_state[0]

        runtime = _time_call(call_solve, block_until_ready=True)
        results.append(_jax_result(operation, solve_phase, k, runtime, cython_results))

        solved_state = input_call(jax_matrix)
        solved_state = solve_call(solved_state).block_until_ready()
        output_call = jax.jit(output_fun).lower(solved_state, jax_matrix).compile()
        runtime = _time_call(
            lambda state=solved_state, matrix=jax_matrix, call=output_call: call(state, matrix),
            block_until_ready=True,
        )
        results.append(_jax_result(operation, output_phase, k, runtime, cython_results))

    return results


def _jax_result(
    operation: str,
    phase: str,
    k: int,
    runtime: float,
    cython_results: dict[tuple[str, int], float],
) -> PhaseBenchmarkResult:
    return PhaseBenchmarkResult("jax_ffi_cpu", operation, phase, k, runtime, runtime / cython_results[(phase, k)])


def _jax_matmat_input_state(op: JaxLinearARG, matrix: Any) -> jax.Array:
    flip_sign = jnp.where(op.flip, -1, 1).astype(matrix.dtype)
    state = jnp.zeros((op.n_nonunique_indices, matrix.shape[1]), dtype=matrix.dtype)
    variant_nonunique_indices = op.nonunique_indices[op.variant_indices]
    return state.at[variant_nonunique_indices, :].add(matrix * flip_sign[:, None])


def _jax_rmatmat_input_state(op: JaxLinearARG, matrix: Any) -> jax.Array:
    state = jnp.zeros((op.n_nonunique_indices, matrix.shape[1]), dtype=matrix.dtype)
    sample_nonunique_indices = op.nonunique_indices[op.sample_indices]
    return state.at[sample_nonunique_indices, :].set(matrix)


def _jax_matmat_output(op: JaxLinearARG, solved_state: Any, matrix: Any) -> jax.Array:
    sample_nonunique_indices = op.nonunique_indices[op.sample_indices]
    flip_sum = jnp.sum(matrix[op._flipped_variant_indices, :], axis=0)
    return solved_state[sample_nonunique_indices, :] + flip_sum


def _jax_rmatmat_output(op: JaxLinearARG, solved_state: Any, matrix: Any) -> jax.Array:
    variant_nonunique_indices = op.nonunique_indices[op.variant_indices]
    values = solved_state[variant_nonunique_indices, :]
    total = jnp.sum(matrix, axis=0)
    return jnp.where(op.flip[:, None], total[None, :] - values, values)


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


def _print_results(results: list[PhaseBenchmarkResult]) -> None:
    print("\n| backend | operation | phase | k | median seconds | ratio |")
    print("|---|---|---|---:|---:|---:|")
    for result in sorted(results, key=lambda item: (item.operation, item.phase, item.backend, item.k)):
        ratio = "" if result.ratio_to_cython is None else f"{result.ratio_to_cython:.3f}"
        print(
            f"| {result.backend} | {result.operation} | {result.phase} | {result.k} | "
            f"{result.median_seconds:.6f} | {ratio} |"
        )
