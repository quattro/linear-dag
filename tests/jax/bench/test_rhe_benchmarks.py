# pattern: Mixed (unavoidable)
# Reason: Ratio calculation is pure, while the opt-in benchmark loads real
# operators, coordinates worker/device execution, and reports wall-clock time.

from __future__ import annotations

import json
import statistics
import time

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np
import polars as pl
import pytest

from jax.sharding import Mesh

from linear_dag.association._heritability_jax import randomized_haseman_elston as randomized_haseman_elston_jax
from linear_dag.association.heritability import randomized_haseman_elston
from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxParallelOperator
from linear_dag.core.jaxlinarg.ingress import _PackedJaxLinearARG
from linear_dag.core.lineararg import list_iids
from linear_dag.core.parallel_processing import GRMOperator
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

_PHENO_COLS = ["phenotype_1", "phenotype_2"]
_COVAR_COLS = ["intercept", "covariate"]
_SEED = 20260522
_WARMUP_ITERATIONS = 1
_TIMED_ITERATIONS = 3
_CYTHON_MAX_NUM_TRAITS = 8


@dataclass(frozen=True)
class _RheBenchmarkResult:
    backend: str
    phase: str
    num_matvecs: int
    seconds: float
    ratio_to_numpy_cython: float | None = None
    execution_units: int = 1
    dtype: str = ""


def test_attach_baseline_ratios_matches_phase_and_probe_count() -> None:
    results = [
        _RheBenchmarkResult("numpy_cython", "construction", 4, 2.0),
        _RheBenchmarkResult("jax_auto", "construction", 4, 4.0),
        _RheBenchmarkResult("numpy_cython", "first_execution", 4, 3.0),
        _RheBenchmarkResult("jax_auto", "first_execution", 4, 1.5),
        _RheBenchmarkResult("numpy_cython", "warm_execution", 4, 2.5),
        _RheBenchmarkResult("jax_auto", "warm_execution", 4, 1.0),
    ]

    observed = _attach_baseline_ratios(results)

    assert [result.ratio_to_numpy_cython for result in observed] == [1.0, 2.0, 1.0, 0.5, 1.0, 0.4]


def test_rhe_phase_results_separate_construction_first_and_warm() -> None:
    observed = _phase_results(
        "jax_auto(pure_jax)",
        num_matvecs=4,
        construction_seconds=0.1,
        first_seconds=0.2,
        warm_seconds=0.05,
        execution_units=2,
        dtype="float32",
    )

    assert [result.phase for result in observed] == ["construction", "first_execution", "warm_execution"]
    assert [result.seconds for result in observed] == [0.1, 0.2, 0.05]


def test_rhe_backend_benchmark(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_parallel_processes: int,
    rhe_benchmark_num_matvecs: tuple[int, ...],
) -> None:
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")

    data = _benchmark_data(linarg_h5_path)
    _validate_probe_counts(rhe_benchmark_num_matvecs, data.height)

    cython_results, cython_estimates = _time_numpy_cython_rhe(
        linarg_h5_path,
        linarg_block_metadata,
        data=data,
        num_processes=min(linarg_parallel_processes, linarg_block_metadata.height),
        num_matvecs_values=rhe_benchmark_num_matvecs,
    )
    jax_results, jax_estimates = _time_jax_rhe(
        linarg_h5_path,
        linarg_block_metadata,
        data=data,
        requested_devices=linarg_parallel_processes,
        num_matvecs_values=rhe_benchmark_num_matvecs,
    )

    _assert_estimates_match(cython_estimates, jax_estimates)
    _print_results(_attach_baseline_ratios([*cython_results, *jax_results]))


def test_promotion_numpy_cython_rhe_child(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_parallel_processes: int,
    rhe_benchmark_num_matvecs: tuple[int, ...],
) -> None:
    _run_promotion_rhe_child(
        request,
        representation=Representation.NUMPY_CYTHON,
        path=linarg_h5_path,
        block_metadata=linarg_block_metadata,
        parallel_processes=linarg_parallel_processes,
        num_matvecs_values=rhe_benchmark_num_matvecs,
    )


def test_promotion_exact_rhe_child(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_parallel_processes: int,
    rhe_benchmark_num_matvecs: tuple[int, ...],
) -> None:
    _run_promotion_rhe_child(
        request,
        representation=Representation.RETAINED_EXACT_RAGGED,
        path=linarg_h5_path,
        block_metadata=linarg_block_metadata,
        parallel_processes=linarg_parallel_processes,
        num_matvecs_values=rhe_benchmark_num_matvecs,
    )


def test_promotion_packed_rhe_child(
    request: pytest.FixtureRequest,
    linarg_h5_path: Path,
    linarg_block_metadata: pl.DataFrame,
    linarg_parallel_processes: int,
    rhe_benchmark_num_matvecs: tuple[int, ...],
) -> None:
    _run_promotion_rhe_child(
        request,
        representation=Representation.PACKED_CANDIDATE,
        path=linarg_h5_path,
        block_metadata=linarg_block_metadata,
        parallel_processes=linarg_parallel_processes,
        num_matvecs_values=rhe_benchmark_num_matvecs,
    )


def _attach_baseline_ratios(results: list[_RheBenchmarkResult]) -> list[_RheBenchmarkResult]:
    baselines = {
        (result.phase, result.num_matvecs): result.seconds for result in results if result.backend == "numpy_cython"
    }
    with_ratios = []
    for result in results:
        key = (result.phase, result.num_matvecs)
        try:
            baseline = baselines[key]
        except KeyError as error:
            raise ValueError(f"missing NumPy/Cython baseline for phase={key[0]!r}, num_matvecs={key[1]}") from error
        with_ratios.append(replace(result, ratio_to_numpy_cython=result.seconds / baseline))
    return with_ratios


def _benchmark_data(path: Path) -> pl.DataFrame:
    iids = list_iids(path).unique(maintain_order=True)
    rng = np.random.default_rng(_SEED)
    n_individuals = len(iids)
    return pl.DataFrame(
        {
            "iid": iids,
            "phenotype_1": rng.standard_normal(n_individuals),
            "phenotype_2": rng.standard_normal(n_individuals),
            "intercept": np.ones(n_individuals),
            "covariate": np.linspace(-1.0, 1.0, n_individuals),
        }
    )


def _validate_probe_counts(num_matvecs_values: tuple[int, ...], n_individuals: int) -> None:
    invalid = [value for value in num_matvecs_values if value > n_individuals]
    if invalid:
        raise ValueError(
            "--rhe-benchmark-num-matvecs values cannot exceed the number of individuals "
            f"({n_individuals}); observed {invalid}"
        )


def _time_numpy_cython_rhe(
    path: Path,
    block_metadata: pl.DataFrame,
    *,
    data: pl.DataFrame,
    num_processes: int,
    num_matvecs_values: tuple[int, ...],
) -> tuple[list[_RheBenchmarkResult], dict[int, pl.DataFrame]]:
    results = []
    estimates = {}
    for num_matvecs in num_matvecs_values:
        start = time.perf_counter()
        with GRMOperator.from_hdf5(
            str(path),
            num_processes=num_processes,
            max_num_traits=_CYTHON_MAX_NUM_TRAITS,
            block_metadata=block_metadata,
            alpha=-1.0,
        ) as grm:
            construction_seconds = time.perf_counter() - start
            start = time.perf_counter()
            estimate = _run_numpy_cython_rhe(grm, data=data, num_matvecs=num_matvecs)
            first_seconds = time.perf_counter() - start
            warm_seconds = _time_warm_calls(
                lambda grm=grm, num_matvecs=num_matvecs: _run_numpy_cython_rhe(
                    grm,
                    data=data,
                    num_matvecs=num_matvecs,
                )
            )
            dtype = str(np.dtype(grm.dtype))

        estimates[num_matvecs] = estimate
        results.extend(
            _phase_results(
                "numpy_cython",
                num_matvecs=num_matvecs,
                construction_seconds=construction_seconds,
                first_seconds=first_seconds,
                warm_seconds=warm_seconds,
                execution_units=num_processes,
                dtype=dtype,
            )
        )
    return results, estimates


def _time_jax_rhe(
    path: Path,
    block_metadata: pl.DataFrame,
    *,
    data: pl.DataFrame,
    requested_devices: int,
    num_matvecs_values: tuple[int, ...],
    backend: Backend = Backend.AUTO,
) -> tuple[list[_RheBenchmarkResult], dict[int, pl.DataFrame]]:
    devices = tuple(jax.devices())
    if not devices:
        pytest.skip("RHE JAX benchmark requires at least one JAX device")
    device_count = max(1, min(requested_devices, len(devices), block_metadata.height))
    mesh = Mesh(np.asarray(devices[:device_count]), ("blocks",))

    results = []
    estimates = {}
    for num_matvecs in num_matvecs_values:
        start = time.perf_counter()
        operator = JaxParallelOperator.from_hdf5(
            str(path),
            mesh=mesh,
            block_metadata=block_metadata,
            backend=backend,
        )
        grm = JaxGRMOperator(operator, alpha=-1.0, iids=list_iids(path))
        construction_seconds = time.perf_counter() - start
        start = time.perf_counter()
        estimate = _run_jax_rhe(grm, data=data, num_matvecs=num_matvecs)
        first_seconds = time.perf_counter() - start
        warm_seconds = _time_warm_calls(
            lambda grm=grm, num_matvecs=num_matvecs: _run_jax_rhe(
                grm,
                data=data,
                num_matvecs=num_matvecs,
            )
        )

        backend_name = _jax_backend_name(operator)
        estimates[num_matvecs] = estimate
        results.extend(
            _phase_results(
                backend_name,
                num_matvecs=num_matvecs,
                construction_seconds=construction_seconds,
                first_seconds=first_seconds,
                warm_seconds=warm_seconds,
                execution_units=device_count,
                dtype=str(np.dtype(grm.dtype)),
            )
        )
    return results, estimates


def _run_numpy_cython_rhe(grm: GRMOperator, *, data: pl.DataFrame, num_matvecs: int) -> pl.DataFrame:
    return randomized_haseman_elston(
        grm,
        data.lazy(),
        _PHENO_COLS,
        _COVAR_COLS,
        num_matvecs=num_matvecs,
        trace_est="hutchinson",
        sampler="rademacher",
        seed=_SEED,
    )


def _run_jax_rhe(grm: JaxGRMOperator, *, data: pl.DataFrame, num_matvecs: int) -> pl.DataFrame:
    return randomized_haseman_elston_jax(
        grm,
        data.lazy(),
        _PHENO_COLS,
        _COVAR_COLS,
        num_matvecs=num_matvecs,
        trace_est="hutchinson",
        sampler="rademacher",
        seed=_SEED,
    )


def _time_warm_calls(call: Callable[[], pl.DataFrame]) -> float:
    for _ in range(_WARMUP_ITERATIONS):
        call()

    timings = []
    for _ in range(_TIMED_ITERATIONS):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _phase_results(
    backend: str,
    *,
    num_matvecs: int,
    construction_seconds: float,
    first_seconds: float,
    warm_seconds: float,
    execution_units: int,
    dtype: str,
) -> list[_RheBenchmarkResult]:
    return [
        _RheBenchmarkResult(
            backend,
            TimingPhase.CONSTRUCTION.value,
            num_matvecs,
            construction_seconds,
            execution_units=execution_units,
            dtype=dtype,
        ),
        _RheBenchmarkResult(
            backend,
            TimingPhase.FIRST_EXECUTION.value,
            num_matvecs,
            first_seconds,
            execution_units=execution_units,
            dtype=dtype,
        ),
        _RheBenchmarkResult(
            backend,
            TimingPhase.WARM_EXECUTION.value,
            num_matvecs,
            warm_seconds,
            execution_units=execution_units,
            dtype=dtype,
        ),
    ]


def _time_packed_rhe(
    path: Path,
    block_metadata: pl.DataFrame,
    *,
    data: pl.DataFrame,
    requested_devices: int,
    num_matvecs_values: tuple[int, ...],
) -> tuple[list[_RheBenchmarkResult], dict[int, pl.DataFrame]]:
    devices = tuple(jax.devices("cpu"))
    if not devices:
        pytest.skip("packed RHE benchmark requires a CPU JAX device")
    device_count = max(1, min(requested_devices, len(devices), block_metadata.height))
    mesh = Mesh(np.asarray(devices[:device_count]), ("graph",))
    max_padding_ratio = None if path.parent.name == "testdata" else 1.25
    results = []
    estimates = {}
    for num_matvecs in num_matvecs_values:
        start = time.perf_counter()
        operator = _PackedJaxLinearARG.from_hdf5(
            path,
            mesh=mesh,
            block_metadata=block_metadata,
            backend=Backend.PURE_JAX,
            max_padding_ratio=max_padding_ratio,
        )
        grm = JaxGRMOperator(operator, alpha=-1.0, iids=list_iids(path))
        construction_seconds = time.perf_counter() - start
        start = time.perf_counter()
        estimate = _run_jax_rhe(grm, data=data, num_matvecs=num_matvecs)
        first_seconds = time.perf_counter() - start
        warm_seconds = _time_warm_calls(
            lambda grm=grm, num_matvecs=num_matvecs: _run_jax_rhe(
                grm,
                data=data,
                num_matvecs=num_matvecs,
            )
        )
        estimates[num_matvecs] = estimate
        results.extend(
            _phase_results(
                "packed(pure_jax)",
                num_matvecs=num_matvecs,
                construction_seconds=construction_seconds,
                first_seconds=first_seconds,
                warm_seconds=warm_seconds,
                execution_units=device_count,
                dtype=str(np.dtype(grm.dtype)),
            )
        )
    return results, estimates


def _run_promotion_rhe_child(
    request: pytest.FixtureRequest,
    *,
    representation: Representation,
    path: Path,
    block_metadata: pl.DataFrame,
    parallel_processes: int,
    num_matvecs_values: tuple[int, ...],
) -> None:
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")
    output_path = request.config.getoption("--jax-promotion-output")
    if output_path is None:
        pytest.fail("promotion child requires --jax-promotion-output PATH")

    data = _benchmark_data(path)
    _validate_probe_counts(num_matvecs_values, data.height)
    execution_units = min(parallel_processes, block_metadata.height)
    if representation is Representation.NUMPY_CYTHON:
        results, estimates = _time_numpy_cython_rhe(
            path,
            block_metadata,
            data=data,
            num_processes=execution_units,
            num_matvecs_values=num_matvecs_values,
        )
        requested_backend = resolved_backend = None
    elif representation is Representation.RETAINED_EXACT_RAGGED:
        results, estimates = _time_jax_rhe(
            path,
            block_metadata,
            data=data,
            requested_devices=parallel_processes,
            num_matvecs_values=num_matvecs_values,
            backend=Backend.PURE_JAX,
        )
        requested_backend = Backend.PURE_JAX.value
        resolved_backend = Backend.PURE_JAX.value
    else:
        results, estimates = _time_packed_rhe(
            path,
            block_metadata,
            data=data,
            requested_devices=parallel_processes,
            num_matvecs_values=num_matvecs_values,
        )
        requested_backend = Backend.PURE_JAX.value
        resolved_backend = Backend.PURE_JAX.value

    fingerprint = compute_dataset_fingerprint(path)
    candidate = git_commit()
    dirty = is_git_dirty()
    records = []
    for result in results:
        estimate = estimates[result.num_matvecs]
        notes = json.dumps(
            {
                "estimate": estimate.select(["s2g", "s2e", "h2g"]).to_numpy().tolist(),
                "historical_ir_counterexample": "genoio@c271a9a",
            },
            sort_keys=True,
        )
        records.append(
            make_record(
                platform_label=request.config.getoption("--platform-label"),
                cache_label=request.config.getoption("--cache-policy"),
                candidate_commit=candidate,
                dataset=fingerprint,
                representation=representation.value,
                operation="rhe",
                phase=result.phase,
                workload_size=result.num_matvecs,
                dtype=result.dtype,
                requested_backend=requested_backend,
                resolved_backend=resolved_backend,
                device_count=result.execution_units,
                timed=TimedPhase(phase=result.phase, seconds=result.seconds),
                metric=PerformanceMetrics(),
                notes=notes,
                dirty_worktree=dirty,
            )
        )
    evidence = build_promotion_evidence(
        cache_label=request.config.getoption("--cache-policy"),
        platform_label=request.config.getoption("--platform-label"),
        records=tuple(records),
        candidate_commit=candidate,
        dataset=fingerprint,
    )
    write_evidence_fragment(output_path, evidence)


def _jax_backend_name(operator: JaxParallelOperator) -> str:
    resolved = {block.backend.value for block in operator.blocks}
    if len(resolved) != 1:
        raise ValueError(f"JAX blocks resolved to inconsistent backends: {sorted(resolved)}")
    return f"jax_auto({resolved.pop()})"


def _assert_estimates_match(
    expected_by_num_matvecs: dict[int, pl.DataFrame],
    observed_by_num_matvecs: dict[int, pl.DataFrame],
) -> None:
    for num_matvecs, expected in expected_by_num_matvecs.items():
        observed = observed_by_num_matvecs[num_matvecs]
        np.testing.assert_allclose(
            observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
            expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
            rtol=2e-5,
            atol=2e-5,
        )


def _print_results(results: list[_RheBenchmarkResult]) -> None:
    print("\n| backend | phase | num matvecs | workers/devices | dtype | seconds | ratio to NumPy/Cython |")
    print("|---|---|---:|---:|---|---:|---:|")
    for result in sorted(
        results,
        key=lambda item: (item.num_matvecs, item.phase, item.backend != "numpy_cython"),
    ):
        ratio = "" if result.ratio_to_numpy_cython is None else f"{result.ratio_to_numpy_cython:.3f}"
        print(
            f"| {result.backend} | {result.phase} | {result.num_matvecs} | "
            f"{result.execution_units} | {result.dtype} | {result.seconds:.6f} | {ratio} |"
        )
