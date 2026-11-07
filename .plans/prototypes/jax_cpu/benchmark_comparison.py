"""Paired installed JAX/Cython CPU benchmark on real HDF5 graph blocks."""

import argparse
import json
import platform
import statistics
import time

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxParallelOperator
from linear_dag.core.lineararg import list_blocks
from linear_dag.core.parallel_processing import GRMOperator, ParallelOperator


def ready(value):
    return jax.block_until_ready(value)


def compare(name, width, native, compiled, repeats):
    expected = np.asarray(native()).copy()
    observed = np.asarray(ready(compiled()))
    error = float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-30))
    assert error < 2e-5 and np.isfinite(observed).all(), (name, width, error)
    times = {"cython": [], "jax": []}
    for repetition in range(repeats):
        order = ("cython", "jax") if repetition % 2 == 0 else ("jax", "cython")
        for backend in order:
            start = time.perf_counter()
            ready(native() if backend == "cython" else compiled())
            times[backend].append(time.perf_counter() - start)
    result = dict(
        operation=name,
        width=width,
        relative_l2=error,
        trials=times,
        seconds={key: statistics.median(value) for key, value in times.items()},
    )
    result["speedup"] = result["seconds"]["cython"] / result["seconds"]["jax"]
    print(json.dumps(result), flush=True)
    return result


def main(args):
    report = dict(
        platform=platform.platform(),
        jax=jax.__version__,
        workers=args.workers,
        repetitions=args.repetitions,
        conditions="Warm, interleaved, synchronized; float32; tile width 8",
        cases=[],
    )
    for blocks in args.blocks:
        metadata = list_blocks(args.h5).head(blocks)
        start = time.perf_counter()
        op = JaxParallelOperator.from_hdf5(
            args.h5, block_metadata=metadata, backend=Backend.FFI_CPU, num_processes=args.workers, max_num_traits=8
        )
        ready(op)
        case = dict(
            blocks=metadata.height,
            shape=op.shape,
            jax_load=time.perf_counter() - start,
            scratch_bytes=op._config.scratch_bytes(),
            timings=[],
            compile_seconds={},
        )
        rng = np.random.default_rng(42)
        d = rng.uniform(0.1, 1.1, op.shape[1]).astype(np.float32)
        dj = jnp.asarray(d)
        grm = JaxGRMOperator(op, center=False)
        start = time.perf_counter()
        with ParallelOperator.from_hdf5(
            args.h5, block_metadata=metadata, num_processes=args.workers, max_num_traits=8
        ) as cpu:
            case["cython_load"] = time.perf_counter() - start
            for width in (1, 8):
                b = rng.normal(size=(op.shape[1], width)).astype(np.float32)
                u = rng.normal(size=(op.shape[0], width)).astype(np.float32)
                bj, uj = ready((jnp.asarray(b), jnp.asarray(u)))
                for name, fn, inputs, reference in (
                    ("forward", jax.jit(lambda op, b: op.matmat(b)), (op, bj), lambda: cpu.matmat(b)),
                    ("reverse", jax.jit(lambda op, u: op.rmatmat(u)), (op, uj), lambda: cpu.rmatmat(u)),
                    (
                        "weighted",
                        jax.jit(lambda g, d, u: g.matmat(u, diagonal=d)),
                        (grm, dj, uj),
                        lambda: cpu.matmat(d[:, None] * cpu.rmatmat(u)),
                    ),
                ):
                    start = time.perf_counter()
                    compiled = fn.lower(*inputs).compile()
                    case["compile_seconds"][f"{name}_{width}"] = time.perf_counter() - start
                    case["timings"].append(compare(name, width, reference, lambda: compiled(*inputs), args.repetitions))
        centered = JaxGRMOperator(op, alpha=-1.0, center=True)
        with GRMOperator.from_hdf5(
            args.h5, block_metadata=metadata, num_processes=args.workers, max_num_traits=8, alpha=-1.0
        ) as cpu:
            for width in (1, 8):
                u = rng.normal(size=(op.shape[0], width)).astype(np.float32)
                uj = ready(jnp.asarray(u))
                start = time.perf_counter()
                compiled = jax.jit(lambda g, u: g.matmat(u)).lower(centered, uj).compile()
                case["compile_seconds"][f"centered_grm_{width}"] = time.perf_counter() - start
                case["timings"].append(
                    compare(
                        "centered_grm", width, lambda: cpu.matmat(u), lambda: compiled(centered, uj), args.repetitions
                    )
                )
        report["cases"].append(case)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--blocks", type=int, nargs="+", default=[8, 35])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
