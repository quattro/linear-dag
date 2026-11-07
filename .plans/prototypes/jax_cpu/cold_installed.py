"""Fresh-process HDF5-to-first-transpose benchmark, with interleaved trials.

Process-cold only: this does not flush the OS filesystem cache. The parent uses
only the standard library and times arrival of the child's first-result record,
separately from process exit/worker cleanup.
"""

import argparse
import json
import subprocess
import time

from pathlib import Path


def child(args):
    start = time.perf_counter()
    import numpy as np

    names = json.loads(Path(args.blocks_json).read_text())["baseline"]["blocks"]
    phases = {}
    if args.backend == "cython":
        from linear_dag.core.lineararg import list_blocks
        from linear_dag.core.parallel_processing import ParallelOperator

        phases["imports"] = time.perf_counter() - start
        tick = time.perf_counter()
        metadata = list_blocks(args.h5).filter(__import__("polars").col("block_name").is_in(names))
        with ParallelOperator.from_hdf5(args.h5, num_processes=4, max_num_traits=8, block_metadata=metadata) as op:
            phases["construction"] = time.perf_counter() - tick
            tick = time.perf_counter()
            u = np.random.default_rng(123).normal(size=(op.shape[0], args.width)).astype(np.float32)
            phases["input"] = time.perf_counter() - tick
            tick = time.perf_counter()
            result = op.rmatmat(u)
            phases["first_product"] = time.perf_counter() - tick
            emit(result, phases, start, args, names)
    else:
        import jax
        import jax.numpy as jnp
        import polars as pl

        from linear_dag.core.jaxlinarg import Backend, JaxParallelOperator
        from linear_dag.core.lineararg import list_blocks

        jax.config.update("jax_enable_compilation_cache", False)
        phases["imports"] = time.perf_counter() - start
        tick = time.perf_counter()
        metadata = list_blocks(args.h5).filter(pl.col("block_name").is_in(names))
        op = JaxParallelOperator.from_hdf5(
            args.h5, block_metadata=metadata, backend=Backend.FFI_CPU, num_processes=4, max_num_traits=8
        )
        jax.block_until_ready(op)
        phases["construction"] = time.perf_counter() - tick
        tick = time.perf_counter()
        u = jnp.asarray(np.random.default_rng(123).normal(size=(op.shape[0], args.width)).astype(np.float32))
        u.block_until_ready()
        phases["input"] = time.perf_counter() - tick
        tick = time.perf_counter()
        compiled = jax.jit(lambda op, u: op.rmatmat(u)).lower(op, u).compile()
        phases["compile"] = time.perf_counter() - tick
        tick = time.perf_counter()
        result = np.asarray(compiled(op, u).block_until_ready())
        phases["first_product"] = time.perf_counter() - tick
        emit(result, phases, start, args, names)


def emit(result, phases, start, args, names):
    import numpy as np

    ready = time.perf_counter() - start
    record = dict(
        backend=args.backend,
        width=args.width,
        blocks=names,
        phases=phases,
        child_to_result_seconds=ready,
        shape=result.shape,
        sum=float(np.sum(result, dtype=np.float64)),
        norm=float(np.linalg.norm(result)),
        sample=result.ravel()[:: max(1, result.size // 32)].tolist(),
    )
    print("COLD_RESULT " + json.dumps(record), flush=True)


def parent(args):
    records = []
    for repetition in range(args.repetitions):
        for width in (1, 8):
            order = ("cython", "jax") if repetition % 2 == 0 else ("jax", "cython")
            for backend in order:
                python = args.cython_python if backend == "cython" else args.jax_python
                command = [
                    python,
                    str(Path(__file__).resolve()),
                    "--backend",
                    backend,
                    "--width",
                    str(width),
                    "--h5",
                    args.h5,
                    "--blocks-json",
                    args.blocks_json,
                ]
                start = time.perf_counter()
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
                    record = None
                    for line in proc.stdout:
                        if line.startswith("COLD_RESULT "):
                            received = time.perf_counter() - start
                            record = json.loads(line.removeprefix("COLD_RESULT "))
                            record["process_to_result_seconds"] = received
                            break
                    _, stderr = proc.communicate(timeout=120)
                    if proc.returncode or record is None:
                        raise RuntimeError(f"{backend} failed: {stderr}")
                record["process_lifecycle_seconds"] = time.perf_counter() - start
                record["repetition"] = repetition
                records.append(record)
                print(backend, width, round(record["process_to_result_seconds"], 3), flush=True)
                Path(args.output).write_text(
                    json.dumps(
                        dict(
                            conditions=(
                                "Fresh processes; OS page cache not flushed; "
                                "persistent JAX compilation cache disabled; 4 workers, RHS tile 8"
                            ),
                            records=records,
                        ),
                        indent=2,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--blocks-json", default=str(Path(__file__).with_name("baseline-eight-blocks.json")))
    parser.add_argument("--backend", choices=("cython", "jax"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--cython-python", default="/private/tmp/linear-dag-hijax-py312/bin/python")
    parser.add_argument("--jax-python", default="/private/tmp/linear-dag-hijax-py312/bin/python")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", default=str(Path(__file__).with_name("cold-installed-eight-blocks.json")))
    options = parser.parse_args()
    child(options) if options.backend else parent(options)
