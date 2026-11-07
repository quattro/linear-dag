"""Benchmark the installed public API against the exported Cython baseline."""

import argparse
import json
import statistics
import time

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxParallelOperator
from linear_dag.core.lineararg import list_blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.input / "baseline.json").read_text())
    metadata = list_blocks(args.hdf5).head(len(baseline["blocks"]))
    assert metadata["block_name"].to_list() == baseline["blocks"]
    start = time.perf_counter()
    op = JaxParallelOperator.from_hdf5(
        args.hdf5, block_metadata=metadata, backend=Backend.FFI_CPU, num_processes=baseline["workers"]
    )
    jax.block_until_ready(op)
    report = dict(
        jax=jax.__version__,
        shape=op.shape,
        workers=op.num_processes,
        load_seconds=time.perf_counter() - start,
        scratch_bytes=op._config.scratch_bytes(),
        timings=[],
    )
    grm = JaxGRMOperator(op, center=False)
    functions = dict(
        forward=jax.jit(lambda op, b: op.matmat(b)),
        reverse=jax.jit(lambda op, u: op.rmatmat(u)),
        weighted=jax.jit(lambda grm, d, u: grm.matmat(u, diagonal=d)),
    )
    for k in baseline["widths"]:
        with np.load(args.input / f"rhs_{k}.npz") as data:
            b, u, d = (jnp.asarray(data[key]) for key in ("b", "u", "d"))
            for name, fn in functions.items():
                inputs = (grm, d, u) if name == "weighted" else (op, b if name == "forward" else u)
                start = time.perf_counter()
                compiled = fn.lower(*inputs).compile()
                compile_seconds = time.perf_counter() - start
                result = compiled(*inputs).block_until_ready()
                error = np.asarray(result) - data[name]
                relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(data[name]), 1e-30))
                assert relative_l2 < 2e-5 and np.isfinite(result).all(), (name, k, relative_l2)
                timings = []
                for _ in range(7):
                    start = time.perf_counter()
                    compiled(*inputs).block_until_ready()
                    timings.append(time.perf_counter() - start)
                report["timings"].append(
                    dict(
                        operation=name,
                        k=k,
                        seconds=statistics.median(timings),
                        compile_seconds=compile_seconds,
                        relative_l2=relative_l2,
                    )
                )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
