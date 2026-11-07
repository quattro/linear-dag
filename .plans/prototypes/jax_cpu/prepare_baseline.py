"""Export real graph buffers, reference products, and current Cython timings.

Run with the existing project environment. Outputs go to the requested directory;
large NPZ files are regenerable inputs, not the durable benchmark report.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time

from pathlib import Path

import numpy as np

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.lineararg import list_blocks
from linear_dag.core.parallel_processing import ParallelOperator


def timed(fn):
    fn()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = list_blocks(args.h5).head(args.blocks)
    names = metadata["block_name"].to_list()
    keys = ("indptr", "indices", "data", "nonunique_indices", "variant_indices", "sample_indices", "flip")
    for i, name in enumerate(names):
        op = JaxLinearARG.from_hdf5_block(args.h5, name, backend=Backend.FFI_CPU)
        values = {key: np.asarray(getattr(op, key)) for key in keys}
        values["min_index_to_keep"] = np.asarray(op.min_index_to_keep)
        np.savez(args.output / f"block_{i}.npz", **values)
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        blocks=names,
        workers=args.workers,
        chunk=args.chunk,
        widths=args.widths,
        timings=[],
    )
    rng = np.random.default_rng(42)
    with ParallelOperator.from_hdf5(
        args.h5, num_processes=args.workers, max_num_traits=args.chunk, block_metadata=metadata
    ) as op:
        report["shape"] = op.shape
        d = rng.uniform(0.1, 1.1, op.shape[1]).astype(np.float32)
        for k in args.widths:
            b = rng.normal(size=(op.shape[1], k)).astype(np.float32)
            u = rng.normal(size=(op.shape[0], k)).astype(np.float32)
            forward, ft = timed(lambda: op.matmat(b))
            reverse, rt = timed(lambda: op.rmatmat(u))
            grm, gt = timed(lambda: op.matmat(d[:, None] * op.rmatmat(u)))
            np.savez(args.output / f"rhs_{k}.npz", b=b, u=u, d=d, forward=forward, reverse=reverse, weighted=grm)
            report["timings"].append(dict(k=k, forward=ft, reverse=rt, weighted=gt))
    (args.output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
