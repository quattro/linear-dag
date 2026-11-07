"""Check prototype parity and time the exact exported real-data workload."""

import argparse
import dataclasses
import hashlib
import json
import resource
import statistics
import time

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from packed import pack, product, weighted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.input / "baseline.json").read_text())
    blocks = []
    for i in range(len(baseline["blocks"])):
        with np.load(args.input / f"block_{i}.npz") as data:
            blocks.append(dict(data))
    start = time.perf_counter()
    graph, cfg = pack(blocks, workers=args.workers, chunk=args.chunk)
    jax.block_until_ready(graph)
    load_time = time.perf_counter() - start
    report = dict(
        jax=jax.__version__,
        config=dataclasses.asdict(cfg),
        baseline=baseline,
        packing_seconds=load_time,
        scratch_bytes=cfg.scratch_bytes(),
        timings=[],
    )
    # Graph payloads are explicit runtime inputs, not closed-over constants.
    functions = {
        "forward": jax.jit(lambda graph, b: product(graph, b, cfg)),
        "reverse": jax.jit(lambda graph, u: product(graph, u, cfg, transpose=True)),
        "weighted": jax.jit(lambda graph, d, u: weighted(graph, d, u, cfg)),
    }
    for k in baseline["widths"]:
        with np.load(args.input / f"rhs_{k}.npz") as data:
            b, u, d = (jnp.asarray(data[key]) for key in ("b", "u", "d"))
            for name, fn in functions.items():
                inputs = (graph, d, u) if name == "weighted" else (graph, b if name == "forward" else u)
                start = time.perf_counter()
                compiled = fn.lower(*inputs).compile()
                compile_time = time.perf_counter() - start
                result = compiled(*inputs).block_until_ready()
                expected = data[name]
                # Cancellation makes elementwise relative errors misleading;
                # report normalized L2 and peak absolute discrepancy separately.
                error = np.asarray(result) - expected
                relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-30))
                if relative_l2 > 2e-5 or not np.all(np.isfinite(result)):
                    raise AssertionError((name, k, relative_l2))
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
                        compile_seconds=compile_time,
                        relative_l2=relative_l2,
                        max_abs=float(np.max(np.abs(error))),
                    )
                )
    report["process_peak_rss_native_units"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["source_sha256"] = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
        for name in ("packed.py", "kernel.cc", "build.py")
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
