"""Fresh-process Kodama RHE comparison using the installed CLI implementation."""

import argparse
import json
import os
import subprocess
import sys
import time

from pathlib import Path


def child(args):
    import jax

    from linear_dag import cli

    jax.config.update("jax_enable_compilation_cache", False)
    field = "randomized_haseman_elston_jax" if args.backend == "jax" else "randomized_haseman_elston"
    original = getattr(cli, field)
    measurements = {}

    def measured(*a, **kw):
        tick = time.perf_counter()
        result = original(*a, **kw)
        measurements["rhe_seconds"] = time.perf_counter() - tick
        measurements["estimates"] = result.to_dicts()
        return result

    setattr(cli, field, measured)
    command = [
        "rhe",
        args.h5,
        str(Path(args.work) / "phenotypes.tsv"),
        "--pheno-name",
        "iid",
        *[f"y{i}" for i in range(args.traits)],
        "--covar",
        str(Path(args.work) / "phenotypes.tsv"),
        "--covar-name",
        "iid",
        "covariate",
        "--num-processes",
        "4",
        "--num-matvecs",
        "100",
        "--estimator",
        "xnystrace",
        "--sampler",
        "normal",
        "--seed",
        "42",
        "--out",
        str(Path(args.work) / f"{args.backend}-{args.traits}"),
    ]
    if args.backend == "jax":
        command.append("--jax-backend")
    tick = time.perf_counter()
    cli._main(command)
    measurements["cli_seconds"] = time.perf_counter() - tick
    print("RHE_RESULT " + json.dumps(measurements), flush=True)


def parent(args):
    import numpy as np
    import polars as pl

    from linear_dag.core.lineararg import list_blocks, list_iids

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    ids = list_iids(args.h5).unique(maintain_order=True)
    rng = np.random.default_rng(20261106)
    covariate = rng.normal(size=len(ids))
    data = {"iid": ids, "covariate": covariate}
    for i in range(8):
        data[f"y{i}"] = 0.3 * covariate + rng.normal(size=len(ids))
    pl.DataFrame(data).write_csv(work / "phenotypes.tsv", separator="\t")
    report = dict(
        blocks=list_blocks(args.h5).height,
        individuals=len(ids),
        workers=4,
        estimator="xnystrace",
        probes=100,
        sampler="normal",
        seed=42,
        phenotype_seed=20261106,
        phenotype_model="0.3 * normal covariate + independent normal noise",
        conditions="Fresh processes, OS file cache not flushed, persistent JAX compilation cache disabled. "
        "RHE phase includes first-use JIT and deferred Cython worker readiness. "
        "Process total includes imports, loading, estimation, output and cleanup.",
        records=[],
    )
    for repetition in range(args.repetitions):
        for traits in (1, 8):
            order = ("cython", "jax") if repetition % 2 == 0 else ("jax", "cython")
            for backend in order:
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--h5",
                    args.h5,
                    "--work",
                    str(work),
                    "--backend",
                    backend,
                    "--traits",
                    str(traits),
                ]
                env = dict(os.environ)
                env["JAX_ENABLE_X64"] = "0"
                tick = time.perf_counter()
                result = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=300)
                elapsed = time.perf_counter() - tick
                (work / f"{repetition}-{traits}-{backend}.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                line = next(line for line in result.stdout.splitlines() if line.startswith("RHE_RESULT "))
                record = json.loads(line.removeprefix("RHE_RESULT "))
                record.update(backend=backend, traits=traits, repetition=repetition, process_seconds=elapsed)
                report["records"].append(record)
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
                print(backend, traits, round(elapsed, 3), round(record["rhe_seconds"], 3), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--backend", choices=["jax", "cython"])
    p.add_argument("--traits", type=int)
    p.add_argument("--work", default="/private/tmp/linear-dag-rhe-comparison")
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--output", default=str(Path(__file__).with_name("rhe-cli-comparison.json")))
    args = p.parse_args()
    child(args) if args.backend else parent(args)
