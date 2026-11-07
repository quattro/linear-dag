"""Exact toy REML: dense Cholesky versus native graph CG with cached exact spectrum.

Uses JAX CG's implicit derivative and the same SciPy optimizer for both paths.
This is a solve/AD benchmark, not a scalable matrix-free logdet implementation.
"""

import argparse
import json
import resource
import threading
import time

from pathlib import Path


def prepare(args):
    import numpy as np

    from linear_dag.core.lineararg import list_blocks, list_iids
    from linear_dag.core.parallel_processing import ParallelOperator

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    metadata = list_blocks(args.h5).head(1)
    iids = list_iids(args.h5).to_numpy()
    ids = list(dict.fromkeys(iids))[: args.n]
    pairs = np.asarray([np.flatnonzero(iids == iid) for iid in ids])
    assert pairs.shape == (args.n, 2)
    tick = time.perf_counter()
    K = np.empty((args.n, args.n), dtype=np.float64)
    with ParallelOperator.from_hdf5(
        args.h5, block_metadata=metadata, num_processes=1, max_num_traits=8, dtype=np.float64
    ) as op:
        frequencies = op.rmatvec(np.ones(op.shape[0])) / op.shape[0]
        pq = frequencies * (1 - frequencies)
        weights = np.divide(1, pq, out=np.zeros_like(pq), where=pq > 0)
        for start in range(0, args.n, 8):
            stop = min(start + 8, args.n)
            basis = np.zeros((op.shape[0], stop - start))
            for j in range(start, stop):
                basis[pairs[j], j - start] = 1
            scores = op.rmatmat(basis) - frequencies[:, None] * basis.sum(axis=0)
            scores *= weights[:, None]
            value = op.matmat(scores) - (frequencies[:, None] * scores).sum(axis=0)
            K[:, start:stop] = 0.5 * value[pairs].sum(axis=1)
    np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-7)
    scale = float(np.trace(K) / args.n)
    K = (K + K.T) / (2 * scale)
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() > -1e-10
    rng = np.random.default_rng(42)
    C = np.column_stack([np.ones(args.n), rng.normal(size=args.n)])
    truth = np.array([0.6, 0.4])
    y = C @ np.array([1.0, 0.3]) + np.linalg.cholesky(truth[0] * K + truth[1] * np.eye(args.n)) @ rng.normal(
        size=args.n
    )
    np.savez(
        work / "model.npz", pairs=pairs, scale=scale, eigenvalues=eigenvalues, C=C, y=y, truth=truth, weights=weights
    )
    np.save(work / "K.npy", K)
    report = dict(
        n=args.n,
        variants=len(weights),
        haplotypes=len(iids),
        blocks=metadata["block_name"].to_list(),
        truth=truth.tolist(),
        prepare_seconds=time.perf_counter() - tick,
        normalization=scale,
        min_eigenvalue=float(eigenvalues.min()),
        max_eigenvalue=float(eigenvalues.max()),
    )
    (work / "model.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


class Memory:
    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()

        def poll():
            while not self.stop.wait(0.005):
                self.peak = max(self.peak, self.process.memory_info().rss)

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()

    def report(self):
        return dict(
            baseline_mib=self.baseline / 2**20,
            peak_mib=self.peak / 2**20,
            increase_mib=(self.peak - self.baseline) / 2**20,
        )


def child(args):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax.scipy.linalg import cho_factor, cho_solve
    from jax.scipy.sparse.linalg import cg
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    work = Path(args.work)
    data = np.load(work / "model.npz")
    y, C, eigenvalues = (jnp.asarray(data[k]) for k in ("y", "C", "eigenvalues"))
    n, p = C.shape
    rhs = jnp.column_stack([y, C])
    init = jnp.log(jnp.array([0.5, 0.5]))
    if args.backend == "operator":
        from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxParallelOperator
        from linear_dag.core.lineararg import list_blocks

        tick = time.perf_counter()
        op = JaxParallelOperator.from_hdf5(
            args.h5,
            block_metadata=list_blocks(args.h5).head(1),
            backend=Backend.FFI_CPU,
            num_processes=1,
            dtype=jnp.float64,
        )
        grm = JaxGRMOperator(op, center=True, diagonal=jnp.asarray(data["weights"]))
        pairs = jnp.asarray(data["pairs"])
        scale = float(data["scale"])

        model = grm

        def kinship(v, model):
            expanded = (
                jnp.zeros((op.shape[0], v.shape[1]), dtype=v.dtype).at[pairs.ravel()].set(jnp.repeat(v, 2, axis=0))
            )
            return 0.5 * model.matmat(expanded)[pairs].sum(axis=1) / scale

        jax.block_until_ready(op)
        load_seconds = time.perf_counter() - tick

        def solve(theta, model):
            sg, se = jnp.exp(theta)

            def apply(v):
                return sg * kinship(v, model) + se * v

            value, _ = cg(apply, rhs, tol=1e-10, atol=0, maxiter=3 * n)
            residual = jnp.linalg.norm(apply(value) - rhs) / jnp.linalg.norm(rhs)
            return value, jnp.sum(jnp.log(sg * eigenvalues + se)), residual
    else:
        tick = time.perf_counter()
        K = jnp.asarray(np.load(work / "K.npy"))
        K.block_until_ready()
        model = K
        load_seconds = time.perf_counter() - tick

        def solve(theta, model):
            sg, se = jnp.exp(theta)
            V = sg * model + se * jnp.eye(n)
            factor = cho_factor(V, lower=True)
            value = cho_solve(factor, rhs)
            return (
                value,
                2 * jnp.log(jnp.diag(factor[0])).sum(),
                jnp.linalg.norm(V @ value - rhs) / jnp.linalg.norm(rhs),
            )

    def likelihood(theta, model):
        solved, logdet, residual = solve(theta, model)
        vy, vc = solved[:, 0], solved[:, 1:]
        small = C.T @ vc
        factor = cho_factor(small, lower=True)
        cy = C.T @ vy
        quadratic = y @ vy - cy @ cho_solve(factor, cy)
        loss = 0.5 * (logdet + 2 * jnp.log(jnp.diag(factor[0])).sum() + quadratic + (n - p) * jnp.log(2 * jnp.pi))
        return loss, residual

    fn = jax.jit(jax.value_and_grad(likelihood, has_aux=True) if args.action == "grad" else likelihood)
    with Memory() as compilation_memory:
        tick = time.perf_counter()
        compiled = fn.lower(init, model).compile()
        compile_seconds = time.perf_counter() - tick
    with Memory() as first_memory:
        tick = time.perf_counter()
        first = jax.block_until_ready(compiled(init, model))
        first_seconds = time.perf_counter() - tick
    times = []
    with Memory() as warm_memory:
        for _ in range(5):
            tick = time.perf_counter()
            jax.block_until_ready(compiled(init, model))
            times.append(time.perf_counter() - tick)
    report = dict(
        backend=args.backend,
        action=args.action,
        load_seconds=load_seconds,
        compile_seconds=compile_seconds,
        first_seconds=first_seconds,
        warm_seconds=float(np.median(times)),
        trials=times,
        compilation_memory=compilation_memory.report(),
        first_memory=first_memory.report(),
        warm_memory=warm_memory.report(),
    )
    if args.action == "grad":
        (value, residual), gradient = first
        report.update(
            initial_value=float(value), initial_gradient=np.asarray(gradient).tolist(), initial_residual=float(residual)
        )
        evaluated = []

        def objective(theta):
            (value, residual), gradient = jax.block_until_ready(compiled(jnp.asarray(theta), model))
            if not np.isfinite(value) or not np.isfinite(gradient).all() or float(residual) > 1e-8:
                raise RuntimeError(f"invalid objective/solve: {value}, {gradient}, residual={residual}")
            evaluated.append(dict(theta=np.asarray(theta).tolist(), value=float(value), residual=float(residual)))
            return float(value), np.asarray(gradient)

        with Memory() as fit_memory:
            tick = time.perf_counter()
            fit = minimize(
                objective,
                np.asarray(init),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-9, 5)] * 2,
                options=dict(ftol=1e-12, gtol=1e-7, maxiter=100),
            )
            fit_seconds = time.perf_counter() - tick
        assert fit.success, fit.message
        report.update(
            fit_seconds=fit_seconds,
            fit_memory=fit_memory.report(),
            fitted_variances=np.exp(fit.x).tolist(),
            objective=float(fit.fun),
            gradient=np.asarray(fit.jac).tolist(),
            iterations=fit.nit,
            evaluations=fit.nfev,
            max_residual=max(x["residual"] for x in evaluated),
            optimizer_message=str(fit.message),
            evaluated=evaluated,
        )
    else:
        value, residual = first
        report.update(initial_value=float(value), initial_residual=float(residual))
    report["process_peak_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="1kg_chromosomes_n3202_blocks.h5")
    parser.add_argument("--work", default="/private/tmp/linear-dag-toy-reml")
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--backend", choices=["dense", "operator"])
    parser.add_argument("--action", choices=["value", "grad"], default="grad")
    parser.add_argument("--output", default="/private/tmp/reml-result.json")
    args = parser.parse_args()
    prepare(args) if args.prepare else child(args)
