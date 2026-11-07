# Toy LMM REML: fit, derivatives and memory

Measured 2026-11-06 with the installed Python 3.12 / JAX 0.12.0.1 backend on Apple M1. This benchmark replaces RHE with an actual two-variance-component REML optimization. No production code changed.

## Model and computation

256 diploid individuals selected by IID from the first real 1KG block: 150,192 variants over a 5 Mb region. The full source graph for 6,404 haplotypes is retained; subset inputs are embedded and outputs gathered, without pruning the graph. One native CPU worker, float64 throughout.

Model: y = C beta + g + e; g ~ N(0, sigma_g^2 K), e ~ N(0, sigma_e^2 I). C contains an intercept and one normal covariate. The centered allele-frequency-weighted diploid GRM is scaled to mean diagonal 1. Simulation seed 42, beta=(1,0.3), true variances (0.6,0.4). Only the two log variance components are differentiated; SNP weights and the graph are fixed.

REML minimizes 0.5 * [logdet(V) + logdet(C.T V^-1 C) + y.T P y + (n-p) log(2*pi)], with V = sigma_g^2 K + sigma_e^2 I and the usual fixed-effect projection P.

A Cython float64 genotype operator independently constructs K in 8-column chunks; preparation, eigendecomposition and simulation together took 3.91 s. The dense reference caches K and uses Cholesky. The graph path applies K through the native operator and uses JAX CG with implicit differentiation, tolerance 1e-10, maximum 768 iterations, and an explicit forward residual check at every optimizer evaluation. It solves the combined phenotype/fixed-effect RHS array. Both use SciPy L-BFGS-B with identical initialization, bounds and stopping criteria.

For exact logdet in the graph path, this toy caches the eigenvalues of the independently constructed K. Consequently it is a graph-solve/AD benchmark with an exact small-n logdet reference, **not a fully matrix-free/scalable REML algorithm**. No Lanczos or stochastic logdet approximation is benchmarked. The graph itself is passed as a dynamic PyTree argument, not embedded as a compilation constant.

## Results

| Path | Warm likelihood | Warm value + gradient | Fit after compilation/warmup | Peak RSS, likelihood process | Peak RSS, gradient/fit process |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.243 ms | 1.290 ms | 0.019 s | 239.1 MiB | 263.5 MiB |
| operator | 662.278 ms | 1406.102 ms | 21.925 s | 462.8 MiB | 480.0 MiB |

Warm evaluation timings are medians of five synchronized calls. Each value-only and value-and-gradient case runs in a separate fresh process. Fit timing is one optimizer run per path and excludes loading, compilation and the preceding warm evaluations. Process peak RSS includes imports, graph loading, compilation and execution; it is not solely device/scratch memory. Additional phase RSS is sampled every 5 ms, so brief transient peaks can be missed there.

Both optimizers converged in 9 iterations / 11 evaluations to variances approximately (0.7209121658, 0.3071328411). These agree between implementations; agreement with the simulation truth is not required for one random realization. REML objective approximately 362.017057434. Initial objectives/gradients and fitted parameters pass rtol=atol=1e-7; independent central finite differences also agree with the native gradient. Maximum native forward solve relative residual across the fit is below 1e-10.

## Memory interpretation

No runaway memory growth was observed in this toy. The native likelihood and gradient runs peaked at about 463 and 480 MiB respectively, largely influenced by graph loading and compilation. During the warmed optimizer run RSS increased by only about 0.08 MiB. Implicit CG differentiation solves an adjoint system instead of backpropagating through a stored tape of all CG iterations; native block scratch also stays out of backward residuals.

A control run closing over the graph, rather than passing it dynamically, reached about 638 MiB peak RSS; the compilation phase alone increased RSS by about 224 MiB. This illustrates an additional source of memory overhead independent of long AD tapes. Its raw result is retained as `reml-operator-grad-closed-over.json`; current reproduction script uses dynamic graph arguments.

The cached dense GRM is overwhelmingly faster here: its 256x256 matrix is only 0.5 MiB in float64, whereas the graph path repeatedly traverses a source graph containing all 6,404 haplotypes. Prefer the dense cached representation for this toy. These measurements do not establish memory scaling for a larger cohort or runtime advantages for graph-based REML. Larger-n REML would additionally need a suitable logdet strategy.

## Reproduction

Run `benchmark_reml.py --prepare` with the real HDF5 fixture, then invoke each combination of `--backend dense|operator` and `--action value|grad`, providing an `--output` JSON path. Python environment used: `/private/tmp/linear-dag-hijax-py312/bin/python`. All invocations use `--h5 1kg_chromosomes_n3202_blocks.h5` by default. Prepared arrays are regenerable under `/private/tmp/linear-dag-toy-reml`; scripts and result records in this directory are durable.

Raw reports: `reml-{dense,operator}-{value,grad}.json`; model metadata: `reml-model.json`; numerical checks: `reml-verification.json`. The unrelated RHE uncertainty-overflow finding remains unfixed; this benchmark uses float64 REML rather than those RHE error-bar formulas.
