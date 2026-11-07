# Packed ragged CPU HiJAX prototype

Recorded 2026-11-06. This is an executable investigation, not a shipped API.
Source, tests, and JSON benchmark reports live in the repository, outside caches
and generated site documentation. Build products and exported large data are
regenerable. Nothing here changes the production dependency bounds or backend.

## What is implemented

- Packed ragged graph buffers with block offsets, without graph-size padding.
- Forward and transpose genotype products, general `X diag(d) X.T U`, and the
  diagonal adjoint `sum_r (X.T U)[j,r] * (X.T G)[j,r]`.
- One FFI operation per RHS tile, using JAX's CPU thread pool and an asynchronous
  completion promise. There is no additional process pool or native thread pool.
- Fixed round-robin block assignment and fixed worker reduction order.
- XLA-owned per-invocation scratch buffers. A sequential XLA loop bounds scratch
  width by `chunk`, including after batching vectors into RHS columns.
- Blockwise weighted products and weight gradients, without retaining a
  genome-wide variant-by-RHS intermediate. Backward recomputes block scores.
- Explicit HiJAX rules for JVP, VJP, linearization, transposition, and batching.
- Shared-diagonal batching becomes a matrix product; independently batched
  diagonals use a correct, sequential mapped fallback.

Graph buffers are runtime integer/bool PyTree arguments, not compile-time array
constants. Floating inputs and weights must have matching float32 or float64
dtypes. The native kernels use scalar loops that the compiler may vectorize;
they do not call BLAS. Fixed graph/worker configuration gives a fixed reduction
order, but changing workers can change floating-point summation.

## Verification

The prototype suite has 13 passing cases (parameterized cases included), covering:

- Independent dense forward/transpose/weighted references for two ragged graphs.
- Float32 and float64, multiple worker counts and RHS chunk widths.
- JIT, JVP, VJP, linearization, forward-over-reverse derivatives, nested/nonleading
  vector maps, shared/batched weights and inputs, and checkpointing.
- Concurrent invocations with separate scratch buffers.
- A differentiable genotype loss combined with an implicitly differentiated
  conjugate-gradient solve, compared to a dense solve and dense gradients.
- Basic rejection of invalid input shapes and empty graph collections.

Real compressed graphs also pass forward/transpose/weighted parity checks against
the existing Cython operator. Normalized L2 discrepancies were below 1.3e-6 in
these runs; absolute discrepancies are recorded in JSON, not hidden by that metric.
The large-data check is not an independent mathematical oracle; dense tests serve
that role. Large-data gradients have not yet been benchmarked or validated.

## Measured results

Hardware: Apple M1, 8 CPU cores, 16 GiB RAM, macOS 26.4.1.
Prototype: Python 3.12.13, JAX/JAXlib 0.12.0.1.
Cython reference: Python 3.11.15, current `jax` branch at base commit
`c199307eeb1f04cb9204ce4bb99342f7ea32a2de`, with the separately recorded constructor fix.

Eight actual 1KG blocks, 6,404 haplotype rows, 1,067,520 variants; four workers
for each implementation and RHS chunk width eight:

| RHS width | Operation | Cython seconds | Prototype seconds | Cython / prototype |
| ---: | --- | ---: | ---: | ---: |
| 1 | Forward | 0.01959 | 0.01997 | 0.98 |
| 1 | Transpose | 0.01982 | 0.02182 | 0.91 |
| 1 | Weighted | 0.03890 | 0.04120 | 0.94 |
| 8 | Forward | 0.09768 | 0.03817 | 2.56 |
| 8 | Transpose | 0.10343 | 0.04791 | 2.16 |
| 8 | Weighted | 0.20486 | 0.08783 | 2.33 |
| 32 | Forward | 0.48698 | 0.17359 | 2.81 |
| 32 | Transpose | 0.42131 | 0.23064 | 1.83 |
| 32 | Weighted | 0.96293 | 0.36633 | 2.63 |

The weighted reference is the same general algebra composed from Cython
`ParallelOperator` calls, not its specialized centered/frequency-weighted
`GRMOperator`. This distinction is necessary when interpreting the timing.

Timings are seven-call warmed-up medians, with synchronization for JAX. They
exclude loading and compilation and were collected in separate processes, not
interleaved trials. Different Python/NumPy environments and different graph
scheduling/kernel loops mean these runs do not isolate the cause of the gains.
There is no established speedup for single-vector workloads or biobank scale.

The eight-block scratch allocation is 57,347,712 bytes at float32/chunk eight.
That is native scratch only, not total live memory. The reported process peak RSS
was about 2.03 GB, including loaded host arrays, expected outputs, JAX inputs,
compilations, and prior operations; there is no matched Cython RSS comparison.
Compilation took approximately 5–14 ms for widths 1/8 and 37–73 ms for width 32.

Exact block names, timings, parity errors, configurations, and measured-source
hashes are in `baseline-two-blocks.json` and `baseline-eight-blocks.json`.

## Reproduction

### Fresh-process transpose benchmark

`cold_benchmark.py` measures process launch to the first `X.T @ U` result, starting
from the original HDF5 file, including imports, graph loading, packing/device setup
where applicable, compilation, and the first product. Each operation runs once in
a fresh process; JAX persistent compilation caching is disabled. OS filesystem
caches are not flushed. This is process-cold, not guaranteed cold-disk I/O.

On the same eight blocks and four workers, three interleaved trials per case:

| RHS vectors | Cython median to result | Prototype median to result | Cython range | Prototype range |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.525 s | 1.320 s | 1.518–2.038 s | 1.313–4.897 s |
| 8 | 1.590 s | 1.343 s | 1.587–1.590 s | 1.343–1.346 s |

The first prototype trial spent 3.849 s in imports; subsequent prototype imports
took about 0.27 s. The outlier is retained. Typical prototype HDF5 reading took
0.78 s, packing/device setup 0.21 s, compilation 7–8 ms, and the first product
18–23 ms (one vector) or 43–47 ms (eight vectors). Cython starts workers
asynchronously, so its recorded first-product phase also includes waiting for
worker initialization/loading; it is not an isolated kernel timing.

Full process lifecycle medians, including worker cleanup/interpreter shutdown,
were 1.848/1.914 s for Cython (one/eight vectors), and 1.401/1.426 s for the
prototype. The primary table excludes cleanup, measuring time to usable result.
Norms and sampled outputs agree across all trials; full-output parity was checked
in the separate warm benchmarks. This is an operator benchmark, not complete
GWAS preprocessing, covariate adjustment, or output writing.

These are the current loading paths: Cython uses the full installed project;
the isolated prototype imports its smaller standalone module and reads required
graph arrays directly from HDF5. A production-integrated JAX API may have different
import overhead. Python/JAX versions differ as recorded above.

Raw trials and phase timings are in `cold-eight-blocks.json`. Reproduce with:

```sh
.venv/bin/python .plans/prototypes/jax_cpu/cold_benchmark.py \
  --h5 1kg_chromosomes_n3202_blocks.h5 --repetitions 3 \
  --output /private/tmp/cold-eight-blocks-repeat.json
```

The prototype environment also needs `h5py` for this HDF5 benchmark (tested with
3.16.0). Both the parent and Cython children need POSIX shared-memory permission.

### Build and warm benchmarks

Create an isolated Python 3.12 environment. The following uses a disposable
location; clearing it removes no source or report in this directory:

```sh
uv venv --python 3.12 /private/tmp/linear-dag-hijax-py312
uv pip install --python /private/tmp/linear-dag-hijax-py312/bin/python \
  -r .plans/prototypes/jax_cpu/requirements.txt
/private/tmp/linear-dag-hijax-py312/bin/python .plans/prototypes/jax_cpu/build.py
JAX_ENABLE_X64=1 /private/tmp/linear-dag-hijax-py312/bin/python -m pytest \
  -p no:capture --confcutdir=.plans/prototypes/jax_cpu \
  .plans/prototypes/jax_cpu/test_packed.py -q
```

The existing project environment must have its native FFI extension built for
the export step. Dataset paths refer to the local 1KG symlink and its external
target. The baseline needs permission to allocate POSIX shared memory.

```sh
.venv/bin/python .plans/prototypes/jax_cpu/prepare_baseline.py \
  --h5 1kg_chromosomes_n3202_blocks.h5 --blocks 8 --workers 4 \
  --widths 1 8 32 --output /private/tmp/linear-dag-packed-eight-blocks
/private/tmp/linear-dag-hijax-py312/bin/python .plans/prototypes/jax_cpu/benchmark.py \
  --input /private/tmp/linear-dag-packed-eight-blocks --workers 4 \
  --output /private/tmp/linear-dag-packed-eight-blocks/repeated-result.json
```

## Remaining before production integration

1. Decide Python/JAX dependency migration and run the existing package against
   the new versions. Current production remains on JAX <0.11.
2. Integrate canonical ingress validation and sample identity alignment. Packing
   currently expects already-valid compressed workspace reuse from existing
   ingress; it does not prove every lifetime invariant for arbitrary graph arrays.
   Packed graph buffers/configuration must not be manually altered.
3. Replace fixed chunk/worker choices with an explicit workspace-budget policy;
   assess balancing using graph edge counts and optimize single-vector execution.
4. Avoid unnecessary reverse-mode work for inputs whose gradients are unrequested.
   Current rules are mathematically correct but may compute both cotangents.
5. Investigate 64-bit packed offsets for inputs exceeding int32 aggregate capacity.
   Such inputs are currently rejected rather than silently overflowing.
6. Profile full dataset and larger-sample workloads, repeated parameterized solves,
   batched diagonal gradients, loading, and matched peak memory. Repeat/interleave
   performance trials before setting release acceptance thresholds.
7. Add platform CI, native error hardening, packaging, public operator/solver
   adapters, and broader transform coverage before promoting this prototype.

The primitive abstraction is promising, but the measured batching gains do not
yet establish that this should replace the current backend everywhere.
