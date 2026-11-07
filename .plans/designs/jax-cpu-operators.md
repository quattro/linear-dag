# CPU-first JAX operators: decisions and baseline

Recorded: 2026-11-06

Status: packed CPU HiJAX backend integrated and verified through an installed
wheel; larger-sample validation remains outstanding.

## Production integration completed

User authorized replacing the existing JAX backend without waiting for unavailable
larger-sample data. Implementation steps:

1. Promote packed native kernels and HiJAX rules into the package; add centered
   products, bounded workspace selection, and native shape checks.
2. Route CPU operator factories/products through packed execution while retaining
   the explicit pure-JAX fallback. Preserve callable operator names and HDF5 schema.
3. Add dynamic diagonal weights to the GRM API and use native CPU worker counts
   in RHE rather than counting virtual JAX devices.
4. Migrate to Python 3.12 and pin the verified experimental JAX interface; build
   wheels and verify installed artifacts.
5. Run native transformation, existing JAX/RHE/CLI, and package regression checks;
   document migration and remaining large-sample uncertainty.

No larger-data performance claim or release is authorized by these checks alone.
This is a repository source document, not a cache or generated site artifact.

## Objective and scope

Build a fast CPU JAX interface to existing compressed LinearARG graphs for both
genotype products in differentiable models and GRM products in LMM/GLMM workflows.
Graph construction is outside this work. GPU execution is not initially required.
The package is unreleased; experimental APIs may be revised deliberately.

Required operations are `X @ B`, `X.T @ U`, and
`X @ diag(d(theta)) @ X.T @ U`. Centering and normalization must be explicit.
The graph is fixed data; numerical inputs and diagonal weights are differentiable.
Ordinary JAX code computes `d(theta)` and propagates gradients to model parameters.
Do not restrict the diagonal to allele-frequency weights or assume positivity.
For real diagonal weights the operator is symmetric; nonnegative weights guarantee
positive semidefiniteness.

Support JIT, batching, forward and reverse AD, and their relevant compositions.
Initial batching should cover a shared graph with shared or independently batched
inputs and diagonal weights. Mapping over unrelated ragged graphs is deferred.

## Architecture selected and validated on available data

- Pack variable-sized graph blocks into arrays with offsets, without padding every
  block to one common size.
- Execute graph traversals using native CPU kernels behind JAX primitives.
- Schedule independent blocks with bounded native parallelism; a JAX CPU device
  need not correspond to each worker.
- Choose RHS chunk width and block concurrency together under a workspace budget.
- Transpose products write disjoint variant ranges. Forward products require a
  sample-space reduction with a defined order and measured memory cost.
- Provide a dedicated blockwise GRM execution path: transpose traversal, diagonal
  weighting, forward traversal, then accumulate sample-space contributions. Avoid
  a genome-wide variant-by-RHS intermediate.
- Keep graph arrays as an ordinary PyTree initially. HiJAX primitives are a
  candidate for explicit batching/derivative rules; custom HiTypes are optional
  and should solve a demonstrated representation problem.
- Consider a Lineax adapter while keeping native execution independent of solvers.
  Solver convergence and implicit differentiation belong to the solver layer.

For `Y = X diag(d) X.T U` and output cotangent `G`, the derivative contracts are:

```text
JVP:       dY = K(d) dU + X diag(dd) X.T U
input VJP: U_bar = K(d) G
weight VJP: d_bar[j] = sum_r (X.T G)[j,r] * (X.T U)[j,r]
```

These computations can be blockwise. Initially investigate recomputing backward
intermediates instead of retaining genome-wide scores. Independently batched
diagonals need their own correct batching behavior; they cannot be treated as a
single shared diagonal merely by merging RHS columns.

## Current branch and compatibility

- Branch: `jax`; base commit `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`.
- This branch does not contain the MSBF recombination work and does not use genoio.
- Runtime inspected: Python 3.11.15, JAX/JAXlib 0.10.2, Equinox 0.13.8.
- Project bounds are JAX/JAXlib `>=0.8.0,<0.11`.
- Installed experimental HiJAX exposes `HiPrimitive`, not the newer documented
  `HiPrim`/`MappingSpec` interface. Select and verify a version before migration.
- Existing FFI batching requests sequential mapped execution. A shared-graph
  batching rule could dispatch multiple vectors to the multiple-RHS kernel.
- Existing product custom VJPs reject forward-mode JVP.
- Existing multi-device forward execution replicates the variant input; transpose
  execution assembles segments on one device. Neither should be assumed to scale
  memory simply by adding devices.

References consulted:

- https://docs.jax.dev/en/latest/301/custom-derivatives.html
- https://docs.jax.dev/en/latest/301/hijax-types.html

## Constructor bug fixed in the working tree

Direct `JaxLinearARG` construction with flipped alleles left the cached flipped
variant indices empty. Forward products omitted the flip correction while
transpose products applied it, violating the adjoint identity.

`src/linear_dag/core/jaxlinarg/operator.py` now derives the cache during direct
initialization. Factory construction retains its already-derived cache.
`tests/jax/test_operator_pure_jax.py` adds eager and JIT checks against explicit
expected products and the adjoint identity. Both regression cases failed before
the fix and passed afterward.

The native extension was initially absent. It was rebuilt using the existing
`hatch_build.build_ffi_cpu_extension_with_optional_blas_fallback` function.
Build diagnostics: CPU FFI available, Accelerate BLAS enabled, native tuning off.

Verification command:

```sh
.venv/bin/python -m pytest -p no:capture \
  tests/jax/test_operator_pure_jax.py \
  tests/jax/test_autodiff.py \
  tests/jax/test_operator_ffi_cpu.py -q
```

Result: **104 passed, 9 skipped**. This is focused verification, not a full-suite
or large-data correctness claim. Changes were not committed at recording time.

## Fresh baseline smoke run

The local `1kg_chromosomes_n3202_blocks.h5` is a symlink to external data. Metadata
inspection found 35 graph blocks, 6,404 haplotype rows, 4,245,055 variants,
14,833,045 graph nodes, and 102,944,653 edges. Per-block edge counts range from
220,285 to 4,707,422. Dataset persistence is separate from this document.

The timed smoke run selected only these two blocks:

- `1_1_5000000`: 150,192 variants, 3,881,020 edges.
- `1_5000001_10000000`: 140,080 variants, 3,789,416 edges.

Combined shape: 6,404 haplotypes by 290,272 variants. RHS width: 8, float32.
Canonical baseline: two shared-memory worker processes. JAX: one CPU device.
These are warmed-up median product timings, excluding loading and compilation.
This is not a biobank-scale result or a controlled equal-thread-count comparison.

| Implementation | Forward product | Transpose product |
| --- | ---: | ---: |
| Cython ParallelOperator, 2 processes | 0.040961 s | 0.039167 s |
| JAX FFI, 1 CPU device | 0.059221 s | 0.056864 s |
| Pure JAX, 1 CPU device | 0.070198 s | 0.080250 s |

JAX FFI was about 45% slower in this run. Older reports without commit provenance
must not supersede a fresh baseline for the eventual implementation.

Reproduction command:

```sh
.venv/bin/python -m pytest -p no:capture \
  tests/jax/bench/test_parallel_benchmarks.py --runbench \
  --linarg-h5-path 1kg_chromosomes_n3202_blocks.h5 \
  --linarg-block-limit 2 --linarg-benchmark-k 8 \
  --linarg-parallel-processes 2 -q
```

Result: one benchmark test passed. The sandbox blocked POSIX shared memory;
the successful run used elevated execution. Passing the benchmark test is not
itself a numerical parity check or a performance acceptance gate.

## Decisions and work remaining

1. Confirm workload envelope and datasets: 1KG is available for development;
   biobank-scale validation remains necessary. A larger local dataset has not
   been selected.
2. Establish representative RHS widths, worker counts, dtype/accumulation policy,
   memory budget, accuracy tolerances, and performance acceptance thresholds.
3. Decide whether explicit full diagonal arrays are sufficient for target
   parameterizations or whether blockwise generation is required.
4. Verify a supported JAX version and native primitive API in an isolated
   compatibility experiment before changing package dependencies.
5. Prototype packed ragged execution, native scheduling, scratch ownership,
   concurrent-call safety, and reduction order. Measure oversubscription.
6. Verify shared and batched input/weight combinations, JVP/VJP, higher-order
   compositions, and eager/JIT consistency against independent dense references.
7. Compare native kernel cost, JAX wrapper overhead, compilation/loading time,
   repeated-call runtime, and peak memory separately.
8. Validate a differentiable genotype model and parameterized GRM solve; decide
   when to introduce the solver adapter.

The next step is a narrow prototype and measured baseline expansion, not an
assumption that HiJAX or native threading automatically beats Cython.

## Prototype milestone, 2026-11-06

The narrow prototype and baseline expansion above are now implemented under
[`../prototypes/jax_cpu/`](../prototypes/jax_cpu/README.md). This directory contains
durable source, tests, reproduction instructions, and two JSON measurement reports.
No production backend or dependency migration has been made.

- Python 3.12.13 / JAX 0.12.0.1 exposes the newer HiPrim interface. An isolated
  environment was used because production currently supports JAX <0.11.
- Ragged buffers are explicit runtime arguments. FFI tasks use JAX's own CPU
  thread pool with asynchronous completion and per-call XLA scratch buffers.
- Forward, transpose, general dynamic-diagonal GRM, and diagonal-gradient kernels
  are implemented. RHS tiling bounds native scratch width.
- Thirteen test cases pass, including differentiation of a parameterized solve
  and a genotype model, batching of both numerical arguments, higher derivatives,
  checkpointing, and concurrent calls.
- Real compressed-graph products agree with the existing Cython implementation.
- On eight 1KG blocks (1.07 million variants, 6,404 haplotype rows), four workers,
  and eight RHS columns, prototype forward/transpose products are approximately
  2.56x/2.16x faster. The general weighted product is 2.33x faster than composing
  Cython products. Single-vector operations are slightly slower.
- This is an Apple M1 development benchmark, not biobank-scale validation or a
  comparison against the specialized Cython GRM implementation. See the prototype
  report for exact conditions, numerical errors, and limitations.

Next: assess production migration, strengthen ingress/native boundary contracts,
optimize single-vector execution and unused cotangents, and establish larger-data
and memory acceptance gates. A custom HiType has not been needed for this milestone.

Fresh-process follow-up: on the same eight blocks, process-launch-to-first-transpose
medians were 1.320 s / 1.343 s for the prototype and 1.525 s / 1.590 s for Cython
(one/eight vectors, three interleaved fresh-process trials). The first prototype
trial was 4.897 s due mostly to imports; all raw trials are retained in
`../prototypes/jax_cpu/cold-eight-blocks.json`. OS filesystem caches were not
flushed. See the prototype README for loading-path differences and phase timings.

## Installed API and migration (2026-11-06)

The packed path is now integrated into `JaxLinearARG`, `JaxParallelOperator`, and
`JaxGRMOperator` under `Backend.FFI_CPU` (selected by `AUTO` when available).
Pure JAX remains an explicit fallback; its existing custom-VJP products retain
reverse-mode differentiation but do not gain the native path's forward-mode AD.

```python
import jax
from linear_dag.core.jaxlinarg import JaxParallelOperator, JaxGRMOperator

x = JaxParallelOperator.from_hdf5(
    "genotypes.h5", num_processes=4, max_num_traits=8,
    workspace_bytes=256 * 1024**2,
)
k = JaxGRMOperator(x, center=True)

# d(theta) is ordinary differentiable JAX code, with one weight per variant.
product = jax.jit(lambda theta, u: k.matmat(u, diagonal=d(theta)))
```

Alternatively, pass `diagonal=` when constructing the GRM. Without a supplied
diagonal, existing frequency-power weighting is preserved (zero for monomorphic
variants). Supplied weights replace frequency weighting; centering is independent.
Signed weights are allowed and do not guarantee positive semidefiniteness.
Native products support float32 and float64 (enable JAX x64 for float64).
Mapping over RHS/weights is supported; mapping over unrelated graphs is not.
Independently mapped diagonals currently use sequential whole-operation mapping.

The native wrapper holds one canonical packed graph, not padded copies of blocks.
Its `.blocks` property reconstructs inspection views on demand. Native CPU execution
uses one device: `num_processes` now means worker threads on this path, including
`rhe --jax-backend`. Explicit multi-device meshes remain a pure-JAX feature.
The workspace budget bounds native scratch, not graph/input/output storage or all
JAX AD/batching intermediates. It reduces workers and tile width to fit, and rejects
a budget that cannot hold one worker and one column.

The experimental HiJAX interface requires Python 3.12 and matching
JAX/jaxlib 0.12.0.1; Equinox is constrained to >=0.13.8,<0.14. Polars is kept below
2. SciPy is constrained below 1.18 because the existing SciPy LinearOperator
subclasses fail with its new `_xp` namespace requirement; migrating those
subclasses is separate work. HDF5 storage and graph construction are unchanged.

A built CPython 3.12 macOS ARM64 wheel was installed and exercised through the
public API. The real-data run in `../prototypes/jax_cpu/installed-eight-blocks.json`
used the same 8 blocks, 6,404 haplotypes, 1,067,520 variants, 4 workers, and stored
Cython oracle inputs/results as the prototype. Warm median milliseconds:

| RHS columns | X B | X.T U | X diag(d) X.T U |
| --- | ---: | ---: | ---: |
| 1 | 17.54 | 17.05 | 34.34 |
| 8 | 30.16 | 37.94 | 70.20 |
| 32 | 146.26 | 189.41 | 284.06 |

Seven synchronized repetitions per operation followed compilation and one warmup.
All relative L2 discrepancies against the saved Cython reference were below 1e-6.
Compilation took 5–7 ms for widths 1/8 and 30–37 ms for width 32. This is not a
fresh interleaved Cython speedup experiment or a cold-start measurement. The earlier
cold prototype timings should not be relabeled as installed-package timings.
Native scratch was 57,347,712 bytes. Larger-sample validation remains outstanding.


## Integration verification

- Built and installed a CPython 3.12 macOS ARM64 wheel; imports resolved to the
  installed package and native handler registration succeeded.
- Full installed-wheel suite: **474 passed, 7 skipped, 2 failed** in 146.84 s,
  using `JAX_ENABLE_X64=1 python -m pytest -p no:capture tests -q`.
  Both failures belong to the pre-existing untracked
  `tests/test_gtex_slurm_script.py`, whose shell-script work was not modified:
  `test_dispatch_uses_metadata_counts_and_chains_steps` expected a 0–2 task
  array but received 0–0; `test_legacy_step2_omits_recombination_method` failed
  because `python` was absent from its subprocess PATH. All JAX/RHE/CLI tests passed.
- Two explicit pure-JAX multi-device regressions passed with two virtual CPU
  devices. This includes coverage otherwise skipped in a one-device process.
- Final packed numerical/AD tests plus CLI help regression: **17 passed** with
  no global x64 environment setting; the packed test fixture scopes x64 itself.
- Ruff checks and `git diff --check` passed. The real-data installed-wheel parity
  and timing report is linked above.
- The existing project `.venv` remains Python 3.11; verification used the isolated
  `/private/tmp/linear-dag-hijax-py312` Python 3.12 environment. Recreate/install a
  Python 3.12 project environment before using the new pinned backend normally.
- Build operators/GRM views at the host boundary; pass them as PyTrees and vary
  RHS/diagonal arrays inside JAX transforms. Graph construction and validation
  are not traced computations.

No changes were committed. Larger-sample performance, non-Apple CPU measurements,
and faster batching for independently varying diagonals remain follow-up work.


## Follow-up RHE benchmark finding (2026-11-06)

The full 35-block CLI benchmark found a numerical limitation not covered by the
small-fixture integration tests: JAX RHE point estimates agree with Cython, but
standard errors become zero/NaN. Independent dense reproduction isolates float32
overflow in the RHE uncertainty denominator squared, outside the packed kernels.
This remains unfixed; do not treat the JAX RHE path as a validated replacement
for uncertainty reporting. Details and measured runtimes are in
`../prototypes/jax_cpu/rhe-performance-comparison.md`.


## Toy REML fit and memory benchmark (2026-11-06)

A two-variance-component REML fit on 256 diploid individuals and one real block
(150,192 variants) matched an independent Cython-built dense covariance reference.
Native CG-based likelihood gradients passed dense AD and central finite-difference
checks; both fits converged to the same estimates. No runaway memory growth was
observed: native process peak RSS was approximately 463 MiB for likelihood alone
and 480 MiB for value/gradient/fitting, with essentially flat warmed optimizer RSS.
Passing the graph dynamically avoids the larger compilation-memory cost seen
when closing over it. The cached dense covariance is vastly faster for this small
toy. The native path uses a cached exact spectrum for logdet, so this is not a
scalable matrix-free REML implementation. Details:
`../prototypes/jax_cpu/reml-performance-comparison.md`.


## Lineax adapter integration

Implemented `as_lineax_operator` as a public adapter for single-/multi-block JAX
genotype operators and JaxGRMOperator. Graph and dynamic diagonal arrays remain
PyTree leaves. Transpose uses explicit existing transpose products (also works
with pure-JAX custom-VJP paths). GRM symmetry is registered, with no inferred PSD
claim for potentially signed weights. Lineax >=0.1.1,<0.2 is a runtime dependency.
Acceptance checks cover products/transposes, JIT/vmap, explicit materialization,
CG composition and input/variance-component differentiation, signed weight AD,
and failed-solver result handling. Existing REML/RHE routines are not migrated.

Lineax verification completed: the source JAX suite passed 255 tests with 7
skipped. Built and installed the CPython 3.12 wheel, confirmed package imports
resolved to site-packages, and ran the 6 adapter regressions against that wheel.
These include CG input and variance-component gradients, JVPs, explicit transpose
products for both backends, and a check that iterative solves never call dense
materialization. Ruff and diff whitespace checks passed. The adapter was reviewed
locally for tagging, dtype/shape, PyTree and transpose contracts; no independent
agent review was used. Changes remain uncommitted.


## JAX hygiene review

Reviewed the promoted packed implementation, native kernel, operator wrappers,
Lineax adapter, and related package changes directly (no delegated review).

- Replaced opaque operation codes with named modes in Python/C++, and named the
  C++ metadata columns. Kept the FFI integer ABI unchanged. The internal HiJAX
  primitive now has an explicitly private name.
- Documented graph-buffer order, host validation boundary, native scratch layout,
  reuse of graph workspace slots, thread completion ordering, fixed-order reduction,
  bilinear derivative semantics, and rectangular Lineax materialization behavior.
- Formatted the new C++ kernel and scoped its helpers to its translation unit.
- Added public examples and corrected documentation for pure-JAX AD limits,
  native versus cached blockwise execution, construction outside transforms,
  sample alignment responsibility, and the allocation cost of block inspection.
- Added constructor/configuration types and replaced an opaque nested selection
  expression in GRM frequency weighting with a named safe intermediate.
- Fixed inconsistent execution-control validation: fractional worker counts, RHS
  widths, and scratch budgets now fail before backend dispatch, including on the
  pure-JAX path. Added six regression cases.

Validation: native extension/wheel rebuilt successfully; source JAX, RHE and CLI
suites passed **353 tests, 7 skipped** with `pytest -p no:capture`. Ruff and
`git diff --check` passed. Five warnings concern existing Polars horizontal-concat
deprecations outside this cleanup. No numerical algorithms or public operator
names changed. The known RHE uncertainty overflow remains a separate unresolved
correctness issue; this review does not certify that path at large GRM scales.
