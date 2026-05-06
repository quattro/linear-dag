# JAX LinearARG operator port Design

## Status
Draft

## Handoff Decision
- Current decision: blocked
- Ready for implementation: no
- Blocking items:
  - Pending plan review and explicit approval.

## Metadata
- Date: 2026-05-06
- Slug: jaxlinarg
- Artifact Directory: `.plans/design-plans/artifacts/2026-05-06-jaxlinarg`

## Summary

`JaxLinearARG` re-expresses the existing scipy/Cython `LinearARG` operator as a JAX-native `equinox.Module` so that `X @ w` and `X^T @ y` can run on accelerators, compose with JAX-based inference, and participate in autodiff. The core triangular solve over the variant-to-sample DAG is reformulated as a fixed-length `lax.scan` of edge updates with a precomputed `src_of_edge` array, giving static shapes under `jit`. A single `_solve` dispatch picks one of three interchangeable kernel implementations — a portable pure-JAX edge-scan trunk (CPU/GPU/TPU), a `jax.ffi` wrapper around the existing Cython solve (CPU fast path), and a Pallas-Triton kernel (GPU) — and a single `custom_vjp` rule wraps the dispatch so all backends share one autodiff rule, with the transpose-direction kernel acting as the VJP.

Genome-wide use is handled by `JaxParallelOperator`, an `equinox.Module` that places per-block `JaxLinearARG`s onto a `("blocks",)` `jax.sharding.Mesh` and uses `shard_map` (with a `psum` collective on the forward path) to replace the existing multiprocessing-based `_ParallelManager` for the JAX path, while leaving `LinearARG` and `ParallelOperator` untouched. To keep the JIT cache bounded across heterogeneous block sizes, ingress (`from_hdf5_block`, `from_lineararg`) pads each block to one of a small auto-chosen set of `(max_nodes, max_nnz)` buckets using identity self-loop edges with `data = 0`. Acceptance is measured against an existing oracle corpus on the `jax` branch, with a benchmark harness gating CPU performance within 2x of Cython at `k = 1` and requiring a GPU win at `k >= 8`.

## Problem Statement
The existing `linear_dag.core.lineararg.LinearARG` operator and its multiprocessing-based blockwise wrapper `parallel_processing.ParallelOperator` are scipy/Cython artifacts coupled to host CPUs and POSIX shared memory. Genotype-side workflows that need accelerator hardware (GPU/TPU), differentiable products of `X` (for gradient-based inference), or composition with JAX-native downstream code currently cannot use the operator without leaving the JAX execution model and paying host-device transfer costs at every product. The triangular sparse solve at the core of the operator is also not expressible as a generic dense or `jax.experimental.sparse` primitive, so a portable JAX implementation requires custom kernels.

This design covers a JAX-backed `JaxLinearARG` operator that lives in parallel with the existing `LinearARG`, exposes the same algebraic surface (matmat/rmatmat over sample/variant axes, including flipped variants), and runs on CPU (pure-JAX trunk and an FFI-wrapped Cython fast path), GPU (Pallas-Triton kernel), and TPU (pure-JAX trunk only, aspirational target). It includes a `shard_map`-based blockwise wrapper that replaces `_ParallelManager` for the JAX path while leaving `ParallelOperator` untouched.

## Definition of Done
1. `JaxLinearARG` computes `X @ w` and `X^T @ y` numerically equivalent (within float32 tolerance) to the existing scipy/Cython `LinearARG` on every case in the Phase 0 oracle corpus, including flipped variants.
2. Forward and reverse products are differentiable via `custom_vjp`; gradient through `w` / `y` matches finite-difference reference within tolerance.
3. Three kernel backends are wired and tested:
   - Pure-JAX edge-scan trunk that runs on CPU, GPU, and TPU.
   - CPU FFI wrapping the existing Cython solve via `jax.ffi`, selected automatically on CPU when available.
   - Pallas-GPU kernel selected automatically on CUDA devices.
4. Sharding integration: a `shard_map`-based wrapper over a `("blocks",)` mesh handles forward and reverse products across multiple operators, validated on a 2-device CPU mesh and intended to drop into multi-GPU contexts unchanged. This replaces `_ParallelManager` for the JAX code path; the existing `ParallelOperator` is left untouched and lives in parallel.
5. HDF5 ingestion: `JaxLinearARG.from_hdf5` and `JaxLinearARG.from_lineararg` materialize one or more blocks with auto-derived bucketing (~4–8 buckets per loaded set) so JIT recompiles do not occur per block.
6. CPU performance: pure-JAX trunk and the CPU-FFI path each stay within 2× of the existing Cython `_matmat` at `k = 1` on a representative production block. A benchmark harness produces this comparison as part of acceptance.
7. GPU performance: Pallas-GPU path measurably beats CPU at `k ≥ 8` on the same representative block. No fixed multiple required; "scale to GPU" is satisfied if there is a clear win.
8. Dependencies: `jax`, `equinox`, `jaxtyping` added as hard dependencies in `pyproject.toml`.
9. Test coverage: every backend parametrizes against the Phase 0 oracle corpus; sharding tests run on a 2-device CPU mesh; AD tests verify gradients vs. finite differences.

Out of scope:
- `GRMOperator`, RHE traces, `number_of_heterozygotes`, `mean_centered` / `normalized` JAX equivalents.
- Pallas-TPU custom kernel (pure-JAX trunk runs on TPU as portability fallback only).
- Deprecating or replacing `LinearARG` / `ParallelOperator`.

## Goals and Non-Goals
### Goals
- Provide a single-block JAX-native operator (`JaxLinearARG`) with the same algebraic surface as `LinearARG` (`matmat`, `rmatmat`, transpose view, flip correction) that runs on CPU/GPU/TPU through one Python API.
- Provide a genome-wide wrapper (`JaxParallelOperator`) that distributes blocks across a `("blocks",)` `jax.sharding.Mesh` using `shard_map`, replacing `_ParallelManager` for the JAX code path.
- Make `X @ w` and `X^T @ y` differentiable through `custom_vjp` so downstream JAX-based inference can use the operator as a building block.
- Support three kernel backends (pure-JAX edge-scan trunk, CPU FFI wrapping the existing Cython solve, Pallas-GPU) behind one dispatch point with automatic resolution from `jax.default_backend()`.
- Land bucketed padding so multiple blocks share JIT cache entries and `from_hdf5` is a single-call ingress.

### Non-Goals
- Porting `GRMOperator`, RHE traces, `number_of_heterozygotes`, `mean_centered`, or `normalized` views to JAX (out of scope; explored later).
- Writing a Pallas-TPU custom kernel. The pure-JAX trunk runs on TPU as a portability fallback only.
- Deprecating, replacing, or modifying `LinearARG`, `ParallelOperator`, or any Cython modules. The JAX path lives in parallel.
- Changing the on-disk HDF5 schema or the Cython graph-construction pipeline.
- Optimizing memory layout or kernel fusion beyond what falls out of static-shape JIT and `shard_map`.

## Existing Patterns
Investigation found the following authoritative patterns this design follows or interacts with:

- **Operator surface in `src/linear_dag/core/lineararg.py:38`.** `LinearARG` subclasses `scipy.sparse.linalg.LinearOperator`, exposes `_matmat`/`_rmatmat`/`_matvec`/`_rmatvec`, and applies flip correction in `_matmat` (lines 592–612) and `_rmatmat` (lines 614–631). `JaxLinearARG` mirrors this surface but uses `eqx.Module` and JAX arrays.
- **Triangular solve in `src/linear_dag/core/solve.pyx:14-262`.** Edge-by-edge sequential pass over CSC arrays with optional `nonunique_indices` column compression. The "edge-scan" reformulation in this design is the same algorithm with `src_of_edge` precomputed instead of derived from a nested loop.
- **Blockwise distribution in `src/linear_dag/core/parallel_processing.py:957-998`.** `_split_blocks` balances by `n_entries` across workers; `JaxParallelOperator.from_hdf5` reuses the same balancing logic for block-to-device assignment.
- **HDF5 ingestion in `src/linear_dag/core/lineararg.py:973-1048`.** `LinearARG.read` is the canonical block loader. Phase 3 of this design wraps it: read via `LinearARG.read`, convert via `from_lineararg`, pad to bucket, build `JaxLinearARG`. No new HDF5 codepaths.
- **Test infrastructure in `tests/jax/oracle.py` and `tests/jax/conftest.py`.** Phase 0 oracle corpus already in place; backend implementations parametrize against `oracle_case` plus a `kernel_backend` axis.
- **House-style conventions:** `eqx.Module` with jaxtyping shape contracts; `__check_init__` for invariants; `static=True` only for non-array metadata; `custom_vjp` over kernel calls (per `scientific-house-style:jax-equinox-numerics`). FCIS classification (per `scientific-house-style:functional-core-imperative-shell`): kernel modules and operator classes are Functional Core; HDF5 and FFI handler registration are Imperative Shell.

This design does not diverge from any existing pattern. It introduces new modules under a new subpackage; existing scipy/Cython code is untouched.

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: the operator and its triangular-solve algorithm are already implemented and validated in `linear_dag.core.lineararg.LinearARG` plus the Cython modules under `linear_dag.core.solve`. This work is a kernel/runtime port, not a model-design exercise. Numerical reference and acceptance behavior are derived from the existing implementation.
- User selection confirmation: confirmed by maintainer in the design conversation on 2026-05-06.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes (Phase 0 oracle corpus on the `jax` branch acts as the parity-target evidence; deeper kernel-by-kernel investigation occurs as part of the implementation phases)
- simulation_contract_complete_if_in_scope: n/a (no synthetic-data validation contract; oracle is the existing operator)

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | | | | |

## Model Option Analysis (Required When `suggested-model`)
| Candidate ID | Model Family | When It Fits | Key Assumptions | Failure Modes | Supporting Citation(s) | Selection Status |
| --- | --- | --- | --- | --- | --- | --- |
| MOD-1 | | | | | | selected/rejected |

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: re-express the algebraic surface of `LinearARG` (matmat / rmatmat / matvec / rmatvec, with flip correction) as a JAX-native `JaxLinearARG` operator that runs on CPU/GPU/TPU through one Python API, supports differentiation via `custom_vjp`, and integrates with `shard_map` for blockwise execution.
- Source selection confirmation: in-tree `linear_dag.core.lineararg` and `linear_dag.core.parallel_processing` on the `jax` branch are the source of truth. No external repos are pulled in.

### Source Pin
| Source ID | Source Type | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | `src/linear_dag/core/lineararg.py` | branch `jax` | Operator class + triangular-solve dispatch |
| PORT-SRC-2 | local-directory | `src/linear_dag/core/solve.pyx` | branch `jax` | Cython forward/backward solves and `nonunique_indices` compression |
| PORT-SRC-3 | local-directory | `src/linear_dag/core/parallel_processing.py` | branch `jax` | Reference for blockwise wrapper semantics (not ported, but informs sharding contract) |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface | Current Behavior | Target Behavior | Evidence Plan |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | numerics | `LinearARG._matmat(w)` returns `X @ w` via Cython forward triangular solve with flip correction. | `JaxLinearARG.matmat(w)` returns the same array within float32 tolerance for every backend. | Parametrize all backends across the Phase 0 oracle corpus (`tests/jax/oracle.py`); assert allclose against `case.Xw`. |
| PORT-BHV-2 | numerics | `LinearARG._rmatmat(y)` returns `X^T @ y` via Cython backward triangular solve with flip correction. | `JaxLinearARG.rmatmat(y)` returns the same array within float32 tolerance for every backend. | Same corpus harness; assert allclose against `case.XTy`. |
| PORT-BHV-3 | numerics | Forward/reverse products are non-differentiable. | `custom_vjp` over `_solve` with backward-direction kernel as VJP; gradients through `w`/`y` match finite differences within tolerance. | New AD test module checking JVP/VJP under `jit`. |
| PORT-BHV-4 | api | `from_hdf5(...)` reads one block via h5py + scipy CSC. | `JaxLinearARG.from_hdf5(path, block, backend, buckets)` reads via `LinearARG.read` then converts; also `from_lineararg(linarg, ...)`. | Round-trip test: load test fixture, compare against scipy reference on test inputs. |
| PORT-BHV-5 | api | `ParallelOperator` distributes blocks across processes via shared memory. | `shard_map`-based wrapper distributes blocks across mesh devices; same numerical outputs as sequential per-block accumulation. | 2-device CPU mesh test; compare to sum of per-block results. |
| PORT-BHV-6 | numerics | CPU performance set by Cython implementation. | Pure-JAX trunk and CPU-FFI path each within 2× of Cython `_matmat` at `k = 1` on a representative production block. | Benchmark harness committed alongside acceptance tests; reports ratio. |
| PORT-BHV-7 | numerics | No GPU implementation. | Pallas-GPU kernel beats CPU on the same block at `k ≥ 8`. | Same benchmark harness extended for GPU device. |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory` | `github-url`
- Investigation completion: yes|no
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence (file:line or commit:path:line) | Status (`confirmed`/`discrepancy`/`addition`/`missing`) |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | | | | confirmed |

## External Research Findings (When Triggered)
| Claim ID | Claim | Source URL | Source Type | Access Date | Confidence (high/med/low) |
| --- | --- | --- | --- | --- | --- |
| EXT-1 | | | official-doc/paper/standard/reference-implementation/secondary | | |

## Mathematical Sanity Checks
- Summary: The operator action `X w = (sample_indices) of (I - A)^{-1} P_{variant}(diag((-1)^flip) w) + sum(w[flip])` and its transpose are well-known sparse-triangular substitutions on a topologically-ordered DAG. Numerical correctness against the existing Cython implementation is the acceptance criterion (Phase 0 oracle corpus); no new mathematical content is introduced.
- Blocking issues: None.
- Accepted risks: Pure-JAX scan over `nnz` may accumulate slightly different rounding error than BLAS axpy in the Cython path. Tolerated at `rtol=1e-5, atol=1e-5` in float32; tighter checks at float64 if needed.

Detailed artifacts:
- `.plans/design-plans/artifacts/2026-05-06-jaxlinarg/model-symbol-table.md`
- `.plans/design-plans/artifacts/2026-05-06-jaxlinarg/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: portable JAX-first implementation that scales from CPU baseline to GPU acceleration, with TPU as aspirational portability target.
- Chosen strategy: pluggable kernel backends behind one dispatch (`PURE_JAX`, `FFI_CPU`, `PALLAS_GPU`); `Backend.AUTO` resolves from `jax.default_backend()`. One `custom_vjp` rule wraps the dispatch. The triangular solve is reformulated as a single edge-scan over precomputed `src_of_edge` so shapes are static under jit.
- Why this strategy: The triangular solve has strict sequential dependence and is not expressible via stock `jax.experimental.sparse` or Lineax solvers; a custom kernel is required. Universal trunk in pure JAX preserves TPU portability; FFI and Pallas are layered as opt-in accelerators without forking the API. `custom_vjp` over the dispatch (with `backend` as `nondiff_argnum`) gives autodiff symmetrically across all backends with one rule.

## Solver Translation Feasibility
- Summary: Feasible. The Cython kernel maps to a `lax.scan` of fixed length `nnz` whose carry is the dense RHS buffer `b`; per-step work is one scatter-add. The CPU FFI path reuses the existing Cython `.so`. The Pallas-GPU path is a Triton kernel with one block, sequential over edges, vectorized across `k`; optional level-scheduling preprocessing extracts wavefront parallelism for wide DAGs.
- Blocking constraints: `lax.scan` step overhead may dominate at small `k`; bounded by the 2× CPU bar in DoD #6, mitigated by FFI fallback. Pallas-TPU is intentionally out of scope.
- Custom-solver rationale: domain-specific kernel; Lineax `AutoLinearSolver` and `jax.experimental.sparse` do not provide a sparse triangular solve over CSC adjacency in topological order. Future work could expose `JaxLinearARG` as a `lineax.AbstractLinearOperator` for composition, but is not in this design.

Detailed artifact:
- `.plans/design-plans/artifacts/2026-05-06-jaxlinarg/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract: HDF5 file path or in-memory `LinearARG` instance. Inputs are converted to `(indptr, indices, data, src_of_edge, variant_indices, flip, sample_indices)` with explicit dtypes (`int32`, `float32`/`float64`, `bool_`) and bucket-padded to a `(max_nodes, max_nnz)` target. Per-block `LinearARG.read` is the sole HDF5 codepath; no new file format support.
- Rejection rules: reject non-CSC adjacencies, non-lower-triangular adjacencies, sample columns containing edges, and shape mismatches between `indices`/`data`/`src_of_edge`. Validated in `JaxLinearARG.__check_init__` (raises `ValueError` at construction).

### Pipeline
- Contract: `JaxLinearARG.matmat(w)` and `JaxLinearARG.rmatmat(y)` accept `(n_variants, k)` and `(n_samples, k)` JAX arrays respectively (1D inputs auto-promoted to 2D). `JaxParallelOperator.matmat` accepts `(genome_n_variants, k)` and dispatches block slices internally.
- Validation-first checks: shape compatibility verified before invoking the kernel. Backend resolution (`Backend.AUTO`) cached on the operator after first call.

### Numerics
- Contract: kernels are pure functions of the form `_SolveFn(indptr, indices, data, src_of_edge, b) -> b'` where `b` is `(N, k)` Fortran-order; both forward and backward variants conform. One `custom_vjp` rule wraps `_solve(backend, ...)` with `backend` as `nondiff_argnum`; backward VJP calls the transpose-direction kernel.
- Result/status semantics: exception-first. Boundary errors raise Python exceptions before entering jit. Traced kernels do not raise; padding rows have `data = 0` so they contribute identity. No `result` channels in the operator surface.

### Egress
- Contract: returns `jax.Array` matching downstream JAX usage; no implicit numpy conversion. Multi-block forward returns one `(n_samples, k)` array (sum-reduced across `("blocks",)` mesh axis); multi-block reverse returns `(n_variants, k)` concatenated along the variant axis (no collective).
- Output/exit-code mapping: not applicable (library, no CLI surface in this plan).

## Data Conversion and Copy Strategy
| Source | Target | Copy mode | Rationale |
| --- | --- | --- | --- |
| `scipy.sparse.csc_matrix` (from `LinearARG.A`) | `jnp.int32` indptr/indices, `jnp.float32` data | `single-copy fallback` | scipy stores as native dtypes; one host→device copy when ingressing onto an accelerator. Acceptable because ingress is once per block at construction. |
| h5py datasets (HDF5 read) | numpy arrays then JAX arrays | `single-copy fallback` | `LinearARG.read` already loads via h5py to numpy; we then convert with `jnp.asarray`. Memory layout is reset to Fortran-order for CSC. |
| `flip` Polars Series / numpy bool | `jnp.bool_` | `single-copy fallback` | small array, negligible cost. |
| `src_of_edge` (computed) | `jnp.int32` | host-side then transferred | computed once via `np.repeat` in `compute_src_of_edge`; cached on the `JaxLinearARG`. |

## Multi-Input Reconciliation Contract (Required When Multiple Tabular Sources Feed Numerics)
Not applicable. This design has one input source per block (the HDF5-stored `LinearARG`); no multi-source reconciliation occurs in scope. Genome-wide assembly is array concatenation along the variant axis using `variant_offsets`, not a tabular join.

## Validation Strategy
- Boundary checks: HDF5 file existence, block-name validity, mesh device count > 0, bucket spec well-formed (sorted, monotone), `Backend` value in enum.
- Shape/range/domain checks: `__check_init__` on every `eqx.Module` enforces shape consistency (`indices.shape == data.shape == src_of_edge.shape`, `indptr.shape == (max_nodes + 1,)`, `n_real_nodes <= max_nodes`, `n_real_edges <= max_nnz`). `JaxParallelOperator.__check_init__` verifies all blocks share `mesh`, `n_samples`, and a backend, and that `variant_offsets` is monotone non-decreasing of length `n_blocks + 1`.
- Multi-input alignment checks: not applicable (single source per block).
- Failure semantics: boundary failures raise `ValueError` at construction. Traced kernel failures (e.g., gradient through a padded edge) are bug surfaces — caught by AD tests, not runtime checks. FFI handler missing emits `UserWarning` and falls back to `PURE_JAX` (per design Q above).

## Testing and Verification Strategy
- TDD scope: every kernel function written test-first against the Phase 0 oracle corpus. AD tests written before `custom_vjp` rule is wired. Sharding wrapper written test-first against a 2-device CPU mesh.
- Regression strategy: every backend × every `OracleCase` case parametrized; failures pin to the smallest case for debugging. Performance regressions caught by the `--runbench` harness with CI thresholds.
- Verification commands: `pytest tests/jax/`, `pytest tests/jax/ -k bench --runbench`, `python -m linear_dag.jaxlinarg.bench --device cpu --device gpu` (final form may differ).

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Subpackage scaffold and dependency lift
**Goal:** Lift `jax_lineararg.py` into a `jaxlinarg/` subpackage with stable internal boundaries; add `jax`, `equinox`, `jaxtyping` as hard dependencies.

**Components:**
- New subpackage `src/linear_dag/core/jaxlinarg/` with module-level responsibilities: `operator.py` (single-block `JaxLinearARG`, `_TransposeView`, `Backend`, `_solve` dispatch and `custom_vjp` wiring), `kernels/__init__.py` plus per-backend modules `kernels/pure_jax.py`, `kernels/ffi_cpu.py`, `kernels/pallas_gpu.py`, `kernels/pallas_tpu.py` (all sharing one `_SolveFn` Protocol), `padding.py` (`compute_src_of_edge`, `pad_to_bucket`, `choose_bucket`), `wrapper.py` (`JaxParallelOperator`), and `ingress.py` (HDF5/`LinearARG` adapters; FCIS Imperative Shell).
- Migrate the existing `src/linear_dag/core/jax_lineararg.py` content into `operator.py`; preserve public re-exports from `linear_dag.core.jaxlinarg`.
- Update `pyproject.toml` to add `jax`, `equinox`, `jaxtyping` to `dependencies`; pin minimum versions and exclude any known-bad releases.
- Update `src/linear_dag/core/__init__.py` to export `JaxLinearARG`, `JaxParallelOperator`, `Backend` from the new subpackage.

**Dependencies:** Phase 0 (oracle corpus already on `jax` branch).

**Done when:** `pip install -e .` succeeds; `from linear_dag.core import JaxLinearARG, JaxParallelOperator, Backend` succeeds; existing `tests/jax/test_oracle.py` still passes.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Pure-JAX kernel + scatter/gather + custom_vjp
**Goal:** Implement the universal-trunk kernel and wire `custom_vjp`; the operator returns numerically correct results on CPU at every oracle-corpus case.

**Covered ACs:** `jaxlinarg.AC1.*` (PURE_JAX backend correctness), `jaxlinarg.AC3.*` (autodiff correctness).

**Components:**
- `kernels/pure_jax.py`: `_pure_jax_solve_forward` and `_pure_jax_solve_backward` as `lax.scan` of fixed length `nnz` over `(src_of_edge, indices, data)`; carry is the dense `(N, k)` buffer.
- `operator.py`: scatter/gather around the solve (variant flip handling, sample/variant gather, row-sum correction); `_solve` dispatch with `Backend.PURE_JAX` registered; `custom_vjp` rule with backward calling the transpose kernel.
- New tests under `tests/jax/test_kernels_pure_jax.py` parametrized over `oracle_case`; AD tests (JVP/VJP vs finite differences under `jit`) under `tests/jax/test_autodiff.py`.

**Dependencies:** Phase 1.

**Done when:** every `oracle_case` passes `np.allclose(jax_op @ case.w, case.Xw, rtol=1e-5, atol=1e-5)` and the symmetric reverse check; AD tests match finite differences within `1e-3` at float32.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Bucketing and ingress
**Goal:** Single-call construction from HDF5 or `LinearARG` with auto-derived bucketing; multiple blocks reuse JIT cache entries.

**Covered ACs:** `jaxlinarg.AC4.*` (ingress + bucketing).

**Components:**
- `padding.py`: `compute_src_of_edge`, `pad_to_bucket` (no-op self-loop edges with `data = 0`), `choose_bucket` (greedy bucket selection minimizing total padded volume, capped at 8 buckets).
- `ingress.py`: `JaxLinearARG.from_lineararg(linarg, *, backend, bucket)` and `JaxLinearARG.from_hdf5_block(path, block, *, backend, bucket)`; this module is FCIS Imperative Shell (h5py reads).
- New tests under `tests/jax/test_padding.py` (round-trip: pad then un-pad recovers original adjacency; numerical correctness of padded forward/reverse vs unpadded) and `tests/jax/test_ingress.py` (HDF5 round-trip on the test fixture; JIT cache reuse via call counter).

**Dependencies:** Phase 2.

**Done when:** all corpus cases pass when run through the padded path; HDF5 ingestion of `tests/testdata/test_chr21_50.h5` produces operators numerically equivalent to `LinearARG.read(path, block)`; two operators sharing a bucket trigger one trace, not two.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: CPU FFI backend
**Goal:** Wrap the existing Cython `spsolve_*_triangular_matmat` via `jax.ffi`; selected automatically on CPU when the handler is registered.

**Covered ACs:** `jaxlinarg.AC2.*` (FFI_CPU correctness), `jaxlinarg.AC5.*` (FFI fallback warning).

**Components:**
- `kernels/ffi_cpu.py`: thin C ABI shim around `spsolve_forward_triangular_matmat` and its backward sibling (compiled via the existing Cython build); `jax.ffi.register_ffi_target` calls at import; `_ffi_cpu_solve_forward` / `_backward` invoke `jax.ffi.ffi_call` with the registered symbols.
- Build-system update: extend `hatch_build.py` (Cython hook) to also expose the FFI symbols. Document the build-time requirement in module docstring.
- `operator.resolve_backend`: extend to detect FFI handler registration; on `cpu` platform without the handler, emit `UserWarning` and fall back to `PURE_JAX`.
- Tests under `tests/jax/test_kernels_ffi_cpu.py` parametrized over `oracle_case`, plus a `test_ffi_fallback.py` verifying the warning is emitted when the handler is absent (monkey-patched).

**Dependencies:** Phase 2 (kernel signature stable).

**Done when:** every corpus case passes on `Backend.FFI_CPU` when handler is registered; fallback warning fires and tests pass on `Backend.PURE_JAX` when handler is absent.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: JaxParallelOperator and shard_map sharding
**Goal:** Genome-wide wrapper distributes blocks across a `("blocks",)` mesh; sharding works on a 2-device CPU mesh.

**Covered ACs:** `jaxlinarg.AC6.*` (sharding correctness and structure).

**Components:**
- `wrapper.py`: `JaxParallelOperator` `eqx.Module` (per-block `tuple[JaxLinearARG, ...]`, `variant_offsets`, mesh, bucket layout, block-to-bucket and block-to-device assignments); classmethods `from_hdf5` and `from_lineargs` with `buckets="auto"` default; `matmat`/`rmatmat` decorated with `shard_map` (forward `psum` over `"blocks"`; reverse no collective); `_split_blocks`-style `n_entries`-balanced device assignment reusing the algorithm from `parallel_processing.py:957`.
- Per-block `allele_frequencies` exposed on `JaxLinearARG` (genome-wide concatenation left to callers per design decision).
- Tests under `tests/jax/test_wrapper.py` on a 2-device CPU mesh comparing against sequential per-block accumulation; `tests/jax/test_block_assignment.py` verifying balanced splits.

**Dependencies:** Phase 3.

**Done when:** `JaxParallelOperator.matmat` and `rmatmat` produce values numerically equivalent to a Python loop over per-block `JaxLinearARG` calls on a 2-device CPU mesh; block-to-device assignment matches `_split_blocks` output for the same metadata.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Pallas-GPU kernel
**Goal:** GPU kernel beating CPU at moderate `k`; level-scheduling preprocessing available behind a flag for wide DAGs.

**Covered ACs:** `jaxlinarg.AC2.*` (PALLAS_GPU correctness), `jaxlinarg.AC7.*` (GPU performance).

**Components:**
- `kernels/pallas_gpu.py`: `_pallas_gpu_solve_forward` and `_backward` Pallas-Triton kernels — single CUDA block, sequential edge-scan, threads vectorized across `k`. Memory layout: `b` in shared memory when feasible, otherwise direct HBM with coalesced reads.
- Optional level scheduling: precompute `level_of_node` (one BFS pass per block at construction) and a per-level kernel variant launched as wavefronts. Behind `JaxLinearARG(..., level_schedule=False)` flag; default off; design includes the wiring, implementation deferred unless performance gates require it.
- Tests under `tests/jax/test_kernels_pallas_gpu.py` parametrized over `oracle_case`, skipped when `jax.default_backend() != "gpu"`.

**Dependencies:** Phase 2 (kernel signature), Phase 5 (so multi-block GPU runs share infrastructure).

**Done when:** every corpus case passes on `Backend.PALLAS_GPU` when GPU is available; level-scheduling flag round-trips a numerically correct result (even if the level-scheduled kernel itself is an alias of the serial kernel until performance gates require it).
<!-- END_PHASE_6 -->

<!-- START_PHASE_7 -->
### Phase 7: Performance harness and acceptance gates
**Goal:** Quantify CPU and GPU performance against the existing Cython implementation; gate CI.

**Covered ACs:** `jaxlinarg.AC8.*` (CPU performance gate), `jaxlinarg.AC7.*` (GPU performance gate).

**Components:**
- `tests/jax/bench/` benchmark module producing a markdown comparison table for `LinearARG._matmat` vs each available JAX backend at `k ∈ {1, 8, 64}` on a representative production block (configurable, default falls back to the test fixture for CI without large data).
- pytest plugin glue: `--runbench` opt-in flag; gates on `PURE_JAX_CPU / Cython ≤ 2.0` at `k = 1` and (when GPU available) `PALLAS_GPU < CPU` at `k ≥ 8`. Below-threshold runs fail the harness.
- Documentation update: short "Benchmarking" section in module docstring or top-level README pointing at the harness.

**Dependencies:** Phases 2, 4, 6.

**Done when:** harness produces a deterministic table; CI thresholds pass on the reference machine for the available backends; the design's CPU and GPU gates from DoD #6 and #7 are mechanically verified.
<!-- END_PHASE_7 -->


## Simulation And Inference-Consistency Validation
- In scope: yes|no
- Simulate entrypoint/signature:
- Inputs:
- Outputs:
- Seed/RNG policy:

### Assumption Alignment
| Inference Assumption | Simulation Rule | Mismatch Risk | Mitigation |
| --- | --- | --- | --- |
| | | | |

### Planned Validation Experiments
| Experiment ID | Type (recovery/SBC/PPC) | Success Criterion | Notes |
| --- | --- | --- | --- |
| SIM-1 | | | |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | Pure-JAX trunk fails the 2× CPU bar at `k = 1` on production-shape blocks. | High | FFI fallback (Phase 4) is precisely this mitigation; CPU FFI auto-selected when handler is registered. If both fail, the trunk is still usable for AD/GPU/TPU paths. | Implementation lead |
| R2 | `shard_map` + `custom_vjp` interaction has subtle edge cases (gradients across shards, residuals captured per shard). | Medium | Phase 5 includes AD-through-sharding test on the 2-device CPU mesh before declaring done. | Implementation lead |
| R3 | FFI handler build/registration differs across macOS / Linux / wheels. | Medium | Use `jax.ffi` (stabilized API); test under both platforms in CI; document fallback path so absence does not break CPU users. | Implementation lead |
| R4 | Pallas-GPU kernel correctness, especially for level-scheduling variant. | Medium | Default off; serial scan remains the correctness baseline; level scheduling is enabled only after passing the same oracle corpus. | Implementation lead |
| R5 | Auto-bucketing produces too many buckets for heterogeneous block sizes, defeating the JIT-cache goal. | Low | `choose_bucket` capped at 8 buckets; if a corpus exceeds that diversity, log and let the user override with explicit `buckets=[(N1, nnz1), ...]`. | Implementation lead |
| Q1 | Where exactly does the FFI C-ABI shim live in the build (Cython hook vs separate C file)? | Open | Resolved during Phase 4; current design assumes extending `hatch_build.py` Cython hook. | Implementation lead |
| Q2 | Is per-block `LinearARG.read` fast enough for the genome-wide construction path, or do we need a batched HDF5 reader? | Open | Profiled during Phase 5; if it dominates construction time, add a batched reader as a Phase 8 follow-up (out of scope for this plan). | Implementation lead |

## Additional Considerations
<!-- Optional: include only when relevant -->

## Acceptance Criteria

### `jaxlinarg.AC1`: Pure-JAX trunk produces correct algebraic outputs
- **jaxlinarg.AC1.1 Success:** `JaxLinearARG.matmat(case.w)` matches `case.Xw` within `rtol=1e-5, atol=1e-5` for every `OracleCase` in the Phase 0 corpus on `Backend.PURE_JAX`.
- **jaxlinarg.AC1.2 Success:** `JaxLinearARG.rmatmat(case.y)` matches `case.XTy` within `rtol=1e-5, atol=1e-5` for every `OracleCase`.
- **jaxlinarg.AC1.3 Success:** `JaxLinearARG.matmat` correctly handles flipped variants (cases with `flip_prob > 0`).
- **jaxlinarg.AC1.4 Edge:** 1D inputs (`w` shape `(n_variants,)`) auto-promote to `(n_variants, 1)` and the output drops the trailing axis symmetrically.
- **jaxlinarg.AC1.5 Failure:** Mismatched input shape raises `ValueError` before invoking the kernel.

### `jaxlinarg.AC2`: Backend kernels are interchangeable
- **jaxlinarg.AC2.1 Success:** `Backend.FFI_CPU` produces outputs equivalent to `Backend.PURE_JAX` within tolerance on every corpus case (when handler is registered).
- **jaxlinarg.AC2.2 Success:** `Backend.PALLAS_GPU` produces outputs equivalent to `Backend.PURE_JAX` within tolerance on every corpus case (when `jax.default_backend() == "gpu"`).
- **jaxlinarg.AC2.3 Success:** `Backend.AUTO` resolves to the expected concrete backend on each platform (`cpu` → `FFI_CPU` if available else `PURE_JAX`; `gpu` → `PALLAS_GPU`; `tpu` → `PURE_JAX`).
- **jaxlinarg.AC2.4 Failure:** Unknown `Backend` enum value raises `ValueError` at operator construction.

### `jaxlinarg.AC3`: Forward and reverse products are differentiable
- **jaxlinarg.AC3.1 Success:** `jax.grad` of `0.5 * jnp.sum((op @ w - target) ** 2)` matches a finite-difference reference within `1e-3` (float32) on small corpus cases under `jit`.
- **jaxlinarg.AC3.2 Success:** Symmetric AD check holds for `op.T @ y`.
- **jaxlinarg.AC3.3 Edge:** Gradient is zero for entries of `w` whose corresponding variant has no path to a sample.

### `jaxlinarg.AC4`: HDF5 ingestion and bucketing work end-to-end
- **jaxlinarg.AC4.1 Success:** `JaxLinearARG.from_hdf5_block(path, block)` produces an operator numerically equivalent to `LinearARG.read(path, block)` on test inputs.
- **jaxlinarg.AC4.2 Success:** `pad_to_bucket` followed by `compute_src_of_edge` round-trips: padded forward solve agrees with unpadded forward solve within tolerance.
- **jaxlinarg.AC4.3 Success:** Two operators sharing a bucket trigger one `jit` trace, not two (verified via cache hit counter).
- **jaxlinarg.AC4.4 Success:** `choose_bucket` produces ≤ 8 buckets for any input set; explicit `buckets=[…]` override is respected.
- **jaxlinarg.AC4.5 Failure:** Padding shrinkage (`max_nodes < N` or `max_nnz < nnz`) raises `ValueError`.

### `jaxlinarg.AC5`: FFI fallback is observable and graceful
- **jaxlinarg.AC5.1 Success:** When the FFI handler is absent and `Backend.FFI_CPU` is requested, a `UserWarning` is emitted and the operator falls back to `Backend.PURE_JAX`.
- **jaxlinarg.AC5.2 Success:** When the FFI handler is registered, `Backend.AUTO` resolves to `FFI_CPU` on CPU.

### `jaxlinarg.AC6`: Sharding wrapper distributes correctly
- **jaxlinarg.AC6.1 Success:** `JaxParallelOperator.matmat(w)` on a 2-device CPU mesh equals the sum of per-block `JaxLinearARG.matmat` calls on the same blocks within tolerance.
- **jaxlinarg.AC6.2 Success:** `JaxParallelOperator.rmatmat(y)` on a 2-device CPU mesh equals the variant-axis-concatenated per-block reverse calls.
- **jaxlinarg.AC6.3 Success:** Block-to-device assignment matches `_split_blocks` (n_entries-balanced) for the same metadata.
- **jaxlinarg.AC6.4 Success:** AD through `JaxParallelOperator.matmat` on a 2-device CPU mesh matches per-block AD aggregated.
- **jaxlinarg.AC6.5 Failure:** Mesh with zero devices, or mesh axis name other than `"blocks"`, raises `ValueError` at construction.

### `jaxlinarg.AC7`: GPU performance scales beyond CPU
- **jaxlinarg.AC7.1 Success:** When GPU is available, `Backend.PALLAS_GPU / Backend.PURE_JAX_CPU < 1.0` (i.e., GPU faster) at `k = 8` on the reference block in the benchmark harness.
- **jaxlinarg.AC7.2 Success:** Same gate holds at `k = 64`.

### `jaxlinarg.AC8`: CPU performance stays within 2× of Cython
- **jaxlinarg.AC8.1 Success:** `Backend.PURE_JAX` runtime / `LinearARG._matmat` runtime ≤ 2.0 at `k = 1` on the reference block in the benchmark harness.
- **jaxlinarg.AC8.2 Success:** `Backend.FFI_CPU` runtime / `LinearARG._matmat` runtime ≤ 2.0 at `k = 1` on the reference block (when handler is registered).

## Glossary

- **LinearARG**: Existing in-tree operator (`linear_dag.core.lineararg.LinearARG`) representing a genotype matrix `X` as a sparse triangular DAG; the parity target for this port.
- **Linear ARG (Ancestral Recombination Graph)**: DAG encoding shared haplotype structure such that `X @ w` is computable as a triangular sparse solve over the graph adjacency.
- **matmat / rmatmat / matvec / rmatvec**: scipy `LinearOperator` action methods for `X @ W`, `X^T @ Y`, `X @ w`, and `X^T @ y` respectively; the algebraic surface being mirrored in JAX.
- **Flip correction**: Per-variant sign and offset adjustment applied around the triangular solve to account for variants stored in flipped (reference/alternate-swapped) orientation.
- **CSC adjacency / `indptr` / `indices` / `data`**: Compressed sparse column representation of the DAG edges; per-edge `src_of_edge` is precomputed from `indptr` so the kernel scan is one flat pass instead of nested loops.
- **Edge-scan trunk**: This design's reformulation of the Cython triangular solve as a single `lax.scan` of length `nnz` whose carry is the dense `(N, k)` RHS buffer.
- **Equinox (`eqx.Module`)**: JAX library providing PyTree-registered dataclass-style modules with `__check_init__` invariant hooks; used here to bundle operator state plus static metadata.
- **jaxtyping**: Runtime/static shape and dtype annotation library used to express array contracts on module fields and function signatures.
- **`custom_vjp`**: JAX mechanism for defining a custom backward (vector-Jacobian product) rule; here it wraps `_solve` with `backend` as a `nondiff_argnum` so all kernels share one autodiff rule.
- **`jax.ffi`**: Stabilized JAX foreign-function interface used to call the existing Cython `spsolve_*_triangular_matmat` symbols from inside `jit`.
- **Pallas / Pallas-Triton**: JAX kernel-authoring DSL; the GPU backend lowers through Triton to a single CUDA block performing the sequential edge scan with `k`-vectorized threads.
- **Lineax / `AbstractLinearOperator`**: JAX linear-operator/solver library considered and rejected here because it does not provide a sparse triangular solve over CSC adjacency in topological order.
- **`jax.experimental.sparse`**: JAX's stock sparse primitives, also rejected as insufficient for the strict sequential dependence of this triangular solve.
- **`shard_map`**: JAX collective-style sharding transform that maps a function across a named `Mesh` axis with explicit per-shard semantics; used to distribute blocks across the `("blocks",)` axis.
- **`jax.sharding.Mesh`**: Named device grid (here a 1D `("blocks",)` mesh) used as the substrate for `shard_map`.
- **`psum`**: JAX collective summing across a named mesh axis; used on the forward path to combine per-block `(n_samples, k)` contributions.
- **`_ParallelManager` / `ParallelOperator`**: Existing multiprocessing + POSIX-shared-memory blockwise wrapper being replaced *only* on the JAX code path by `JaxParallelOperator`.
- **Bucketing / `choose_bucket` / `pad_to_bucket`**: Strategy for grouping heterogeneous block shapes onto a small set of `(max_nodes, max_nnz)` targets via identity self-loop padding so JIT traces are reused across blocks.
- **Level scheduling**: Optional per-block BFS preprocessing that groups DAG nodes into wavefronts to expose parallelism on GPU; wired but disabled by default.
- **Oracle corpus**: Pre-built parametrized test cases under `tests/jax/oracle.py` providing reference `(w, Xw, y, XTy)` quadruples from the Cython implementation.
- **FCIS (Functional Core / Imperative Shell)**: House-style separation rule classifying kernels and operator classes as pure (Functional Core) and HDF5/FFI registration as side-effecting (Imperative Shell).
- **Definition of Done (DoD) / Acceptance Criteria (AC)**: Document conventions naming the binding completion and pass/fail gates referenced throughout the phases.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-05-06 | N/A | Draft | Plan created | |
