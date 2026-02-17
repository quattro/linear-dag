# Core from_hdf5 Constructor Unification Design

## Summary
This design unifies `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` under one constructor contract and one internal setup pipeline in `src/linear_dag/core/parallel_processing.py`. Both classmethods will accept the same parameters and defaults, including `max_num_traits=8`, while keeping class-specific compute concerns isolated (worker callable, shared-memory layout, output shape, and GRM-specific `alpha` behavior).

The shared pipeline centralizes metadata resolution, optional BED loading, filtering-count derivation, manager bootstrap, worker startup, and IID loading. This reduces drift between operators, makes filtering behavior consistent (including GRM), and keeps process lifecycle guarantees explicit. The rollout is phase-based and test-driven, with mandatory downstream `requesting-code-review` loop completion (zero findings) before implementation is considered done.

## Definition of Done
- `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` expose the same public signature, including matching default values (`max_num_traits=8` per your choice).
- Both constructors execute a shared setup pipeline (block metadata loading/filtering, BED handling, filtered variant counting where applicable, shared-memory spec creation, manager creation, worker startup, IID loading), with only class-specific pieces separated (worker entrypoint, SHM layout, output shape, and GRM-only math logic like `alpha`).
- Parallel-processing lifecycle follows current best practices for Python multiprocessing/shared memory and HDF5 process access, and the design documents concrete guardrails for reliability/cleanup.
- Existing CLI call sites and tests remain compatible or are updated in a deliberate, explicit way, with behavior parity verified by tests.

## Acceptance Criteria
### core-from-hdf5-constructor-unification.AC1: Constructor Contract Is Unified
- **core-from-hdf5-constructor-unification.AC1.1 Success:** `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` expose identical parameter names and ordering.
- **core-from-hdf5-constructor-unification.AC1.2 Success:** Both constructors use `max_num_traits=8` by default.
- **core-from-hdf5-constructor-unification.AC1.3 Success:** Both constructors accept the unified parameter set; `alpha` is GRM-specific in effect and does not create required state or behavioral dependency in `ParallelOperator`.
- **core-from-hdf5-constructor-unification.AC1.4 Failure:** Invalid `num_processes` (`<1`) raises a deterministic, user-facing `ValueError`.

### core-from-hdf5-constructor-unification.AC2: Shared Construction Pipeline Is Used
- **core-from-hdf5-constructor-unification.AC2.1 Success:** Both constructors execute shared metadata resolution, optional BED loading, filter-count computation, manager creation, worker startup, and IID loading.
- **core-from-hdf5-constructor-unification.AC2.2 Success:** Class-specific behavior/state is isolated to worker callable, SHM layout/handle binding, output shape, and operator-specific state (`alpha` ownership/use in GRM math paths only).
- **core-from-hdf5-constructor-unification.AC2.3 Failure:** Worker startup or worker execution errors are surfaced via the existing runtime error path and do not leave orphaned worker processes.

### core-from-hdf5-constructor-unification.AC3: Filtering Semantics Match Across Operators
- **core-from-hdf5-constructor-unification.AC3.1 Success:** MAF-only filtering yields GRM outputs matching serial filtered baselines.
- **core-from-hdf5-constructor-unification.AC3.2 Success:** BED+MAF filtering yields GRM outputs matching serial filtered baselines.
- **core-from-hdf5-constructor-unification.AC3.3 Edge:** Unfiltered constructor behavior remains numerically consistent with current baseline tests.
- **core-from-hdf5-constructor-unification.AC3.4 Failure:** Filtering mismatch between constructor metadata and worker computation is caught by tests (shape or numeric mismatch).

### core-from-hdf5-constructor-unification.AC4: Parallel Lifecycle Remains Safe
- **core-from-hdf5-constructor-unification.AC4.1 Success:** Context-managed operator shutdown continues to close workers and release shared memory without leaks in normal paths.
- **core-from-hdf5-constructor-unification.AC4.2 Success:** Borrowed/shared memory views are closed safely before manager teardown.
- **core-from-hdf5-constructor-unification.AC4.3 Failure:** Error-path teardown still performs cleanup and prevents zombie worker persistence.

### core-from-hdf5-constructor-unification.AC5: Call-Site Compatibility Is Preserved
- **core-from-hdf5-constructor-unification.AC5.1 Success:** Existing assoc/score/rhe workflows continue to run with compatible behavior.
- **core-from-hdf5-constructor-unification.AC5.2 Success:** CLI forwarding tests verify expected kwargs passed to both constructor paths.
- **core-from-hdf5-constructor-unification.AC5.3 Edge:** Passing non-default `alpha` to `ParallelOperator` (if accepted for signature parity) is a documented no-op and does not change outputs.
- **core-from-hdf5-constructor-unification.AC5.4 Edge:** Existing tests outside unification scope continue to pass unchanged.

### core-from-hdf5-constructor-unification.AC6: Review Gate Is Enforced
- **core-from-hdf5-constructor-unification.AC6.1 Success:** Implementation completion requires running the `requesting-code-review` review/fix/re-review loop until zero unresolved issues remain.

## Glossary
- **LinearARG**: A sparse linear-operator representation of genotype data loaded from block-based HDF5 groups.
- **GRM (Genetic Relatedness Matrix)**: Sample-by-sample matrix induced by genotype similarity, represented here as a `LinearOperator`.
- **MAF (Minor Allele Frequency) filtering**: Variant filtering based on allele-frequency threshold(s), optionally stratified by BED region membership.
- **BED regions**: Genomic intervals loaded from a BED file and interpreted as UCSC 0-based half-open coordinates.
- **Shared memory (SHM)**: Process-shared buffers used to exchange trait/block data between parent and worker processes.
- **Worker process**: A subprocess that loads block-local `LinearARG` objects and executes matmul-related operations based on flag signaling.
- **Manager bootstrap**: The constructor phase that partitions blocks, allocates SHM, spawns workers, and initializes runtime handles.
- **Context-managed lifecycle**: The requirement that parallel operators run in `with` blocks to guarantee worker shutdown and SHM cleanup.

## Architecture
Both `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` will expose the same signature and flow through a shared constructor pipeline in `src/linear_dag/core/parallel_processing.py`.

The shared constructor pipeline is responsible for:
- Resolving `block_metadata` (or loading with `list_blocks`).
- Loading BED regions from `bed_file` when provided.
- Applying the same filtering metadata transformation for both operators via `_compute_filtered_variant_counts`.
- Deriving common dimensions (`num_samples`, filtered `num_variants`) used by downstream shape rules.
- Building and starting workers through `_ManagerFactory.create_manager(...)`.
- Loading `iids` via `list_iids(hdf5_file)`.

Class-specific hooks are limited to:
- Worker entrypoint (`ParallelOperator._worker` vs `GRMOperator._worker`).
- Shared-memory layout and handle binding.
- Output `shape` calculation.
- Operator-specific state (`alpha` for GRM weighting).

### Public Constructor Contract
Both classmethods use the same parameters and defaults:
- `hdf5_file: str`
- `num_processes: Optional[int] = None`
- `max_num_traits: int = 8`
- `maf_log10_threshold: Optional[int] = None`
- `block_metadata: Optional[pl.DataFrame] = None`
- `bed_file: Optional[str] = None`
- `bed_maf_log10_threshold: Optional[int] = None`
- `alpha: float = -1.0`

`alpha` is operational in GRM paths and a documented no-op in genotype-only paths.

## Existing Patterns
This design follows patterns already present in:
- `src/linear_dag/core/parallel_processing.py` for shared-memory process orchestration, per-worker flag loops, and context-manager cleanup.
- `tests/core/test_parallel_processing.py` for parallel-vs-serial numeric parity validation.
- `tests/cli/test_cli.py` for constructor-kwargs forwarding checks from CLI workflows.
- `src/linear_dag/cli.py` for centralized option validation and construction kwargs (`_validate_num_processes`, helper kwargs builders).

The design intentionally preserves these existing boundaries:
- Workers open HDF5-backed `LinearARG` objects inside worker processes (`LinearARG.read(...)` in worker context), not in the parent process.
- Operator lifecycle remains context-managed (`with ... as op`) to guarantee worker shutdown and shared-memory cleanup.
- Block-driven workload partitioning stays in `_ManagerFactory`.

Divergence from current behavior:
- GRM constructor behavior is expanded to accept and apply the same filtering inputs as parallel genotype construction.
- Constructor duplication between `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` is removed in favor of explicit shared helpers.

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Unify Public Constructor Contracts
**Goal:** Expose identical `from_hdf5` signatures and defaults for `ParallelOperator` and `GRMOperator`.

**Components:**
- `src/linear_dag/core/parallel_processing.py`:
  - Update both classmethod signatures to the unified parameter list.
  - Document shared parameter semantics, including `alpha` no-op behavior in genotype-only paths.
- `tests/core/test_parallel_processing.py`:
  - Add signature-level assertions for both constructors (parameter names and defaults).

**Dependencies:** None.

**Done when:** Both constructors publish the same signature/defaults, and signature contract tests pass.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Implement Shared from_hdf5 Pipeline
**Goal:** Move duplicated constructor setup into shared internal helpers while preserving class-specific hooks.

**Components:**
- `src/linear_dag/core/parallel_processing.py`:
  - Add shared helper(s) for metadata resolution, BED loading, filtering-count rewrite, and manager bootstrap.
  - Add internal hook contract for operator-specific SHM spec, worker callable, and operator instantiation.
  - Keep `_ManagerFactory.create_manager(...)` as the process-partitioning backend.
- `tests/core/test_parallel_processing.py`:
  - Add/adjust constructor behavior tests for shape and basic startup consistency under the unified path.

**Dependencies:** Phase 1.

**Done when:** Both constructors route through one shared setup path, with no duplicated metadata/manager bootstrap logic remaining.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Align Filtering Semantics Across Operators
**Goal:** Ensure GRM construction applies the same MAF/BED filtering semantics as parallel genotype construction.

**Components:**
- `src/linear_dag/core/parallel_processing.py`:
  - Apply filtering metadata rewrite for GRM path using the same helper path used by ParallelOperator.
  - Ensure worker inputs and shape math remain consistent after filtering.
- `tests/core/test_parallel_processing.py`:
  - Add GRM filtering parity tests (MAF-only and BED-aware cases) against serial filtered baselines.
- `tests/core/test_bed_filtering.py`:
  - Extend or add cases for GRM filter behavior where needed.

**Dependencies:** Phase 2.

**Done when:** GRM filtered outputs and dimensions match serial filtered expectations, and unfiltered parity tests remain green.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Integrate Call Sites and Guardrails
**Goal:** Keep CLI and library call sites aligned with unified constructor contracts.

**Components:**
- `src/linear_dag/cli.py`:
  - Update GRM constructor kwargs forwarding (and helper structure, if needed) to match unified contract.
  - Preserve existing user-visible CLI flags unless explicit expansion is approved.
- `tests/cli/test_cli.py`:
  - Extend RHE constructor forwarding tests to cover unified kwargs handling.
- `tests/association/test_rhe.py`:
  - Confirm behavior remains compatible with existing RHE workflows.

**Dependencies:** Phase 3.

**Done when:** Existing CLI workflows remain backward compatible and forwarding tests pass for both operator paths.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Reliability Validation and Review Gate
**Goal:** Confirm lifecycle and error-handling behavior remains safe after unification.

**Components:**
- `tests/core/test_parallel_processing.py`:
  - Add invalid `num_processes` constructor-path checks.
  - Add cleanup/error-path checks where practical.
- Engineering process:
  - Run `requesting-code-review` review/fix/re-review loop after implementation tasks until zero findings.

**Dependencies:** Phases 1-4.

**Done when:** Reliability tests pass and code-review loop reports zero unresolved findings.
<!-- END_PHASE_5 -->

## Additional Considerations
Parallel-processing guardrails:
- Continue opening HDF5 state inside workers to avoid sharing HDF5 handles across fork boundaries.
- Maintain explicit `close()` and `unlink()` ownership for shared-memory objects and avoid widening lock scopes beyond aggregation writes.
- Keep worker-count validation explicit (`num_processes >= 1`) to fail early for invalid configurations.

Behavioral guardrails:
- Unifying signatures does not imply all parameters affect both classes equally; parameter semantics must be explicit in docstrings.
- Introducing GRM filtering changes effective variant inclusion; tests must verify this behavior directly and avoid silent shape assumptions.

External best-practice references:
- Python multiprocessing context guidance (library design): https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- Python shared memory lifecycle (`close()`/`unlink()`): https://docs.python.org/3/library/multiprocessing.shared_memory.html
- h5py multiprocessing note (open files independently per process): https://docs.h5py.org/en/stable/mpi.html
