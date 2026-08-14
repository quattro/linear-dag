# Cross-Platform Promotion Gates Implementation Plan

**Goal:** Produce reproducible correctness, IR, memory, and performance evidence across required platforms and make an explicit promote/coexist/reject decision without changing the public facade or weakening failed or missing gates.

**Architecture:** Add a typed, machine-readable promotion result schema shared by product and RHE benchmarks. A portable runner records two cache states on each machine, and an aggregator compares isolated packed-candidate and retained exact-ragged rows from the same candidate commit. A committed decision report is the hard boundary before any separately approved public facade or CLI routing change.

**Tech Stack:** pytest opt-in benchmarks, JSON, JAX lowering/compilation APIs, XLA dumps, HDF5 fingerprinting, shell orchestration, MkDocs.

**Scope:** Phase 7 of 7 from the approved design. This phase builds and runs the gates. If required platform evidence is missing or any gate fails, the correct completion state is coexistence under AC8.5; public promotion then remains a separate, explicitly approved follow-up change.

**Codebase verified:** 2026-08-13 at `19bba4d`; benchmark, packaging, docs, CLI, and repository automation surfaces inspected. No CI workflow or stored cross-platform result set currently exists.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### jax-packed-sharded-lineararg.AC8: Promotion is evidence-gated across platforms
- **jax-packed-sharded-lineararg.AC8.1 Success:** Correctness, transform, IR, and residency suites pass on arm64 CPU, x86_64 CPU, forced two-device CPU, and an available GPU before promotion.
- **jax-packed-sharded-lineararg.AC8.2 Success:** Benchmarks report cold construction/compilation separately from warm execution and include dense communication, padding, peak/final graph residency, and IR metrics.
- **jax-packed-sharded-lineararg.AC8.3 Success:** At $K\in\{4,20\}$, warm packed-product median runtime is no more than 1.20 times the retained exact-ragged median executed at the same candidate commit, backend, and device count on reference environments.
- **jax-packed-sharded-lineararg.AC8.4 Success:** Benchmark output retains NumPy/Cython comparisons for products and RHE but does not conflate cold and warm ratios.
- **jax-packed-sharded-lineararg.AC8.5 Failure:** If any promotion gate fails, the packed path remains experimental, the exact-ragged path remains available, and the blocker is recorded instead of silently weakening the gate.

The phase also reruns AC1-AC7 as release regressions. AC7.5 covers target-contract preparation and coexistence; this implementation increment never claims or performs public promotion, even when AC8.1-AC8.4 pass.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Define the promotion evidence schema and gate evaluator

**Verifies:** jax-packed-sharded-lineararg.AC8.2, jax-packed-sharded-lineararg.AC8.3, jax-packed-sharded-lineararg.AC8.4, jax-packed-sharded-lineararg.AC8.5

**Files:**
- Create: `tests/jax/bench/_promotion.py`
- Create: `tests/jax/bench/test_promotion_harness.py`
- Modify: `tests/conftest.py`

**Implementation:**
- Classify `_promotion.py` as `# pattern: Functional Core`. It is justified as a separate module because product benchmarks, RHE benchmarks, the cross-machine aggregator, and unit tests share one versioned evidence/gate contract.
- Define frozen records for environment, dataset fingerprint, operator/backend configuration, phase timing, memory/IR metrics, numerical status, and gate outcome. Emit canonical JSON with `schema_version`, candidate commit/dirty status, `behavioral_reference_commit=b68e7da` as source-selection provenance, representation identity (`packed_candidate`, `retained_exact_ragged`, or `numpy_cython`), dataset SHA-256/size/block count/logical shape, OS/architecture, Python and numeric-stack versions, device descriptions, requested/resolved backend, FFI/BLAS build config, XLA flags, and persistent-cache state. Do not label the historical commit as an executed benchmark baseline.
- Timing phases are distinct enum/string values: `construction`, `lowering`, `compilation`, `first_execution`, and `warm_execution`. A backend that cannot expose a separate phase records `null` plus a reason; it must not relabel combined time as compilation.
- Record canonical, descriptor, padded, structurally accounted staging peak, final total, maximum-device graph bytes, logical dense collective bytes, graph constant bytes, graph operand count, StableHLO byte/operation counts, and XLA buffer-assignment bytes where available.
- Evaluate ratios only after exact match on candidate commit, dataset fingerprint, operation, concrete backend, dtype, device count, cache policy, and $K$. The denominator is the retained exact-ragged implementation executed from that same candidate checkout, not historical commit `b68e7da`. Reject ambiguous/missing baselines. The hard product gate is `packed warm / exact-ragged warm <= 1.20` for both operations and $K\in\{4,20\}$.
- Retain NumPy/Cython ratios by matching phase and workload. Never compare construction/cold/first-execution rows with warm rows.
- Represent every gate as pass/fail/missing with evidence identifiers and reasons. Missing required platform data is `missing`, which aggregates to a coexistence decision rather than being skipped.
- Add `--jax-promotion-output PATH` and `--jax-enforce-promotion-gates`. Ordinary runs report results; enforcement fails reference runs on a failed local quantitative gate. Cross-platform completeness is evaluated only by the aggregator.

**Testing:**
- Round-trip canonical JSON and reject unknown schema versions, malformed metrics, dirty/mismatched commits, mismatched datasets/backends/devices/dtypes, duplicate evidence IDs, cold/warm ratio pairing, and missing exact baseline.
- Test threshold boundaries at 1.20, failure aggregation, missing platform aggregation, and the three decisions `promote`, `continue_coexistence`, and `reject`.
- Test deterministic dataset fingerprints and environment normalization without loading the full HDF5 file into memory.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/bench/test_promotion_harness.py`
Expected: schema, matching, ratio, and decision unit tests pass without `--runbench`.

**Commit:** `test(jax): define promotion evidence schema`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add matched packed/exact/Cython promotion benchmarks

**Verifies:** jax-packed-sharded-lineararg.AC8.2, jax-packed-sharded-lineararg.AC8.3, jax-packed-sharded-lineararg.AC8.4

**Files:**
- Create: `tests/jax/bench/test_promotion_benchmarks.py`
- Modify: `tests/jax/bench/test_parallel_benchmarks.py`
- Modify: `tests/jax/bench/test_rhe_benchmarks.py`
- Modify: `scripts/summarize_xla_memory_dump.sh`
- Modify: `tests/jax/bench/test_promotion_harness.py`

**Implementation:**
- Keep opt-in benchmark orchestration in benchmark test modules; use `_promotion.py` only for records, matching, serialization, and gate evaluation.
- Run the private packed candidate, retained exact-ragged `JaxParallelOperator`, and NumPy/Cython comparison in separate runner-owned subprocesses so graph residency, allocator state, compilation caches, and process pools cannot contaminate one another. Each child reconstructs from the same HDF5 fingerprint and deterministic input seed, uses the same candidate checkout, concrete backend, dtype, device count, operation, and cache policy, and writes a schema-validated JSON fragment for aggregation.
- Record packed construction, `jax.jit(function).lower(operator, values)`, `lowered.compile()`, compiled first execution, and calibrated post-warmup median separately. Synchronize every timed JAX result. Record exact-ragged construction, first call (including its per-range compilation), and warm median; leave unsupported separate lowering/compile phases null with a reason.
- Benchmark `matmat` and `rmatmat` at $K=4,20$ for pure JAX and available CPU FFI on one device and forced two-device CPU. Compare packed warm time with the isolated retained exact-ragged warm time at the same candidate commit for AC8.3.
- Derive logical dense communication bytes from the selected collective/output contract and report them separately from observed graph residency. Reuse Phase 1 diagnostics and Phase 2 jaxpr/StableHLO helpers for padding, staging/final residency, graph constants/operands, and IR counts.
- Extend RHE results with separate operator construction and first/warm estimate phases. Preserve fixed-probe estimate parity and phase-matched NumPy/Cython ratios; do not call existing `cold_total` a compilation time.
- Treat `genoio@c271a9a` as a recorded historical closure-capture/IR counterexample, not a performance threshold. Do not mix a result collected under a different JAX/runtime version into matched ratios.
- Add a `--json-output PATH` mode to `summarize_xla_memory_dump.sh` for total buffer-assignment bytes, large allocations, aliases, and custom calls. Keep the existing Markdown output for human inspection.

**Testing:**
- Unit-test timing phase labels, synchronization, null phase reasons, subprocess command/environment parity, child-fragment validation, failed-child propagation, result matching, Markdown rendering, JSON emission, and XLA-summary parsing on small fixtures.
- Opt-in smoke uses bundled data with explicit padding override and does not enforce production thresholds.
- Production gate uses the representative HDF5 file, no padding override, $K=4,20$, and fails locally when padding, residency, IR, numerical, or warm ratio thresholds fail.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/bench/test_promotion_harness.py tests/jax/bench/test_parallel_benchmarks.py tests/jax/bench/test_rhe_benchmarks.py`
Expected: non-opt-in unit tests pass and benchmark cases skip without `--runbench`.

Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/bench/test_promotion_benchmarks.py --runbench --jax-enforce-promotion-gates --jax-promotion-output /tmp/linear-dag-jax-promotion-local.json --linarg-h5-path 1kg_chromosomes_n3202_blocks.h5 --linarg-parallel-processes 2 --linarg-benchmark-k 4 20 --rhe-benchmark-num-matvecs 4 20`
Expected: a valid local JSON artifact is written; local gates pass or fail with named evidence rather than an unstructured benchmark error.

**Commit:** `perf(jax): add packed promotion benchmarks`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-5) -->
<!-- START_TASK_3 -->
### Task 3: Add a portable cross-machine evidence runner

**Verifies:** jax-packed-sharded-lineararg.AC8.1, jax-packed-sharded-lineararg.AC8.2, jax-packed-sharded-lineararg.AC8.5

**Files:**
- Create: `scripts/run_jax_promotion.sh`
- Create: `.plans/implementation-plans/2026-08-13-jax-packed-sharded-lineararg/evidence/README.md`
- Modify: `tests/jax/bench/test_promotion_harness.py`

**Implementation:**
- Classify the runner as an Imperative Shell in its header comments. It is justified because the same clean-build, fresh-cache, reused-cache, test, and benchmark protocol must run on several independent machines.
- Require explicit arguments for repository root, representative HDF5 path, output directory, platform label, and device count. Resolve/validate exact paths, require a clean candidate commit for promotable evidence, and never delete a caller-provided directory.
- Build with `LINEAR_DAG_REQUIRE_FFI_CPU=1` for CPU FFI evidence. Run correctness/transform/IR/residency suites with `pytest -p no:capture`, float32, and `JAX_ENABLE_X64=1` float64. On GPU, run the packed pure-JAX path and record that no accelerator-specific backend exists.
- Run the promotion benchmark once with a fresh persistent-cache directory and again from a new process reusing that exact directory. Use a runner-owned temporary directory under the explicit output directory and clean only that resolved directory on success.
- Write one evidence JSON per cache state plus command/environment logs and checksums. Do not include hostnames, usernames, credentials, or absolute dataset paths in committed artifacts; use caller-provided platform labels and fingerprints.
- Required evidence labels are arm64 CPU, x86_64 CPU, forced two-device CPU, and available GPU. A single run may satisfy multiple labels only when its recorded architecture/devices prove them.
- Do not add speculative GitHub Actions runner labels or GPU workflows to a repository with no CI convention. The portable runner is the cross-platform contract; CI integration can later invoke it after runner infrastructure is explicitly approved.

**Testing:**
- In subprocess/unit tests, assert every generated pytest command includes `-p no:capture`, cache states differ as intended, unsafe/broad output paths are rejected, dirty commits are non-promotable, and logs redact absolute source paths.
- Smoke the runner with bundled data and enforcement disabled; unit-test the command builder for CPU/FFI and GPU/pure-JAX modes.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/bench/test_promotion_harness.py -k 'runner or platform or cache or command'`
Expected: runner safety, command, and environment tests pass.

**Commit:** `test(jax): add cross-machine promotion runner`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Collect evidence and record the promotion decision

**Verifies:** jax-packed-sharded-lineararg.AC8.1, jax-packed-sharded-lineararg.AC8.3, jax-packed-sharded-lineararg.AC8.4, jax-packed-sharded-lineararg.AC8.5

**Files:**
- Create: `.plans/implementation-plans/2026-08-13-jax-packed-sharded-lineararg/promotion-decision.md`
- Create as evidence is obtained: `.plans/implementation-plans/2026-08-13-jax-packed-sharded-lineararg/evidence/*.json`

**Implementation:**
- Run Task 3 on arm64 CPU, x86_64 CPU, forced two-device CPU, and an available GPU using the same clean candidate commit and dataset fingerprint. Commit normalized evidence JSON; keep large XLA dumps outside Git and record their checksum/location.
- Aggregate all evidence and enumerate AC1-AC8 gate status, matched performance rows, missing rows, failure reasons, JAX cache behavior, and retained fallbacks in `promotion-decision.md`.
- The decision must be exactly `promote`, `continue_coexistence`, or `reject`. Missing x86_64 or GPU evidence is a blocker, not “not applicable.” Shared-runner timing noise may be reported but cannot silently change the 1.20 reference threshold.
- If any gate is failed or missing, choose `continue_coexistence` or `reject`: keep the packed class private, retain public `JaxLinearARG`/`JaxParallelOperator`, keep `--jax-backend` opt-in and exact-ragged, and name every blocker.
- If all gates pass, record `promote` but stop before changing the public facade. The class rename, `JaxParallelOperator.from_linearargs` compatibility migration, export routing, and CLI route require a separate maintainer-approved implementation plan because they are public API changes not safely inferable before evidence exists.
- Do not stage the machine-local HDF5 symlink or benchmark dataset.

**Testing:**
- Validate every committed evidence file against the schema and recompute the decision from JSON in a unit test/verification command.
- Cross-check Markdown decision rows against JSON identifiers and ratios; the written conclusion must equal the evaluator's result.
- For coexistence, run public inspection/CLI tests proving no promotion leaked. For a future promote decision, require a new approved plan rather than editing exports in this task.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/bench/test_promotion_harness.py tests/jax/test_coexistence.py tests/cli/test_cli.py`
Expected: evidence validation, computed decision, and retained public behavior pass.

**Commit:** `docs(jax): record packed promotion decision`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Complete release documentation and project-context audit

**Verifies:** jax-packed-sharded-lineararg.AC7.4, jax-packed-sharded-lineararg.AC7.5, jax-packed-sharded-lineararg.AC7.6, jax-packed-sharded-lineararg.AC8.5

**Files:**
- Modify: `README.md`
- Modify: `docs/api/jax.md`
- Modify: `docs/api/parallel_ops.md`
- Modify: `docs/cli.md`
- Modify: `docs/install.md`
- Modify: `docs/contributing.md`
- Modify: `mkdocs.yml`
- Modify: `AGENTS.md`
- Modify if audit finds drift: `pyproject.toml`
- Modify if audit finds drift: `.gitignore`
- Modify if audit finds drift: `uv.lock`

**Implementation:**
- Update docs to the recorded decision, including explicit-operator JIT usage, safe helpers, opaque/non-learnable graph state, supported dense transforms, memory/padding limits, exact-ragged fallback, current backends, HDF5 reconstruction, downstream `genoio` Zarr gate, and benchmark reproduction commands.
- If the decision is coexistence, label packed as experimental/private and publish no private import. If later promotion is separately approved, documentation changes belong with that facade migration.
- Correct existing contributor commands to use `pytest -p no:capture` and correct stale plan/site paths.
- Audit package metadata and the tracked lockfile for Python `>=3.12,<3.15`, exact JAX/JAXlib 0.11.0, NumPy 2.1+, SciPy 1.15+, classifiers, Hatch matrix, and Ruff target.
- Update `AGENTS.md` contracts to remove the stale Pallas availability guarantee, describe actual pure-JAX/CPU-FFI selection, and avoid claiming real Zarr support on this branch. Use the project-context maintenance workflow during implementation.
- Keep the public top-level names unchanged for a coexistence decision; no HiJAX or private packed types appear in docs/API signatures.

**Testing:**
- Strict docs build, package build, metadata inspection under Python 3.12, public API inspection, and grep/source checks for stale Pallas/Zarr/pytest claims.
- Run all JAX, RHE, CLI, formatting, typing, and build checks after context changes.

**Verification:**
Run: `uv run --extra docs mkdocs build --strict`
Expected: docs build with no warnings.

Run: `LINEAR_DAG_REQUIRE_FFI_CPU=1 uv build`
Expected: sdist/wheel build succeeds with required CPU FFI.

Run: `uv run pytest -p no:capture tests/jax tests/association/test_heritability_jax.py tests/association/test_rhe.py tests/core/test_alignment.py tests/cli/test_cli.py`
Expected: all non-opt-in release regressions pass.

Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_packing.py tests/jax/test_packed_products.py tests/jax/test_hijax.py tests/jax/test_transform_composition.py tests/jax/test_grm_operator.py tests/association/test_heritability_jax.py`
Expected: forced two-device correctness, transform, IR, and residency suites pass.

Run: `JAX_ENABLE_X64=1 uv run pytest -p no:capture tests/jax`
Expected: x64-enabled JAX suite passes.

Run: `uv run ruff check src tests scripts`
Expected: Ruff exits successfully for supported file types.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

Run: `git diff --check`
Expected: no whitespace errors.

**Commit:** `docs(jax): finalize packed validation status`
<!-- END_TASK_5 -->
<!-- END_SUBCOMPONENT_B -->
