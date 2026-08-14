# Streaming Ingress and Coexistence Implementation Plan

**Goal:** Construct the packed prototype from the branch's real HDF5 and in-memory sources with bounded ingress memory, while keeping the exact-ragged public implementation and CLI route intact until promotion gates pass.

**Architecture:** Keep `ingress.py` as the sole I/O shell. It opens an HDF5 source once, canonicalizes one block at a time, fills assigned host shard buffers, and assembles globally sharded graph arrays directly from committed device-local arrays. The packed class implements the intended future `JaxLinearARG` constructor/method contract privately; `JaxParallelOperator` remains the public exact-ragged fallback.

**Tech Stack:** h5py/HDF5, NumPy host staging, JAX named sharding, existing LinearARG metadata readers, MkDocs, pytest.

**Scope:** Phase 6 of 7 from the approved design. Real Zarr storage is not present on `jax-focused`; its implementation remains on `genoio` and is a downstream integration gate, not work imported by this phase.

**Codebase verified:** 2026-08-13 at `19bba4d` using `ingress.py`, `lineararg.py`, `wrapper.py`, CLI/export modules, and the verified `genoio` branch comparison.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### jax-packed-sharded-lineararg.AC1: Packing preserves state with bounded memory
- **jax-packed-sharded-lineararg.AC1.2 Success:** Unpacking valid rows reconstructs every canonical source graph field and the exact logical sample/variant mappings.
- **jax-packed-sharded-lineararg.AC1.5 Success:** During HDF5 and in-memory ingress, graph residency on the default device never exceeds that device's final assigned shard plus one source block of staging data; the same requirement is recorded for downstream `genoio` Zarr integration.
- **jax-packed-sharded-lineararg.AC1.6 Failure:** If whole-block rebalancing cannot satisfy the configured padding limit, construction fails with diagnostics unless the caller supplies an explicit override; the prototype does not claim to subdivide one source graph block.

### jax-packed-sharded-lineararg.AC7: Backends, ingress, and public API remain compatible
- **jax-packed-sharded-lineararg.AC7.3 Success:** HDF5 inputs require no schema change and reconstruct the same logical operator as the current reader. The `genoio` branch's Zarr reader must pass the same reconstruction and residency contract before a later branch integration; fake group fixtures do not satisfy this criterion.
- **jax-packed-sharded-lineararg.AC7.4 Success:** Public package signatures, annotations, PyTree inspection, and documentation expose no HiJAX types or primitives.
- **jax-packed-sharded-lineararg.AC7.5 Success:** The private packed candidate implements the target single- and multi-block `JaxLinearARG` constructor and method contract, while the public exact `JaxLinearARG` and exact-ragged `JaxParallelOperator` remain constructible and covered. Renaming and exporting the candidate is deferred to a separately approved promotion plan.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Finalize streaming HDF5 and in-memory constructors

**Verifies:** jax-packed-sharded-lineararg.AC1.2, jax-packed-sharded-lineararg.AC1.5, jax-packed-sharded-lineararg.AC1.6, jax-packed-sharded-lineararg.AC7.3

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/ingress.py`
- Modify: `src/linear_dag/core/jaxlinarg/packing.py`
- Modify: `tests/jax/test_ingress.py`
- Modify: `tests/jax/test_packing.py`

**Implementation:**
- Keep `ingress.py` classified Imperative Shell and `packing.py` Functional Core. Do not create `constructors.py`, `streaming.py`, `serialization.py`, or a packed on-disk format.
- Give the private packed class target constructors with consistent signatures: `from_lineararg_arrays`, `from_lineararg`, `from_linearargs`, `from_hdf5_block`, and `from_hdf5`. Single-block calls accept `mesh=None` and create a concrete one-device `"graph"` mesh; plural/file-wide calls accept a concrete single-host mesh.
- Use `max_padding_ratio=1.25` for the padded/canonical byte ratio. A caller may pass a larger explicit value or `None` to override rejection; diagnostics always report the actual ratio and configured limit. Do not conflate this ratio with 25 percentage points of overhead in names or messages.
- For `from_hdf5`, call `_ensure_hdf5_plugins`, open the file once, determine canonical block order from validated metadata, and feed one group at a time into the Phase 1 planner/host staging buffers. Release every temporary block array before reading the next. Do not call `read_hdf5_blocks` or construct exact JAX operators first.
- Support both current multi-block group files and root-level single-block HDF5 files. A root file is valid only when the required root datasets/attributes are present; reject mixed/ambiguous root-plus-block layouts. Preserve root `iids`, optional `n_individuals`, optional `allele_counts`/`nonunique_indices`, dtype normalization, and exact logical variant order.
- Validate caller-supplied block metadata against HDF5 names, counts, `n_entries`, `n_variants`, `n_samples`, and block order before device placement. Filtering/subsetting of block names must remain ordered and update logical offsets consistently.
- Assemble global arrays only from explicitly committed local arrays in addressable-device order. Keep host staging diagnostics distinct from device residency. Synchronize before final residency measurement and avoid process-RSS claims.
- In-memory plural ingress must consume each `LinearARG`/canonical block once, avoid first constructing a tuple of JAX block objects, and preserve IIDs/metadata shared across blocks.
- Do not implement packed `write`, pickle, Equinox serialization, or HDF5 schema fields. “Round trip” means durable source to packed reconstruction and numerical/schema parity.

**Testing:**
- Compare HDF5/in-memory packed fields and products with `LinearARG.read`, individual current block readers, and exact-ragged products for multi-block and root-level single-block fixtures.
- Cover optional arrays present/absent, diploid/haplotype sample indices, metadata reorder/mismatch, subset ordering, corrupt/mixed layout, dtype, backend, and empty source.
- Instrument block reads and device puts: at most one canonical source block is live outside final host buffers, no full graph is placed on the default device, and each final graph field has only its assigned local shards.
- Assert default skew rejection is diagnostic and explicit override succeeds without bypassing structural validation.
- Keep the existing duck-typed group helper tests labeled as generic-group unit tests; do not call them Zarr integration coverage.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/test_ingress.py tests/jax/test_packing.py`
Expected: single-device HDF5/in-memory reconstruction and failure tests pass.

Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_ingress.py tests/jax/test_packing.py -k 'stream or residency or hdf5 or metadata'`
Expected: two-device streaming/residency tests pass.

**Commit:** `feat(jax): stream hdf5 into packed shards`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Establish the exact-ragged/packed coexistence contract

**Verifies:** jax-packed-sharded-lineararg.AC7.4, jax-packed-sharded-lineararg.AC7.5

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/packing.py`
- Modify: `src/linear_dag/core/jaxlinarg/wrapper.py`
- Modify: `src/linear_dag/core/jaxlinarg/grm.py`
- Modify: `tests/jax/test_wrapper.py`
- Modify: `tests/jax/test_grm_operator.py`
- Modify: `tests/cli/test_cli.py`
- Create: `tests/jax/test_coexistence.py`

**Implementation:**
- Keep the packed implementation private under the temporary name `_PackedJaxLinearARG`. Give it the target public shape/dtype/IID properties, product methods, safe compile helpers, and constructor signatures so promotion later is a controlled rename/export change rather than an API redesign.
- Keep the existing exact single-block `JaxLinearARG` and exact-ragged `JaxParallelOperator` public and behaviorally unchanged in this phase. Do not alias a differently named private class as `JaxLinearARG`; class identity and annotations are part of the later promotion review.
- Make `JaxParallelOperator`'s block type and direct constructor contract explicit as exact blocks. Continue accepting current public single-block `JaxLinearARG` values. Do not silently pack them or accept the private packed multi-block carrier as one exact block.
- Make `JaxGRMOperator` dispatch explicitly between private packed and public exact-ragged implementations while keeping public annotations free of HiJAX/private types.
- Preserve `--jax-backend` as an opt-in RHE flag routed through `JaxParallelOperator.from_hdf5`. Do not rename the flag, reinterpret it as a backend enum selector, switch CLI defaults, or auto-fallback from a failed packed constructor.
- Add one private promotion checklist/test mapping target constructor signatures, methods, metadata, class naming, annotations, and fallback behavior. It must identify the exact changes still required to rename the exact block implementation and promote packed `JaxLinearARG` safely.

**Testing:**
- Run the same forward/reverse/GRM fixtures through private packed and public exact-ragged operators and assert logical parity.
- Inspect public `__all__`, imports, signatures, annotations, PyTree leaves, repr/class names, and documentation inventory; no private packed or HiJAX names may leak.
- Assert `JaxParallelOperator.from_linearargs` and direct construction retain exact block behavior and residency validation.
- Assert CLI `--jax-backend` still constructs exact-ragged during coexistence and reports/uses resolved backend correctly.
- Assert a packed padding failure propagates when the private constructor is chosen in tests; no implicit exact-ragged fallback occurs.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/test_coexistence.py tests/jax/test_wrapper.py tests/jax/test_grm_operator.py tests/cli/test_cli.py`
Expected: coexistence, public-isolation, exact fallback, and CLI tests pass.

**Commit:** `test(jax): define packed coexistence contract`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (task 3) -->
<!-- START_TASK_3 -->
### Task 3: Document the experimental contract and downstream Zarr gate

**Verifies:** jax-packed-sharded-lineararg.AC2.5, jax-packed-sharded-lineararg.AC7.3, jax-packed-sharded-lineararg.AC7.4, jax-packed-sharded-lineararg.AC7.5

**Files:**
- Modify: `README.md`
- Create: `docs/api/jax.md`
- Modify: `docs/api/parallel_ops.md`
- Modify: `docs/cli.md`
- Modify: `mkdocs.yml`

**Implementation:**
- Use project docstring/documentation style and internal references. Add one cohesive JAX API page rather than separate pages per class or backend.
- Document the current public exact-ragged API and its bound-outer-JIT restriction. Describe the packed path as an internal candidate under promotion testing; do not publish an import path for private classes.
- Explain the intended functional contract using pseudocode with an explicit operator argument, bound eager methods, safe compile helpers, opaque/non-learnable graph state, supported dense transforms, and why raw closure capture is outside the memory guarantee.
- Document actual backends only: pure JAX and optional CPU FFI. State that this branch has no Pallas/accelerator backend and that GPU exercises the portable pure-JAX path.
- Document HDF5 reconstruction and the absence of packed serialization. State that real Zarr support belongs to `genoio`; its eventual merge must pass reconstruction, peak residency, transform, and schema-parity gates. Do not present duck-typed group helpers as durable Zarr support.
- Preserve the existing CLI flag name and RHE-only scope. Add `--jax-backend` to CLI docs and state that it remains exact-ragged/experimental during coexistence.
- Document padding ratio/override diagnostics and explicit choice of `JaxParallelOperator` as fallback; do not promise automatic fallback.

**Testing:**
- Build docs strictly and inspect internal references.
- Add lightweight documentation/source assertions for explicit-operator examples, no private imports, no Pallas claim, correct Zarr status, and preserved CLI flag.
- Re-run public inspection tests from Task 2.

**Verification:**
Run: `uv run --extra docs mkdocs build --strict`
Expected: documentation builds with no warnings or unresolved internal references.

Run: `uv run ruff check src/linear_dag/core/jaxlinarg/packing.py src/linear_dag/core/jaxlinarg/ingress.py src/linear_dag/core/jaxlinarg/wrapper.py src/linear_dag/core/jaxlinarg/grm.py tests/jax/test_ingress.py tests/jax/test_packing.py tests/jax/test_coexistence.py tests/jax/test_wrapper.py tests/jax/test_grm_operator.py tests/cli/test_cli.py`
Expected: Ruff exits successfully.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

**Commit:** `docs: describe packed jax coexistence`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
