# Packed Representation Spike Implementation Plan

**Goal:** Convert heterogeneous LinearARG blocks into a fixed set of losslessly packed, explicitly sharded graph arrays with bounded padding and observable memory accounting.

**Architecture:** Add one cohesive `packing.py` functional core for canonical block arrays, whole-block assignment, capacity planning, packing/unpacking, invariants, and diagnostics. Keep HDF5 and generic duck-typed group reads plus JAX device placement in the existing `ingress.py` imperative shell; assemble each global field from committed single-device arrays with `jax.make_array_from_single_device_arrays`.

**Tech Stack:** Python 3.11/JAX 0.10.2 baseline for this spike, NumPy, JAX arrays and `NamedSharding`, Equinox PyTrees, HDF5, pytest. Phase 3 performs the approved Python 3.12/JAX 0.11.0 migration and reruns this phase's tests.

**Scope:** Phase 1 of 7 from the approved design.

**Codebase verified:** 2026-08-13 at `19bba4d`; JAX 0.11.0 assembly contracts cross-checked against local tag `a1521744c6dc074443fe549f19f48d7197abf759`.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### jax-packed-sharded-lineararg.AC1: Packing preserves state with bounded memory
- **jax-packed-sharded-lineararg.AC1.1 Success:** The packed graph exposes the same fixed set of array components for datasets with different source block counts; only shapes, descriptor values, and valid lengths vary.
- **jax-packed-sharded-lineararg.AC1.2 Success:** Unpacking valid rows reconstructs every canonical source graph field and the exact logical sample/variant mappings.
- **jax-packed-sharded-lineararg.AC1.3 Success:** On the production benchmark, aggregate packed graph bytes excluding separately reported descriptor metadata are no more than 1.25 times the unpadded canonical graph bytes.
- **jax-packed-sharded-lineararg.AC1.4 Success:** On a balanced two-device production load, maximum graph residency on either device is no more than 0.65 times the unpadded total graph bytes, and every graph array resides only on its assigned device.
- **jax-packed-sharded-lineararg.AC1.6 Failure:** If whole-block rebalancing cannot satisfy the configured padding limit, construction fails with diagnostics unless the caller supplies an explicit override; the prototype does not claim to subdivide one source graph block.

AC1.5 is completed by Phase 6 after all durable-format constructors and promotion paths exist.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Define canonical packing and whole-block assignment contracts

**Verifies:** jax-packed-sharded-lineararg.AC1.1, jax-packed-sharded-lineararg.AC1.3, jax-packed-sharded-lineararg.AC1.6

**Files:**
- Create: `src/linear_dag/core/jaxlinarg/packing.py`
- Modify: `src/linear_dag/core/jaxlinarg/ingress.py:27-57`
- Create: `tests/jax/test_packing.py`

**Implementation:**
- Classify `packing.py` as `# pattern: Functional Core`.
- Move `LinearARGBlockArrays` from `ingress.py` into `packing.py`; import it back into `ingress.py` so existing internal imports continue to work. Keep the passive configuration, descriptor, plan, host-buffer, and diagnostics records colocated in `packing.py`; do not add `types.py`, `descriptors.py`, `metrics.py`, or another subpackage.
- Normalize every block before byte accounting: `int32` graph/mapping arrays, requested floating `data` dtype, Boolean `flip`, identity `nonunique_indices` when absent, and `-1` allele counts when absent. Count the physical-to-logical variant mapping as graph bytes; exclude only block descriptor scalars and valid-length metadata from the AC1.3 numerator.
- Define a fixed component order independent of source block count. It must cover graph pointers, edge indices/data, variant indices, flip, sample indices, nonunique indices, allele counts, per-block cutoffs, valid lengths, and logical variant mappings needed by Phase 2. Do not store source blocks in a traced Python tuple.
- Implement a deterministic whole-block planner. Score candidate assignments using canonical bytes plus node/edge solve work, calculate per-field per-device lengths and capacities, and permit empty device assignments. Source graph blocks are indivisible in this plan.
- Default `max_padding_ratio` to `1.25`. Rebalance before rejecting. If the bound remains exceeded, raise `ValueError` with canonical bytes, padded bytes, ratio, per-device loads, and guidance for an explicit override or exact-ragged fallback. An override disables only the rejection; diagnostics still report the actual ratio.
- Keep `split_blocks_by_n_entries` in `wrapper.py` unchanged because it is the compatibility oracle for `JaxParallelOperator`, not the packed planner.

**Testing:**
- AC1.1: plan synthetic inputs with one, two, and several source blocks at identical capacities and assert the same component names/count and descriptor schema.
- AC1.3: validate byte-category arithmetic independently of JAX allocation; the production threshold itself is exercised in Task 4.
- AC1.6: use a deliberately skewed synthetic input and the bundled two-block fixture to prove default rejection, complete diagnostics, and explicit override. Record that `test_chr21_50.h5` is expected to require the override rather than weakening the default.
- Test stable tie-breaking, complete/non-overlapping assignment, empty assignments, inconsistent sample counts, inconsistent dtypes, invalid metadata, and `num_devices < 1`.

**Verification:**
Run: `pytest -p no:capture tests/jax/test_packing.py -k 'plan or padding or assignment'`
Expected: all selected tests pass.

**Commit:** `feat(jax): add packed graph planning`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Pack and losslessly unpack canonical graph fields

**Verifies:** jax-packed-sharded-lineararg.AC1.1, jax-packed-sharded-lineararg.AC1.2, jax-packed-sharded-lineararg.AC1.6

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/packing.py`
- Modify: `tests/jax/test_packing.py`

**Implementation:**
- Materialize equal-capacity host buffers for each fixed field with leading device-shard dimension. Use inert values plus explicit valid lengths; do not rely on arbitrary padded index values being ignored accidentally.
- Rebase concatenated node, edge, compressed-row, variant, and block descriptor offsets without allowing source blocks to alias. Preserve each block's `min_index_to_keep` and compressed-row extent as descriptor data.
- Preserve logical variant order through an explicit bijective mapping over valid variant rows. Physical assignment may reorder whole blocks, but unpacking must reconstruct original block and variant order exactly.
- Ensure padded CSC columns are edge-free, padding indices stay in bounds, padded variants are masked, and empty device assignments have valid zero/inert buffers.
- Provide validation and a test-oriented unpacking function in the same module. Validation must reject out-of-range descriptor spans, non-bijective logical mappings, overlapping assignments, inconsistent samples/dtypes, and non-inert padding before any JAX execution.

**Testing:**
- AC1.1: compare the packed PyTree definition for different block counts and verify only array shapes/data change.
- AC1.2: pack then unpack canonical fields from existing HDF5 fixture blocks and synthetic blocks; use `numpy.testing.assert_array_equal` for every field and exact logical sample/variant mappings.
- Cover absent optional arrays, flipped variants, nonunique compressed indices, empty assignments, physical block order differing from logical order, and explicit high-padding override.
- AC1.6: prove the override does not bypass structural validation.

**Verification:**
Run: `pytest -p no:capture tests/jax/test_packing.py`
Expected: all packing tests pass.

**Commit:** `feat(jax): pack canonical graph shards`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Assemble explicitly sharded packed arrays during ingress

**Verifies:** jax-packed-sharded-lineararg.AC1.1, jax-packed-sharded-lineararg.AC1.2, jax-packed-sharded-lineararg.AC1.4

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/ingress.py`
- Modify: `tests/jax/test_ingress.py`
- Modify: `tests/jax/test_packing.py`

**Implementation:**
- Keep `ingress.py` classified as `# pattern: Imperative Shell` and keep the packed carrier private; do not change package export modules in this phase.
- Add private constructors from an iterable of canonical `LinearARGBlockArrays`, HDF5 block metadata/names, and the existing duck-typed group-reader boundary used by unit fixtures. Do not label that generic group seam as Zarr support or add a Zarr dependency; the real `genoio` reader remains a downstream integration gate.
- Stage one source block at a time into its assigned host shard buffers. Track staging ownership deterministically in packing diagnostics and release the block reference before loading the next block.
- For every field, create local arrays with identical shape `(1, capacity, ...)` in `NamedSharding.addressable_devices` order using explicit `jax.device_put(host_local, device)`. Assemble the global `(D, capacity, ...)` array using `jax.make_array_from_single_device_arrays`; never construct the full graph on the default device first.
- Use a dedicated mesh axis named `"graph"`. Do not reuse `wrapper.py` helpers that require `"blocks"`.
- Define the private Phase 1 carrier as `_PackedJaxLinearARG(eqx.Module)`. Store only static global shape/capacity metadata and the fixed packed array fields in it. Ensure every graph array has graph-axis sharding. This temporary class name remains private through Phase 7 unless promotion is separately approved.
- Report canonical graph bytes, padded graph bytes, descriptor bytes, staging bytes, final bytes by device, padding ratio, and component/PyTree leaf count. State in the diagnostic API that staging peak is deterministic ingress accounting, not a JAX allocator high-water mark.

**Testing:**
- AC1.1/AC1.2: construct from canonical arrays and HDF5 and verify component structure plus unpacked content.
- AC1.4: in a fresh forced two-device process, inspect each field's `sharding`, `addressable_shards`, shard indices, device set, and `on_device_size_in_bytes()`/`nbytes`; each device must own only its local graph slice.
- Test that `jax.make_array_from_single_device_arrays` receives committed equal-shaped local arrays in addressable-device order and rejects a malformed local layout.
- Test empty device assignments and use an explicit padding override for the skewed bundled fixture.
- Retain existing exact-ragged ingress tests unchanged.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -p no:capture tests/jax/test_packing.py tests/jax/test_ingress.py`
Expected: all tests pass on two forced CPU devices.

**Commit:** `feat(jax): add sharded packed ingress`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Add structural memory and production packing gates

**Verifies:** jax-packed-sharded-lineararg.AC1.3, jax-packed-sharded-lineararg.AC1.4

**Files:**
- Modify: `tests/jax/bench/test_parallel_benchmarks.py`
- Modify: `tests/jax/test_packing.py`

**Implementation:**
- Extend the existing opt-in parallel benchmark result/table rather than creating a benchmark utility module.
- Reuse `--runbench`, `--linarg-h5-path`, metadata fixtures, and `_graph_bytes_by_device`. Add packed rows/columns for canonical graph bytes, padded graph bytes, descriptor bytes, padding ratio, total/max resident bytes, fixed component/PyTree leaf count, and structurally accounted staging bytes.
- Require an explicit representative `--linarg-h5-path` for the production padding/residency gate. Treat the bundled fixture as smoke coverage only because its two blocks intentionally exceed the default padding threshold.
- Synchronize any constructed JAX arrays before reading residency metrics. Use `addressable_shards` and device-buffer byte methods; do not infer device residency from process RSS.
- Keep product timing and StableHLO collective gates out of this phase; Phase 2 owns them.

**Testing:**
- AC1.3: production benchmark fails if `padded_graph_bytes / canonical_graph_bytes > 1.25`; descriptor bytes are separately visible.
- AC1.4: on two forced CPU devices, production benchmark fails if maximum device graph bytes exceed `0.65 * canonical_graph_bytes` or a field has an unexpected resident device.
- Unit-test table/metric calculation on small deterministic records without requiring the production file.

**Verification:**
Run: `pytest -p no:capture tests/jax/test_packing.py`
Expected: all non-benchmark metric tests pass.

Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -p no:capture tests/jax/bench/test_parallel_benchmarks.py --runbench --linarg-h5-path 1kg_chromosomes_n3202_blocks.h5 --linarg-parallel-processes 2 --linarg-benchmark-k 4 20`
Expected: benchmark reports padding at or below 1.25 and maximum per-device graph residency at or below 0.65 of canonical bytes.

Run: `uv run ruff check src/linear_dag/core/jaxlinarg/packing.py src/linear_dag/core/jaxlinarg/ingress.py tests/jax/test_packing.py tests/jax/test_ingress.py tests/jax/bench/test_parallel_benchmarks.py`
Expected: Ruff exits successfully.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

**Commit:** `test(jax): gate packed graph memory`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
