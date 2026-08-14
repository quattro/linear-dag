# Explicit Pure-JAX Sharded Products Implementation Plan

**Goal:** Compute packed forward and transpose products under outer JIT with the graph supplied as explicit sharded operands rather than closed-over constants.

**Architecture:** Add one `packed_products.py` mixed boundary for local LinearARG algebra, `shard_map` execution, collectives, and logical output reconstruction. Extend the private carrier in `packing.py` only with delegating convenience methods; preserve `operator.py` and `wrapper.py` as exact-ragged numerical and memory oracles.

**Tech Stack:** JAX pure kernels, `jax.shard_map`, `NamedSharding`, StableHLO inspection, Equinox/PyTrees, pytest.

**Scope:** Phase 2 of 7 from the approved design.

**Codebase verified:** 2026-08-13 at `19bba4d`; `shard_map`/collective lowering cross-checked against local JAX 0.11.0 tag `a1521744c6dc074443fe549f19f48d7197abf759`.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### jax-packed-sharded-lineararg.AC2: Compiled programs treat graph state as explicit data
- **jax-packed-sharded-lineararg.AC2.1 Success:** `jax.make_jaxpr` for the supported functional call contains no graph-sized constants; packed graph arrays are dynamic input variables.
- **jax-packed-sharded-lineararg.AC2.2 Success:** At fixed packed capacities, the number of graph operands and StableHLO operations does not grow with source block count.
- **jax-packed-sharded-lineararg.AC2.3 Success:** Lowered graph operands retain their declared graph-axis shardings, and StableHLO contains no graph all-gather or graph broadcast.
- **jax-packed-sharded-lineararg.AC2.4 Success:** The safe compilation helper satisfies AC2.1 through AC2.3 while keeping the operator at the Python convenience layer.
- **jax-packed-sharded-lineararg.AC2.5 Contract:** Documentation marks raw bound-method closure capture as outside the memory guarantee, does not rely on brittle tracer inspection to reject it, and directs callers to the explicit-operator API or safe compilation helpers.

### jax-packed-sharded-lineararg.AC3: Packed products preserve LinearARG numerics
- **jax-packed-sharded-lineararg.AC3.1 Success:** Packed `matmat` matches Cython `LinearARG` and exact-ragged `jax-focused` results within dtype-appropriate tolerances across the existing oracle corpus.
- **jax-packed-sharded-lineararg.AC3.2 Success:** Packed `rmatmat` matches both references and returns exact logical variant order and shape.
- **jax-packed-sharded-lineararg.AC3.3 Success:** Rank-one and rank-two operands, flipped variants, compressed nonunique indices, float32, and float64 when JAX x64 is enabled behave consistently with the existing operator contract.
- **jax-packed-sharded-lineararg.AC3.4 Failure:** Invalid operand shapes/dtypes, descriptor ranges, graph indices, or non-bijective logical mappings fail before numerical execution.

### jax-packed-sharded-lineararg.AC4: Multi-device execution preserves graph ownership
- **jax-packed-sharded-lineararg.AC4.1 Success:** Single-device and forced two-device CPU products match the same logical reference outputs.
- **jax-packed-sharded-lineararg.AC4.2 Success:** Each local `shard_map` body receives only its assigned graph shard and the dense data needed for that operation.
- **jax-packed-sharded-lineararg.AC4.3 Success:** Forward StableHLO contains the selected sample-space reduction (`psum_scatter` when compatible with the requested output sharding, otherwise replicated `psum`); reverse StableHLO communicates only dense sample/result data and never graph arrays.
- **jax-packed-sharded-lineararg.AC4.4 Edge:** Uneven source-block counts and empty assignments on overprovisioned meshes remain numerically correct with valid descriptors and shardings; a skewed fixture may run only with an explicit padding override.
- **jax-packed-sharded-lineararg.AC4.5 Failure:** Unsupported mesh axes, incompatible shardings, or unavailable required collectives fail with an actionable construction or lowering error.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Implement local packed forward and transpose algebra

**Verifies:** jax-packed-sharded-lineararg.AC3.1, jax-packed-sharded-lineararg.AC3.2, jax-packed-sharded-lineararg.AC3.3, jax-packed-sharded-lineararg.AC3.4

**Files:**
- Create: `src/linear_dag/core/jaxlinarg/packed_products.py`
- Create: `tests/jax/test_packed_products.py`

**Implementation:**
- Classify `packed_products.py` as `# pattern: Mixed (unavoidable)` because it combines pure LinearARG algebra with explicit mesh collectives/device-local execution.
- Define module-level project-owned functions `lineararg_matmat(operator: _PackedJaxLinearARG, values)` and `lineararg_rmatmat(operator: _PackedJaxLinearARG, values)`, plus private rank-two local bodies. Import the private carrier only for internal typing and do not export these functions publicly yet.
- Reuse `_as_rank2_matrix` shape/rank conventions from `operator.py`; preserve current numeric dtype conversion rather than introducing a new strict rejection of cast-compatible values. AC3.4 dtype failures cover unsupported/non-numeric or policy-invalid dtypes, not ordinary castable operands.
- In each local body, remove the singleton local graph-shard axis, iterate a static descriptor capacity rather than the logical source block count, mask invalid descriptor slots, and invoke `pure_jax_solve_forward_compressed` / `pure_jax_solve_backward_compressed` directly. Do not call the existing `_solve_*` custom-VJP wrappers because a packed per-block cutoff is dynamic descriptor data.
- Preserve exact algebra from `operator.py`: scatter-add repeated compressed variant rows, apply flip signs and forward sample-wide correction, seed reverse sample rows, and replace reverse flipped values by `sum(samples) - value`.
- Make padding inert before every gather/scatter; padded indices remain in bounds but cannot contribute. Scatter local reverse results into a zero logical `(n_variants, K)` buffer through the bijective mapping.
- Validate packed invariants and public operand shape outside the numerical body. Do not encode source block count in Python branches or `lax.switch` arms.

**Testing:**
- AC3.1/AC3.2: compare local and single-device global results against `tests/jax/oracle.py`, Cython `LinearARG`, and exact-ragged block accumulation.
- AC3.3: cover rank one/two, deterministic flips, repeated/nonunique mappings, float32, and float64 under `jax_enable_x64`.
- AC3.4: cover wrong ranks/leading dimensions, unsupported dtype, out-of-range descriptors, invalid graph indices, and non-bijective logical mappings before solve execution.
- Include an empty/inert descriptor slot and physical block order different from logical order.

**Verification:**
Run: `pytest -p no:capture tests/jax/test_packed_products.py -k 'local or single_device or invalid'`
Expected: all selected tests pass.

**Commit:** `feat(jax): add packed local pure-jax products`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add explicit `shard_map` products and safe compilation helpers

**Verifies:** jax-packed-sharded-lineararg.AC2.4, jax-packed-sharded-lineararg.AC2.5, jax-packed-sharded-lineararg.AC4.1, jax-packed-sharded-lineararg.AC4.2, jax-packed-sharded-lineararg.AC4.3, jax-packed-sharded-lineararg.AC4.4, jax-packed-sharded-lineararg.AC4.5

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/packed_products.py`
- Modify: `src/linear_dag/core/jaxlinarg/packing.py`
- Modify: `tests/jax/test_packed_products.py`

**Implementation:**
- Wrap local bodies with `jax.shard_map` using the carrier as an explicit argument, a PyTree-prefix graph-axis `in_specs`, explicit dense `in_specs`, `axis_names={"graph"}`, and `check_vma=True`. Validate the carrier mesh is concrete, single-host, and owns a `"graph"` axis before tracing.
- Forward: compute per-device sample partials and use replicated `lax.psum` with `out_specs=P()` as the correctness baseline. Add a sample-sharded `psum_scatter` variant only when output leading dimension and requested sharding are compatible; never pad or silently reshape the public sample result merely to use reduce-scatter.
- Reverse: scatter local valid variants into the exact global logical dense shape and reduce dense result data with `lax.psum`. This communicates result values but never graph fields. Avoid an all-gather of graph operands.
- Add carrier-bound `matmat`/`rmatmat` conveniences that use method-local imports to delegate to module-level functions. `packed_products.py` may import `_PackedJaxLinearARG` under `TYPE_CHECKING`, but neither module performs a runtime top-level import of the other.
- Add `compile_matmat()` and `compile_rmatmat()`. The returned wrapper must call a module-level jitted function with `operator` supplied as an executable argument on every invocation; do not implement it as `jax.jit(lambda values: self.matmat(values))`.
- Document that raw bound-method closure capture is outside the memory guarantee and do not inspect private tracer classes to attempt rejection.

**Testing:**
- AC4.1/AC4.2: compare one- and forced-two-device results and inspect local shard indices/devices for every graph field.
- AC4.3: assert the intended output/result collective while proving no graph operand is gathered; exact IR evidence is consolidated in Task 3.
- AC4.4: cover uneven block counts, explicit high-padding fixture override, and empty device assignment.
- AC4.5: reject missing/wrong mesh axis, incompatible output sharding, non-single-host mesh, and invalid reduce-scatter shape with actionable messages.
- AC2.4: lower/call each safe helper and compare with the explicit functional operation.
- AC2.5: inspect docstrings and prove the helper passes the carrier as an argument rather than closure constants.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -p no:capture tests/jax/test_packed_products.py -k 'shard or compile or mesh or empty'`
Expected: all selected tests pass on two CPU devices.

**Commit:** `feat(jax): add explicit packed shard-map products`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (task 3) -->
<!-- START_TASK_3 -->
### Task 3: Enforce graph operand, sharding, and lowered-IR contracts

**Verifies:** jax-packed-sharded-lineararg.AC2.1, jax-packed-sharded-lineararg.AC2.2, jax-packed-sharded-lineararg.AC2.3, jax-packed-sharded-lineararg.AC2.4, jax-packed-sharded-lineararg.AC4.2, jax-packed-sharded-lineararg.AC4.3

**Files:**
- Modify: `tests/jax/test_packed_products.py`
- Modify: `tests/jax/bench/test_parallel_benchmarks.py`

**Implementation:**
- Keep IR inspection helpers local to `test_packed_products.py` until another test module needs them; do not add an IR utility module.
- Inspect `ClosedJaxpr.consts` and `jaxpr.constvars` recursively and report array constant bytes. Assert zero graph-sized constants for explicit calls and safe helpers.
- Compare two carriers with identical capacities and different source-block counts. Assert a fixed packed PyTree structure, graph operand count after DCE, recursive jaxpr equation structure, and StableHLO operation counts. Permit genuinely unused fields such as allele counts to disappear through DCE.
- Lower with `jax.jit(function).lower(operator, values).compiler_ir("stablehlo")`. Assert graph operands carry `sdy.sharding`/manual-computation structure and no source-block-dependent branch/switch expansion.
- Assert the selected forward collective (`stablehlo.all_reduce` for replicated `psum`, reduce-scatter form for a compatible sample-sharded output) and absence of graph all-gather. Do not globally ban `stablehlo.broadcast_in_dim`; ordinary masks/scalars use it.
- Inspect final `addressable_shards` after products to prove graph leaves remain on their original assigned devices.
- Extend the opt-in benchmark table with graph constant bytes, graph operand count, and StableHLO operation count; reuse Phase 1 packing metrics.

**Testing:**
- AC2.1: explicit functional and safe-helper jaxprs have no graph constants; add a deliberately closed-over diagnostic control that demonstrates why it is unsupported without making that pattern part of the API.
- AC2.2: block-count variation at fixed capacities leaves operand and IR structure unchanged.
- AC2.3/AC4.2: graph arguments retain graph-axis sharding and graph residency.
- AC4.3: collectives are restricted to dense sample/result values.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -p no:capture tests/jax/test_packed_products.py`
Expected: all product, sharding, and IR tests pass.

Run: `uv run ruff check src/linear_dag/core/jaxlinarg/packing.py src/linear_dag/core/jaxlinarg/packed_products.py tests/jax/test_packed_products.py tests/jax/bench/test_parallel_benchmarks.py`
Expected: Ruff exits successfully.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

**Commit:** `test(jax): enforce packed graph IR contracts`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
