# Packed CPU-FFI Backend Implementation Plan

**Goal:** Run packed local products through the existing native CPU acceleration boundary without reintroducing graph constants, graph replication, or backend-specific autodiff behavior.

**Architecture:** Extend the native FFI with descriptor-aware packed forward/backward solve targets. Phase 2 chooses a pure-JAX or FFI local solve inside the same `shard_map` body; Phase 3's paired product primitives continue to own JVP, VJP, transpose, and batching. Explicit unavailable FFI requests fail before lowering, while `AUTO` retains documented pure-JAX fallback.

**Tech Stack:** JAX 0.11 FFI API, XLA FFI C++ API, existing BLAS/Accelerate hooks, HiJAX transform boundary, pytest.

**Scope:** Phase 5 of 7 from the approved design. This branch has no Pallas or other accelerator backend; adding one is out of scope and must later satisfy the same contract.

**Codebase verified:** 2026-08-13 at `19bba4d`; raw FFI and HiJAX composition behavior verified in local JAX 0.11.0 source at `a1521744c6dc074443fe549f19f48d7197abf759`.

---

## Acceptance Criteria Coverage

This phase implements and tests backend-specific portions of:

### jax-packed-sharded-lineararg.AC3: Packed products preserve LinearARG numerics
- **jax-packed-sharded-lineararg.AC3.1 Success:** Packed `matmat` matches Cython `LinearARG` and exact-ragged `jax-focused` results within dtype-appropriate tolerances across the existing oracle corpus.
- **jax-packed-sharded-lineararg.AC3.2 Success:** Packed `rmatmat` matches both references and returns exact logical variant order and shape.
- **jax-packed-sharded-lineararg.AC3.3 Success:** Rank-one and rank-two operands, flipped variants, compressed nonunique indices, float32, and float64 when JAX x64 is enabled behave consistently with the existing operator contract.
- **jax-packed-sharded-lineararg.AC3.4 Failure:** Invalid operand shapes/dtypes, descriptor ranges, graph indices, or non-bijective logical mappings fail before numerical execution.

### jax-packed-sharded-lineararg.AC5: Dense operands support composable JAX transformations
- **jax-packed-sharded-lineararg.AC5.1 Success:** Forward JVPs for `matmat` and `rmatmat` equal the corresponding product applied to the dense tangent.
- **jax-packed-sharded-lineararg.AC5.2 Success:** VJPs equal the companion adjoint product and match analytical and finite-difference references.
- **jax-packed-sharded-lineararg.AC5.3 Success:** `jit`, `jit(grad)`, `grad(jit)`, `value_and_grad`, nested/higher-order derivatives, `vmap`, `scan`, and `remat` pass their documented transform-composition cases with the operator explicit.
- **jax-packed-sharded-lineararg.AC5.4 Success:** Symbolic-zero and dead-code paths do not retain graph cotangents or duplicate graph residuals.

### jax-packed-sharded-lineararg.AC7: Backends, ingress, and public API remain compatible
- **jax-packed-sharded-lineararg.AC7.1 Success:** Pure-JAX, CPU-FFI, and each actually available future accelerator backend pass the same numerical, transform, sharding, and graph-residency contracts. This branch currently exposes no accelerator backend.
- **jax-packed-sharded-lineararg.AC7.2 Failure:** An explicitly requested unavailable or transform-incompatible backend fails before lowering; `Backend.AUTO` uses only its documented fallback.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Specify and test a descriptor-aware packed FFI ABI

**Verifies:** jax-packed-sharded-lineararg.AC3.4, jax-packed-sharded-lineararg.AC7.1

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/kernels/ffi_cpu.py`
- Modify: `src/linear_dag/core/jaxlinarg/kernels/ffi_cpu_impl.cc`
- Modify: `tests/jax/test_kernels_ffi_cpu.py`

**Implementation:**
- Keep `ffi_cpu.py` classified Imperative Shell. Retain the current four exact single-block targets unchanged for `JaxLinearARG`/`JaxParallelOperator` compatibility.
- Add distinct packed forward/backward targets for float32 and float64. Do not overload the current ABI or infer packed mode from buffer shapes.
- Define one versioned descriptor-column contract shared with `packing.py`: valid flag/count, node/indptr/edge/compressed-row starts and lengths, and per-block `min_index_to_keep`. Document units and whether stored offsets are block-local or rebased. Pass descriptors as an integer buffer, not hashable FFI attributes, so different block layouts do not trigger new target definitions.
- The packed handler receives one device-local flattened graph shard, fixed descriptor capacity, and a rank-two aggregate compressed-state buffer. It loops valid descriptors and then graph nodes natively; padded descriptors and fields are inert. No Python/source-block loop or HLO branch is created per logical block.
- Preserve output/input aliasing for the mutable work buffer when XLA can reuse it. Treat graph arrays and descriptors as read-only; never alias them or allocate a second copy.
- Add dimension, dtype, monotonic-offset, valid-length, and range checks in the C++ boundary before pointer arithmetic. Return `ffi::Error` for malformed descriptors instead of risking out-of-bounds access. Python construction remains the primary validator, but the native ABI must be memory safe when called directly.
- Reuse existing `Axpy`, BLAS selection, registration, and build metadata. Register the new targets through the existing cached process-wide loader.
- Add Python wrappers `ffi_cpu_packed_solve_forward`/`ffi_cpu_packed_solve_backward` that select dtype targets, supply explicit result metadata, use the supported 0.11 `ffi_call` options, and fail with an actionable error if registration is unavailable.

**Testing:**
- Unit-test target selection, argument order, result shape/dtype, alias index, registration names, and that no call occurs when unavailable.
- Native-test one, several, empty, and padded descriptor slots against repeated pure-JAX compressed solves for forward/backward, float32, and float64 with x64 enabled.
- Directly pass malformed descriptors/dimensions and assert Python validation or an FFI error, never a crash or silent result.
- Retain every existing exact single-block FFI test and build metadata test.

**Verification:**
Run: `LINEAR_DAG_REQUIRE_FFI_CPU=1 uv build`
Expected: the native extension builds with all exact and packed targets registered.

Run: `JAX_ENABLE_X64=1 uv run pytest -p no:capture tests/jax/test_kernels_ffi_cpu.py`
Expected: exact and packed FFI ABI/numerical tests pass in float32 and float64.

**Commit:** `feat(jax): add packed cpu ffi solve abi`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Dispatch packed local products through CPU FFI

**Verifies:** jax-packed-sharded-lineararg.AC3.1, jax-packed-sharded-lineararg.AC3.2, jax-packed-sharded-lineararg.AC3.3, jax-packed-sharded-lineararg.AC5.1, jax-packed-sharded-lineararg.AC5.2, jax-packed-sharded-lineararg.AC5.3, jax-packed-sharded-lineararg.AC5.4, jax-packed-sharded-lineararg.AC7.1

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/packed_products.py`
- Modify: `src/linear_dag/core/jaxlinarg/_hijax.py`
- Modify: `tests/jax/test_packed_products.py`
- Modify: `tests/jax/test_hijax.py`
- Modify: `tests/jax/test_operator_ffi_cpu.py`

**Implementation:**
- Add a backend-specific local solve selector in `packed_products.py`. Pure JAX calls the Phase 2 descriptor-capacity body; CPU FFI calls the new descriptor-aware native wrapper once per local graph shard. Keep seeding, flip correction, logical variant scatter/reconstruction, and dense collectives backend-neutral.
- Invoke FFI inside the `shard_map` local body so every call sees only the assigned local graph shard. Do not move graph fields to an assembly/default device.
- Carry the resolved backend as compact primitive/type metadata. HiJAX JVP/VJP/transpose/batch rules bind the same or companion product primitive with that backend; they must never transform or directly differentiate `ffi_call`.
- Keep the current exact single-block FFI route operational. Where packed and exact paths share backend resolution, refactor one selector rather than creating diverging enums or availability caches.
- Do not add Pallas enum members, placeholder accelerator targets, or device callbacks.

**Testing:**
- Parameterize packed numerical, GRM, and transform tests over `PURE_JAX` and available `FFI_CPU`; compare both with exact-ragged/Cython.
- Cover outer `jit`, JVP, VJP, `jit(grad)`, higher-order nonlinear loss, `vmap`, `scan`, and `remat` on CPU FFI. Assert raw `ffi_call` appears only after high-level expansion.
- Forced two-device tests inspect graph residency before/after FFI products and StableHLO for local custom calls plus dense-only collectives; no graph all-gather/broadcast is allowed.
- Compare empty assignments, uneven blocks, flips, nonunique rows, and both supported dtypes.
- Re-run exact operator FFI tests to ensure the new ABI did not change old single-block results.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 uv run pytest -p no:capture tests/jax/test_packed_products.py tests/jax/test_hijax.py tests/jax/test_operator_ffi_cpu.py -k 'ffi or backend'`
Expected: packed and exact CPU-FFI numerical, transform, IR, and residency tests pass on forced two-device CPU.

**Commit:** `feat(jax): dispatch packed products to cpu ffi`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (task 3) -->
<!-- START_TASK_3 -->
### Task 3: Make backend selection explicit and observable

**Verifies:** jax-packed-sharded-lineararg.AC7.1, jax-packed-sharded-lineararg.AC7.2

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/operator.py`
- Modify: `src/linear_dag/core/jaxlinarg/build_config.py`
- Modify: `tests/jax/test_backend_resolution.py`
- Modify: `tests/jax/test_ffi_fallback.py`
- Modify: `tests/jax/test_operator_ffi_cpu.py`
- Modify: `tests/jax/bench/test_parallel_benchmarks.py`

**Implementation:**
- Separate exact and packed CPU-FFI capability checks. Preserve the existing exact-target availability result for `JaxLinearARG`/`JaxParallelOperator`, and add a packed-target result used only by `_PackedJaxLinearARG`; expose both results and their independent registration errors in build configuration.
- Change representation-aware `resolve_backend(Backend.FFI_CPU, require_packed_targets=...)` to fail before construction/lowering when the platform is not CPU or the targets required by that representation are unavailable. Include the matching exact/packed registration reason and recommend `Backend.AUTO` or `Backend.PURE_JAX`.
- Preserve `Backend.AUTO` per representation: exact operators use CPU FFI whenever the existing exact targets are complete, even if a backward-compatible extension lacks new packed targets; packed operators require the complete packed target set. Otherwise select pure JAX without warning. Accelerator platforms use pure JAX because this branch exposes no accelerator backend.
- Make each target set internally atomic: partial packed registration makes only packed FFI unavailable, while partial exact registration makes exact FFI unavailable. Do not collapse both states into one global Boolean.
- Update `Backend` docstrings and build-config output to describe actual behavior. Remove stale text and tests asserting an explicit FFI warning/fallback.
- Record requested and resolved backend plus FFI/BLAS/native-tuning configuration in benchmark rows. A row labeled FFI must have actually used FFI.

**Testing:**
- Cover AUTO and explicit resolution across CPU/GPU/TPU labels, exact-only/full/partial/unavailable registration, source-only installs, and non-CPU explicit FFI requests. Prove that an exact-only extension retains exact FFI while packed AUTO falls back and explicit packed FFI fails.
- Assert explicit failure occurs during construction and before tracing/lowering/FFI invocation.
- Assert AUTO fallback is silent and resolves to the backend recorded by the operator and benchmark.
- Preserve build-config type/diagnostic tests.

**Verification:**
Run: `uv run pytest -p no:capture tests/jax/test_backend_resolution.py tests/jax/test_ffi_fallback.py tests/jax/test_operator_ffi_cpu.py tests/jax/test_kernels_ffi_cpu.py`
Expected: explicit failure and AUTO fallback contracts pass.

Run: `uv run ruff check src/linear_dag/core/jaxlinarg/operator.py src/linear_dag/core/jaxlinarg/build_config.py src/linear_dag/core/jaxlinarg/packed_products.py src/linear_dag/core/jaxlinarg/_hijax.py src/linear_dag/core/jaxlinarg/kernels/ffi_cpu.py tests/jax`
Expected: Ruff exits successfully.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

**Commit:** `fix(jax): fail explicit unavailable ffi requests`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
