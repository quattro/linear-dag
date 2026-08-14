# Composed Autodiff and GRM/RHE Integration Implementation Plan

**Goal:** Make the packed operator usable inside realistic nonlinear losses, GRM products, and the JAX RHE pipeline while preserving the exact-ragged fallback.

**Architecture:** Keep LinearARG products as the only graph primitives. Express GRM centering/weighting and research losses as ordinary JAX composition over those products. Refactor the packed RHE path so every compiled boundary receives the GRM/operator explicitly; retain Python blockwise orchestration only for `JaxParallelOperator`.

**Tech Stack:** JAX/HiJAX product primitives, Equinox, Polars IID alignment, NumPy/Cython RHE oracle, pytest.

**Scope:** Phase 4 of 7 from the approved design.

**Codebase verified:** 2026-08-13 at `19bba4d` using `grm.py`, `_heritability_jax.py`, `alignment.py`, and their current tests.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### jax-packed-sharded-lineararg.AC5: Dense operands support composable JAX transformations
- **jax-packed-sharded-lineararg.AC5.3 Success:** `jit`, `jit(grad)`, `grad(jit)`, `value_and_grad`, nested/higher-order derivatives, `vmap`, `scan`, and `remat` pass their documented transform-composition cases with the operator explicit.
- **jax-packed-sharded-lineararg.AC5.4 Success:** Symbolic-zero and dead-code paths do not retain graph cotangents or duplicate graph residuals.

### jax-packed-sharded-lineararg.AC6: GRM, RHE, and research losses compose normally
- **jax-packed-sharded-lineararg.AC6.1 Success:** Packed multi-device GRM products and reverse-mode gradients match the existing symmetric GRM reference.
- **jax-packed-sharded-lineararg.AC6.2 Success:** With fixed probe vectors and seed, JAX RHE outputs match the NumPy/Cython estimator within the existing numerical tolerance.
- **jax-packed-sharded-lineararg.AC6.3 Success:** A representative nonlinear loss with learnable parameters around a LinearARG product runs under outer JIT with JVP, gradient, and a higher-order derivative check without a hand-coded loss gradient.
- **jax-packed-sharded-lineararg.AC6.4 Failure:** IID/phenotype/covariate alignment errors retain the existing fatal pipeline behavior before conversion to packed numerical arrays.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Compose packed GRM algebra over product primitives

**Verifies:** jax-packed-sharded-lineararg.AC5.3, jax-packed-sharded-lineararg.AC5.4, jax-packed-sharded-lineararg.AC6.1

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/grm.py`
- Modify: `tests/jax/test_grm_operator.py`

**Implementation:**
- Keep `grm.py` classified `# pattern: Mixed (unavoidable)`: GRM algebra is functional, while the exact-ragged branch still orchestrates device-local calls.
- Add the private packed carrier to `JaxGRMOperator`'s accepted operator types without exposing that carrier in public annotations. Use a private protocol/helper for `shape`, `dtype`, `allele_counts`, `matmat`, and `rmatmat` if this avoids a public HiJAX name.
- For the packed path, implement the existing formula as ordinary JAX operations around the Phase 3 product binders: compute allele frequencies/weights in exact logical variant order, apply `rmatmat`, center and weight, then apply `matmat`. Do not add a GRM-specific primitive.
- Remove the local `jax.custom_vjp` wrapper for packed GRM calls so both forward and reverse modes derive from the paired LinearARG primitives. Preserve the current symmetric custom-VJP/exact blockwise implementation for `JaxParallelOperator` until fallback removal is separately approved.
- Add a module-level functional entry point used as `jax.jit(_packed_grm_matmat)(grm_or_operator, values, ...)`, with the packed graph supplied explicitly. A bound eager `JaxGRMOperator.matmat` delegates to it. Do not add a GRM compile helper in this phase; compiled callers use the module-level explicit-operator function.
- Keep `matmat_blockwise`, `_device_blocks`, and range programs exact-ragged only. Make packed dispatch unambiguous so `_should_use_blockwise_grm` cannot select this fallback merely because a method exists.
- Preserve rank-one/rank-two conventions, dtype conversion, `alpha`, centering, allele-count semantics, symmetric `rmatmat`, and metadata/IID behavior.

**Testing:**
- Compare packed one-/two-device GRM products with dense, exact-ragged, and Cython references for centered and uncentered cases and representative `alpha` values.
- Compare `jax.grad` and `jax.jvp` through packed GRM with the symmetric adjoint and finite differences; include a two-device reverse-mode regression.
- Run outer `jit` with the GRM/operator explicit and assert zero graph-sized constants, unchanged graph residency, and no graph collective.
- Prove the exact-ragged fallback still uses direct/blockwise calls and preserves its documented bound-outer-JIT restriction.
- Cover rank errors, incompatible allele counts, zero-frequency variants, and dtype/x64 behavior.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_grm_operator.py`
Expected: packed and exact-ragged GRM numerical, derivative, IR, and fallback tests pass.

**Commit:** `feat(jax): compose packed grm products`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add a representative nonlinear transform-composition suite

**Verifies:** jax-packed-sharded-lineararg.AC5.3, jax-packed-sharded-lineararg.AC5.4, jax-packed-sharded-lineararg.AC6.3

**Files:**
- Create: `tests/jax/test_transform_composition.py`

**Implementation:**
- Keep the composition suite in one test module; do not add production abstractions for test-only losses or transform matrices.
- Define the deterministic nonlinear research loss `mean((tanh(operator.matmat(theta)) - target) ** 2) + 1e-3 * sum(theta ** 2)` with learnable dense `theta`. The packed operator/graph is an explicit nondifferentiated argument; no test supplies a custom gradient for the loss.
- Parameterize the supported compositions rather than duplicating fixtures: `jit`, `jit(grad)`, `grad(jit)`, `value_and_grad`, JVP, VJP, Hessian-vector or second directional derivative, `vmap` with invariant graph, `scan` with invariant graph, and `remat`.
- Use analytical/dense and finite-difference references appropriate to each derivative order. Keep finite differences small and deterministic; they supplement, not replace, exact adjoint comparisons.
- Inspect recursive jaxprs for graph constant bytes and duplicate graph residuals in `jit(grad)`, `grad(jit)`, higher-order, and rematerialized cases.

**Testing:**
- Exercise rank-one and multi-column dense values, one device, and forced two-device CPU.
- Verify primal values, first derivatives, JVPs, and one higher-order check within dtype-appropriate tolerance.
- Verify mapped graph axes and graph differentiation fail with the Phase 3 actionable errors.
- Verify source block count at fixed capacity does not alter the transform program structure.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_transform_composition.py`
Expected: every documented transform composition passes and negative graph-transform cases fail as designed.

**Commit:** `test(jax): cover composed packed transforms`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (task 3) -->
<!-- START_TASK_3 -->
### Task 3: Integrate explicit packed GRM execution into JAX RHE

**Verifies:** jax-packed-sharded-lineararg.AC6.2, jax-packed-sharded-lineararg.AC6.4

**Files:**
- Modify: `src/linear_dag/association/_heritability_jax.py`
- Modify: `src/linear_dag/core/alignment.py`
- Modify: `tests/association/test_heritability_jax.py`
- Modify: `tests/association/test_rhe.py`
- Modify: `tests/core/test_alignment.py`

**Implementation:**
- Keep `_heritability_jax.py` classified Mixed and `alignment.py` classified Functional Core.
- Replace the packed path's captured `base_matmat` closure with `_ResidualizedJaxGRM(eqx.Module)`. Store the packed GRM, JAX IID index arrays, and covariate basis as dynamic fields and shape/residual rank as static fields. Its bound method calls a module-level jitted function with the residualized operator supplied explicitly; do not assign `jax.jit(self._matmat)` or otherwise close over `self`.
- Keep alignment gather/scatter and projection algebra in the module-level function. Graph arrays stay inside the explicit packed GRM argument; small alignment and basis arrays remain ordinary operands.
- Change `_should_use_blockwise_grm` to select blockwise execution only for the exact-ragged `JaxParallelOperator` path and retain the environment opt-out. Packed operators must use the composable functional path.
- Preserve current RHE entrypoint signature, samplers, stochastic estimators, fixed seed/probe behavior, intercept checks, output columns, and NumPy/Cython comparison tolerance.
- Enforce the repository's fatal merge invariant in `get_iid_alignment`: an empty IID intersection raises before any JAX/NumPy numerical conversion. Retain existing duplicate/diploid multiplicity validation and add explicit failures for missing phenotype/covariate columns, non-intercept first covariate, all-missing phenotypes, and inconsistent IID multiplicity where not already covered.
- Apply the zero-overlap check to the shared alignment core so NumPy/Cython and JAX paths fail consistently; do not create a JAX-only exception.

**Testing:**
- AC6.2: compare packed pure-JAX RHE with existing NumPy/Cython RHE for fixed Hutchinson probes/seeds at `num_matvecs` 4 and 20 where runtime permits; retain reordered phenotype coverage.
- Exercise the functional packed path under forced two-device CPU and assert the residualized helper's jaxpr has no graph constants. Prove exact-ragged still takes the blockwise fallback.
- AC6.4: test zero IID overlap, missing IIDs, invalid diploid multiplicity, missing phenotype/covariate columns, invalid intercept, and all-missing phenotype. Assert failure happens before graph products using a recording operator.
- Re-run shared alignment NumPy/JAX gather/scatter parity after adding the empty-intersection failure.

**Verification:**
Run: `uv run pytest -p no:capture tests/core/test_alignment.py tests/association/test_rhe.py tests/association/test_heritability_jax.py`
Expected: shared alignment and single-device RHE tests pass.

Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/association/test_heritability_jax.py -k 'packed or explicit or matches'`
Expected: packed two-device RHE and explicit-operand tests pass.

Run: `uv run ruff check src/linear_dag/core/jaxlinarg/grm.py src/linear_dag/association/_heritability_jax.py src/linear_dag/core/alignment.py tests/jax/test_grm_operator.py tests/jax/test_transform_composition.py tests/association/test_heritability_jax.py tests/association/test_rhe.py tests/core/test_alignment.py`
Expected: Ruff exits successfully.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

**Commit:** `feat(jax): integrate packed grm with rhe`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
