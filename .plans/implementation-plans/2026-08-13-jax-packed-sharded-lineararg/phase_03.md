# Private HiJAX Transformation Boundary Implementation Plan

**Goal:** Represent packed graph state as one private high-level JAX value and give forward/transpose products explicit, composable transformation rules without exposing experimental HiJAX classes publicly.

**Architecture:** Raise the supported runtime to Python 3.12 and JAX/JAXlib 0.11.0, then isolate all HiJAX imports and registrations in `_hijax.py`. A private non-PyTree graph value lowers to the fixed Phase 1 array components. Paired product primitives expand through Phase 2's explicit `shard_map` functions and treat the graph as opaque, immutable, and nondifferentiable.

**Tech Stack:** Python 3.12-3.14, JAX/JAXlib 0.11.0, `jax.experimental.hijax`, NumPy 2.1+, SciPy 1.15+, Equinox, pytest.

**Scope:** Phase 3 of 7 from the approved design.

**Codebase verified:** 2026-08-13 at `19bba4d`; API signatures and upstream behavior verified against local JAX tag `jax-v0.11.0` at `a1521744c6dc074443fe549f19f48d7197abf759`.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### jax-packed-sharded-lineararg.AC5: Dense operands support composable JAX transformations
- **jax-packed-sharded-lineararg.AC5.1 Success:** Forward JVPs for `matmat` and `rmatmat` equal the corresponding product applied to the dense tangent.
- **jax-packed-sharded-lineararg.AC5.2 Success:** VJPs equal the companion adjoint product and match analytical and finite-difference references.
- **jax-packed-sharded-lineararg.AC5.3 Success:** `jit`, `jit(grad)`, `grad(jit)`, `value_and_grad`, nested/higher-order derivatives, `vmap`, `scan`, and `remat` pass their documented transform-composition cases with the operator explicit.
- **jax-packed-sharded-lineararg.AC5.4 Success:** Symbolic-zero and dead-code paths do not retain graph cotangents or duplicate graph residuals.
- **jax-packed-sharded-lineararg.AC5.5 Failure:** Requests to differentiate graph topology, edge values, allele metadata, or packing state fail with an actionable message identifying graph state as opaque.

### jax-packed-sharded-lineararg.AC7: Backends, ingress, and public API remain compatible
- **jax-packed-sharded-lineararg.AC7.4 Success:** Public package signatures, annotations, PyTree inspection, and documentation expose no HiJAX types or primitives.
- **jax-packed-sharded-lineararg.AC7.6 Success:** Package metadata, lockfile, classifiers, and test matrices require Python `>=3.12,<3.15`, JAX/JAXlib 0.11.0, NumPy 2.1 or newer, and SciPy 1.15 or newer; the compatibility suite must pass before widening the exact JAX pin.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Raise and lock the supported Python/JAX baseline

**Verifies:** jax-packed-sharded-lineararg.AC7.6

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: `uv.lock`

**Implementation:**
- Change `requires-python` to `>=3.12,<3.15`, remove the Python 3.11 classifier and Hatch matrix entry, and change Ruff's target to `py312`.
- Pin both build and runtime dependencies to `jax==0.11.0` and `jaxlib==0.11.0`. Keep the exact pin until the full compatibility suite passes on another JAX release.
- Set direct dependency floors to `numpy>=2.1` and `scipy>=1.15`; apply the same floors where build hooks name those packages.
- Remove the `uv.lock` ignore rule, regenerate the lock from the edited metadata, and commit it. Inspect the resolved Python markers and JAX/JAXlib versions rather than hand-editing the lockfile.
- Do not add a Python 3.11 compatibility extra or conditional older JAX range; HiJAX 0.11 is the selected prototype baseline.
- Recreate/synchronize the project environment with Python 3.12 before running migrated tests. Do not reuse the Phase 1/2 Python 3.11 environment after changing `requires-python`.

**Testing:**
- Assert package metadata excludes Python 3.11 and contains the exact JAX/JAXlib pins and direct NumPy/SciPy floors.
- Run collection and the existing JAX unit suite under Python 3.12 before adding HiJAX code, so dependency migration failures are separated from primitive failures.
- Build metadata must reject Python 3.11 during resolution and admit Python 3.12, 3.13, and 3.14.

**Verification:**
Run: `uv lock`
Expected: resolution succeeds with JAX/JAXlib 0.11.0 and no Python 3.11 solution.

Run: `uv sync --python 3.12`
Expected: the project environment is recreated/synchronized with Python 3.12 and the locked dependencies.

Run: `uv run --python 3.12 python -c "import jax, numpy, scipy; assert jax.__version__ == '0.11.0'; print(jax.__version__, numpy.__version__, scipy.__version__)"`
Expected: JAX is 0.11.0, NumPy is at least 2.1, and SciPy is at least 1.15.

Run: `uv run --python 3.12 pytest -p no:capture tests/jax --ignore=tests/jax/bench`
Expected: the pre-HiJAX JAX suite passes under the new dependency floor.

**Commit:** `build: require python 3.12 and jax 0.11`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Define the private opaque graph value and high-level type

**Verifies:** jax-packed-sharded-lineararg.AC5.4, jax-packed-sharded-lineararg.AC5.5, jax-packed-sharded-lineararg.AC7.4

**Files:**
- Create: `src/linear_dag/core/jaxlinarg/_hijax.py`
- Modify: `src/linear_dag/core/jaxlinarg/packing.py`
- Create: `tests/jax/test_hijax.py`

**Implementation:**
- Classify `_hijax.py` as `# pattern: Mixed (unavoidable)` because it owns JAX transformation registration and lowering while delegating numerical work to project functions. Keep `packing.py` a Functional Core.
- Define a private, frozen, non-PyTree `_PackedGraphValue` whose only runtime payload is the fixed ordered tuple of packed graph arrays from Phase 1. Do not subclass `eqx.Module`, register it as a PyTree, or store source block objects.
- Define a hashable `_PackedGraphType(jax.experimental.hijax.HiType)` from abstract component types plus compact immutable logical metadata. `lo_ty()` must return a fixed component list determined entirely by the abstract type; `lower_val()` and `raise_val()` must preserve that exact order.
- Register `_PackedGraphValue` with `register_hitype`. Keep registration and every HiJAX import private to `_hijax.py`; `packing.py` constructs the value through a private adapter imported from `_hijax.py` but does not mention HiJAX types in public annotations.
- Define private graph mapping/sharding specs implementing `MappingSpec` and `HiPspec`. `HiPspec.to_lo()` must return the fixed graph-axis `PartitionSpec` prefix for every lowered component, and tangent/cotangent specs must use the graph-zero contract.
- Define a private graph-zero value/type with no graph array payload. Its vector-space zero/add operations remain inert. Primitive linearization rules must inspect `nzs_in` and reject an actually differentiated graph input; the zero type is only for inactive symbolic tangents/cotangents.
- Implement `dec_rank`, `inc_rank`, and `leading_axis_spec` so mapped graph axes are rejected and an invariant graph can pass through `vmap`/`scan`. Implement `shard`, `unshard`, and `nospec` by transforming/checking the lowered component shardings; reject a mesh without the `"graph"` axis.
- Make the Phase 1 convenience carrier hold `_PackedGraphValue` as one dynamic leaf plus compact static logical metadata. Do not put graph arrays into primitive `params` or closures.

**Testing:**
- Prove `jax.typeof(graph_value)` is the private graph type and that flattening the containing convenience carrier exposes one high-level graph leaf rather than one public leaf per source block.
- Round-trip `lower_val`/`raise_val` and compare all components, shapes, dtypes, and shardings.
- Verify type equality/hash changes for capacity, dtype, shape, or sharding changes but not for source block count at fixed packed component types.
- Exercise invariant `vmap`/`scan` specs, graph-axis `shard_map` specs, graph-zero construction/addition, and actionable failures for mapped/mutated/differentiated graph values.
- Inspect `linear_dag`, `linear_dag.core`, and `linear_dag.core.jaxlinarg` exports and public annotations; none may contain `_hijax`, `HiType`, `HiPspec`, `MappingSpec`, or primitive classes.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_hijax.py -k 'type or lower or sharding or public'`
Expected: opaque-value, lowering, sharding, and public-isolation tests pass.

**Commit:** `feat(jax): add private packed graph hitype`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Bind paired HiJAX product primitives and derivative rules

**Verifies:** jax-packed-sharded-lineararg.AC5.1, jax-packed-sharded-lineararg.AC5.2, jax-packed-sharded-lineararg.AC5.3, jax-packed-sharded-lineararg.AC5.4, jax-packed-sharded-lineararg.AC5.5

**Files:**
- Modify: `src/linear_dag/core/jaxlinarg/_hijax.py`
- Modify: `src/linear_dag/core/jaxlinarg/packed_products.py`
- Modify: `src/linear_dag/core/jaxlinarg/packing.py`
- Modify: `tests/jax/test_hijax.py`

**Implementation:**
- Define private paired `VJPHiPrimitive` subclasses for `matmat(graph, values)` and `rmatmat(graph, values)`. Their `params` contain only hashable shapes/dtypes/backend signature data, never graph arrays, Python operator instances, readers, meshes, or callbacks.
- `expand()` delegates to Phase 2's explicit lower-level `shard_map` product with `_PackedGraphValue` and the dense operand as arguments. Do not inline duplicate LinearARG algebra into `_hijax.py`.
- Handwrite `lin(nzs_in, ...)` and `linearized(residuals, ...)`: reject a nonzero graph tangent, retain only the graph value as the linearization residual, and bind the same product primitive for the dense tangent. Set `jvp = jvp_from_lin`.
- Handwrite `vjp_fwd`/`vjp_bwd_retval` or use `vjp_fwd_from_lin` only after tests prove residuals contain no dense primal unnecessarily. Backward returns no graph cotangent and binds the companion primitive for the dense cotangent: `matmat` transposes through `rmatmat`, and `rmatmat` transposes through `matmat`.
- Implement explicit transpose accumulation with the same companion-primitive contract. Do not call `expand()` from a derivative rule; this is required for higher-order differentiation and transform composition.
- Implement dense batching with graph `in_dim=None` only. Move the dense mapped axis to a known position, fuse batch and right-hand-side axes into one rank-two product, bind one primitive call, and restore the mapped output axis. Reject any mapped graph dimension.
- Retain default full rematerialization and DCE behavior unless a focused failing test demonstrates a smaller correct rule. Test both so later upstream changes are detected.
- Route the private carrier's `matmat`, `rmatmat`, `compile_matmat`, and `compile_rmatmat` through these binders. The safe compile helpers must still pass the carrier explicitly at the Python/JIT boundary.

**Testing:**
- Compare primal, JVP, VJP, linear transpose, and second derivative results with dense/Cython and exact-ragged oracles.
- Assert the forward tangent invokes `matmat` and reverse cotangent invokes `rmatmat`; test the reciprocal rule for `rmatmat`.
- Prove graph differentiation and graph batching raise `TypeError` with guidance to use the graph as an invariant opaque operand.
- Test dense `vmap` for vector and matrix right-hand sides, invariant graph through `scan`, `remat`, DCE of unused results, and symbolic-zero dense tangents.
- Inspect the high-level jaxpr for one project product equation and the lowered jaxpr/StableHLO for the Phase 2 explicit graph operands and collective contract. Do not claim that one high-level equation alone reduces lowered IR.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_hijax.py`
Expected: all primitive and unit transformation tests pass on two CPU devices.

**Commit:** `feat(jax): define packed product transform rules`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Lock the private compatibility and transform-unit contract

**Verifies:** jax-packed-sharded-lineararg.AC5.3, jax-packed-sharded-lineararg.AC5.4, jax-packed-sharded-lineararg.AC7.4, jax-packed-sharded-lineararg.AC7.6

**Files:**
- Modify: `tests/jax/test_hijax.py`
- Modify: `tests/jax/test_packed_products.py`
- Modify: `src/linear_dag/core/jaxlinarg/_hijax.py`

**Implementation:**
- Add one private compatibility assertion at import time for the exact HiJAX names and callable signatures used by the adapter. Raise an actionable `ImportError` naming supported JAX 0.11.0 if the experimental surface is absent. Do not scatter version checks through product modules.
- Keep a project-owned binder surface in `_hijax.py` so later HiJAX replacement affects only that file and private tests.
- Add source/inspection tests that graph arrays occur only in `lower_val` results and primitive positional inputs, never primitive params or closure cells.
- Preserve the existing exact-ragged custom-VJP behavior and its forward-mode rejection tests; Phase 3 changes only the packed path.

**Testing:**
- Cover the unit transform matrix for `jit`, `jvp`, `vjp`, `linearize`, `linear_transpose`, `vmap`, `scan`, `remat`, DCE, and one second-order dense derivative.
- Monkeypatch/remove one required HiJAX symbol in an isolated subprocess and assert the compatibility error is actionable.
- Re-run Phase 2 constant, sharding, and StableHLO tests through the HiJAX binder to prove high-level opacity did not reintroduce closure constants or graph collectives.
- Assert public module `__all__`, signatures, generated annotations, and PyTree inspection do not expose the compatibility layer.

**Verification:**
Run: `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_hijax.py tests/jax/test_packed_products.py`
Expected: primitive compatibility, transform-unit, IR, and residency tests pass.

Run: `uv run ruff check src/linear_dag/core/jaxlinarg/_hijax.py src/linear_dag/core/jaxlinarg/packing.py src/linear_dag/core/jaxlinarg/packed_products.py tests/jax/test_hijax.py tests/jax/test_packed_products.py`
Expected: Ruff exits successfully.

Run: `uv run ty check src tests`
Expected: type checking exits successfully.

**Commit:** `test(jax): lock hijax compatibility contract`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
