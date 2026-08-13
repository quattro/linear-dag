# Packed Sharded JAX LinearARG Design

## Status
Approved for Implementation

## Handoff Decision
- Current decision: approved
- Ready for implementation: yes
- Blocking items:
  - None.

## Metadata
- Date: 2026-08-13
- Slug: jax-packed-sharded-lineararg
- Artifact Directory: `.plans/design-plans/artifacts/2026-08-13-jax-packed-sharded-lineararg`

## Summary

The design replaces tuples of ragged, per-block JAX objects with a packed graph representation. Ingress balances source blocks across devices, concatenates their fields into a fixed collection of compatible globally sharded arrays, and records descriptors and valid lengths that preserve block boundaries and logical variant order while masking padding. These arrays enter products as dynamic operands, so `shard_map` gives each device only its assigned graph shard instead of capturing graph data as compiled constants.

Each device runs the existing local solve kernels. `matmat` combines device-local sample-space contributions with an explicit collective, while `rmatmat` reconstructs variant-space results in public logical order and communicates only dense data. A private HiJAX compatibility layer supplies transformation rules and treats graph state as opaque; differentiation applies only to dense operands and surrounding model parameters. The exact-ragged operator remains the fallback while the packed path is validated.

## Problem Statement
The current JAX implementation has two incompatible execution modes. The `genoio`/`jax` branch can place a multi-block product inside an outer `jax.jit`, but it stores the graph as a tuple of ragged `JaxLinearARG` PyTrees and closes those arrays over `shard_map` branches. The graph therefore enters the compiled program as constants and can be duplicated or moved away from its intended devices. The `jax-focused` branch fixes graph residency by constructing each block on its assigned device and launching exact-shape per-device programs from Python, but an outer `jax.jit` around a bound multi-block method bypasses that ownership boundary and is intentionally unsupported.

This split prevents a JAX-native LinearARG from serving both production pipelines and exploratory research. Production use needs graph memory proportional to one sharded copy, while new statistical methods need ordinary JAX composition: users should be able to write a loss with an explicit LinearARG argument and apply `jit`, reverse- or forward-mode autodiff, batching, loops, and rematerialization without understanding the triangular solve or supplying a hand-derived gradient.

The physical constraint is ragged graph state. Ordinary global `jax.Array` shards require a statically known, compatible local shape. A tuple containing one differently shaped graph per genomic block cannot be one portable multi-device argument. HiJAX can give the operation a stable primitive and transform rules, but it does not change this storage constraint: its lowered values are still arrays, and arrays placed in primitive parameters or Python closures remain constants. The representation must therefore change before HiJAX can help.

## Definition of Done
1. A packed graph representation lowers every LinearARG dataset to a fixed set of globally sharded JAX arrays whose component count is independent of the number of source blocks. Each device owns only its assigned graph shard plus bounded padding; no device receives a full graph copy during ingress, tracing, execution, or differentiation.
2. The public compiled contract passes `JaxLinearARG` explicitly as a dynamic argument. `jit` and transformed `jit` compositions contain no graph-sized closed-over constants. Bound methods remain available as conveniences, with an explicit safe compilation helper for callers who do not want a functional call surface.
3. `matmat` and `rmatmat` preserve existing sample-by-variant semantics, flipped-variant behavior, dtype behavior, and numerical parity with the current Cython `LinearARG` and `jax-focused` implementations.
4. The graph is opaque and non-learnable. JAX differentiation is supported with respect to dense operands and arbitrary surrounding loss parameters. Forward JVP, reverse VJP, nested/higher-order differentiation, `vmap`, `scan`, and `remat` have explicit, tested behavior; attempts to differentiate graph state fail with an actionable error rather than producing graph cotangents.
5. Multi-device execution uses `shard_map` over packed local shards. Forward products combine sample-space partials with an explicit collective; reverse products compute variant-space local results and reconstruct the exact logical variant order without replicating graph state.
6. Pure JAX, CPU FFI, and accelerator kernels share one project-owned operation contract. HiJAX is isolated behind a private compatibility boundary and does not appear in the public API. The design can replace or remove HiJAX without changing user code.
7. HDF5 and Zarr schemas remain unchanged. Ingress streams source blocks into their final packed shards and reports load balance, padding, and resident-byte metrics. The current exact-ragged `JaxParallelOperator` remains available until numerical, transform, memory, IR, and performance gates pass.
8. The prototype is validated on arm64 and x86_64 CPU hosts and on an available GPU. Benchmarks report cold compilation separately from warm execution and compare the packed implementation with both `jax-focused` and NumPy/Cython baselines.
9. The package requires Python 3.12 or newer and pins the prototype to JAX/JAXlib 0.11.0 until compatibility tests justify a wider range. Direct NumPy and SciPy minimums satisfy JAX 0.11.0 requirements.

Out of scope:
- Differentiating graph topology, edge values, allele metadata, or packing decisions.
- Changing the durable LinearARG HDF5 or Zarr schema.
- Multi-host execution in the first implementation.
- Replacing `LinearARG`, `ParallelOperator`, or the current JAX wrapper before the migration gates pass.
- Exposing HiJAX types, primitives, or version-specific APIs as public package contracts.
- Designing or implementing a graph-preserving subdivision of one durable LinearARG block. Packing assigns whole source blocks; inputs that cannot meet the padding limit after rebalancing require an explicit override or use the exact-ragged fallback.

## Goals and Non-Goals
### Goals
- Keep a single physical copy of graph state distributed across a single-host device mesh.
- Make an explicit `JaxLinearARG` value a valid dynamic operand to compiled JAX programs.
- Support autodiff of arbitrary surrounding loss functions while treating the graph as opaque state.
- Keep standard-array inputs and exact logical outputs so downstream users are not required to manipulate padded graph buffers.
- Reuse the current kernel backends, HDF5/Zarr readers, block metadata, balancing logic, and algebraic oracle.
- Provide measurable gates for graph residency, padding, IR growth, transformation coverage, correctness, and performance.

### Non-Goals
- Making graph state learnable.
- Guaranteeing memory-safe outer JIT for a Python closure that captures a bound operator; the supported JIT contract passes the operator explicitly or uses the safe compilation helper.
- Requiring a public packed-vector type in ordinary downstream losses.
- Treating multiple host processes as one JAX mesh in the first release.
- Using `custom_partitioning` before `shard_map` proves insufficient.
- Replacing the existing CPU FFI or pure-JAX triangular-solve algorithms as part of the representation prototype.

## Existing Patterns
Investigation confirmed the following in-tree patterns and divergences:

- `src/linear_dag/core/jaxlinarg/operator.py` defines `JaxLinearARG` as an Equinox PyTree with graph arrays as dynamic leaves and backend/shape metadata as static fields. The proposed public class keeps the operator abstraction but replaces the per-block physical layout with packed sharded state.
- `src/linear_dag/core/jaxlinarg/wrapper.py` on `jax-focused` assigns contiguous block ranges by `n_entries`, creates HDF5 and Zarr blocks directly on their assigned device, validates exclusive residency, and launches one cached exact-shape program per device range. These placement checks, ingress discipline, and balance metadata remain the memory-correctness oracle.
- `wrapper.py` on `genoio` uses `shard_map` for forward products, but only the dense operand is an explicit mapped input; the tuple of ragged graph blocks is captured by `lax.switch` branches. This design keeps `shard_map` and removes closure capture by making packed graph arrays explicit mapped operands.
- The padding implementation removed in commit `d47b419` padded each individual block into one of several node/edge buckets to reduce recompilation. It did not create a globally sharded graph value, did not bound aggregate per-device padding, and did not make graph state an explicit operand. The new packing policy balances aggregate device shards and pads a fixed set of per-device fields.
- `operator.py` and `wrapper.py` currently use `custom_vjp`, so reverse mode is supported but forward mode deliberately fails. The new operation contract defines both forward and reverse linearization through a private HiJAX layer.
- `src/linear_dag/core/jaxlinarg/kernels/` already separates pure-JAX and CPU-FFI numerical kernels behind `Backend`. Those local kernels remain interchangeable execution engines.
- `src/linear_dag/core/jaxlinarg/ingress.py` is the existing host-array/HDF5/Zarr side-effect boundary. It remains the Imperative Shell. Packing calculations are a Functional Core; mesh placement and collective orchestration remain a justified Mixed boundary.
- `src/linear_dag/core/jaxlinarg/grm.py` composes genotype products into a GRM. The new design keeps GRM as ordinary composition rather than introducing a graph-specific GRM primitive.

The proposal is therefore not a return to the old bucket implementation. It combines the outer-JIT capability attempted on `genoio` with the physical ownership guarantee implemented on `jax-focused`, using a different aggregate representation.

## Architecture
### Packed graph state
Ingress balances source blocks across $D$ devices using estimated graph bytes and solve work. It then concatenates each device's assigned blocks into a fixed set of fields with a leading graph-shard axis, such as graph pointers, edge indices and values, variant/sample mappings, flip flags, allele counts, block descriptors, and valid lengths. Each field has shape `[D, capacity, ...]` and sharding `P("graph", None, ...)`; a device receives one compatible local slice.

Capacity is selected per field from the maximum assigned load after balancing. Padding is masked by valid lengths and cannot contribute to the solve. If aggregate padding exceeds the configured limit after whole-block rebalancing, packing fails with diagnostics unless the caller explicitly overrides the limit. Source block boundaries survive only as descriptor data used by local kernels; they are not separate JAX operands or separate branches in the lowered program.

Dense operands and results remain ordinary logical arrays. A forward product may replicate or shard the dense variant operand because it is much smaller than graph state; each device gathers its logical variant rows using the packed mapping. Reverse execution produces a padded device-local variant buffer, then reconstructs the exact public variant order. That reconstruction may communicate dense result rows but never graph arrays. An internal packed dense-vector form may be retained between compatible library operations, but it is not required of public callers.

### Execution and collectives
The core public operations are project-owned functions conceptually equivalent to:

```text
lineararg_matmat(operator, values) -> sample_values
lineararg_rmatmat(operator, values) -> variant_values
```

The operator is always a dynamic operand of compiled code. A private primitive expands to `shard_map`, where local graph fields and descriptors are explicit mapped operands. The local body invokes the selected pure-JAX, CPU-FFI, or accelerator kernel.

For `matmat`, each graph shard computes its contribution $X_d W_d$. The default collective sums partial sample values and may reduce-scatter them when downstream sharding can retain sample ownership; the exact public result is a normal JAX array. For `rmatmat`, sample values are made available to each graph shard, each shard computes $X_d^T Y$, and the packed local results are unpadded and restored to logical variant order. Collective and output sharding decisions are explicit and verified in StableHLO rather than inferred from Python residency.

### Transform contract
The graph has a zero-only tangent type. For fixed $X$, the derivative rules are:

- `matmat` JVP in dense direction $\dot W$: $X\dot W$.
- `matmat` VJP for cotangent $\bar Y$: $X^T\bar Y$.
- `rmatmat` JVP in dense direction $\dot Y$: $X^T\dot Y$.
- `rmatmat` VJP for cotangent $\bar W$: $X\bar W$.

Rules call the companion public primitive, so nested and higher-order differentiation of surrounding losses remains composable. Batching treats rank-two right-hand sides as the efficient base case and supplies an explicit rule for `vmap`. `scan`, `remat`, dead-code elimination, and symbolic-zero paths preserve graph opacity and do not save graph cotangents or duplicate graph residuals.

HiJAX supplies the private high-level primitive/type and transformation rules. It is not the physical storage layer: its lower type contains the packed arrays with their explicit shardings. The compatibility boundary pins supported JAX versions, contains version-specific construction/derivative helpers, and permits a fallback to lower-level project-owned primitives if the experimental API changes.

### Public API
The supported compiled style passes the operator explicitly:

```text
loss(parameters, operator, phenotype) -> scalar
```

Users apply `jit`, `grad`, `value_and_grad`, or their compositions to this function with differentiation restricted to learnable parameters. Bound `operator.matmat(values)` and `operator.rmatmat(values)` remain available for eager use.

The long-term public type is `JaxLinearARG` for both single-block and multi-block datasets; a single block on one device is the degenerate packed case. HiJAX classes, primitive objects, and the prototype packed carrier remain private. `JaxParallelOperator` remains the public compatibility/fallback path during validation and may delegate to the packed `JaxLinearARG` only after promotion.

`JaxLinearARG.compile_matmat()` and `JaxLinearARG.compile_rmatmat()` are the named safe convenience helpers. They return callables that retain the operator in a Python wrapper but always supply it as a dynamic input to the compiled executable. Raw `jax.jit(lambda values: operator.matmat(values))` closure capture remains outside the supported memory contract because JAX provides no stable public way to distinguish that trace from supported nested transformations.

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: the scientific operator and triangular-solve algorithms already exist. This work ports their physical representation and JAX transformation boundary without changing the genotype model.
- User selection confirmation: the maintainer selected the packed globally sharded design, explicit operator argument, `shard_map`, and private HiJAX layer during the 2026-08-13 design discussion.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | `.plans/design-plans/2026-05-06-jaxlinarg.md` | in-tree design | Original JAX port, algebra, backends, and parity targets. | high |
| SRC-2 | `src/linear_dag/core/lineararg.py` | in-tree implementation | Durable sample-by-variant operator semantics and HDF5 contract. | high |
| SRC-3 | `src/linear_dag/core/jaxlinarg/` at `b68e7da` | in-tree implementation | Memory-aware exact-ragged JAX reference. | high |
| SRC-4 | `genoio` at `c271a9a` | in-tree git reference | Outer-JIT/shard-map implementation used for comparison. | high |

## Model Option Analysis (Required When `suggested-model`)
Not applicable. The model is inherited from `LinearARG`; the design selects a runtime representation rather than a statistical model.

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: retain the current LinearARG algebra and backend kernels while replacing ragged per-block JAX operands with one packed, globally sharded operator state that supports safe outer JIT and the full dense-operand autodiff surface.
- Source selection confirmation: the Cython `LinearARG` is the numerical oracle, `jax-focused` is the memory/residency oracle, and `genoio` is the outer-JIT counterexample and IR baseline.

### Source Pin
| Source ID | Source Type (`local-directory` or `github-url`) | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | /Users/nicholas/Projects/linear-dag/.worktrees/jax-focused | b68e7da71f65cac510567fa74f1d1c6a70f8231d | `src/linear_dag/core/lineararg.py`: algebra, flip correction, shape semantics, durable I/O. |
| PORT-SRC-2 | local-directory | /Users/nicholas/Projects/linear-dag/.worktrees/jax-focused | b68e7da71f65cac510567fa74f1d1c6a70f8231d | `src/linear_dag/core/jaxlinarg/`: backend kernels, native ingress, exact-ragged device placement, GRM. |
| PORT-SRC-3 | local-directory | /Users/nicholas/Projects/linear-dag/.worktrees/jax-focused | c271a9a37e9bf0bf4ad81fbf91808faf54dd0a20 | `genoio` version of `src/linear_dag/core/jaxlinarg/`: closure-capturing `shard_map` and transform tests. |
| PORT-SRC-4 | local-directory | /Users/nicholas/Projects/linear-dag/.worktrees/jax-focused | 12dd8b19a70c0f088fad987ccd3eebcdfd0d2ac6 | Parent of padding-removal commit; comparison only. |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface | Current Behavior | Target Behavior | Evidence Plan |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | numerics | Cython and JAX compute $XW$ and $X^TY$ with flip correction. | Same logical results across packed pure-JAX, FFI, and accelerator paths. | Existing oracle/fixture suite plus packed-vs-exact-ragged differential tests. |
| PORT-BHV-2 | api | Bound single-block methods compose with reverse AD; multi-block outer JIT is unsafe on `jax-focused`. | Functional API accepts an explicit dynamic operator under JIT and all supported transforms. | Jaxpr constant inspection and transform-composition matrix. |
| PORT-BHV-3 | numerics | `custom_vjp` supplies reverse mode and rejects forward mode. | Graph remains nondifferentiable while dense inputs support JVP and VJP. | Analytical adjoint, finite-difference, and nested-transform tests. |
| PORT-BHV-4 | io | HDF5/Zarr ingress constructs exact block objects directly on assigned devices. | Streaming ingress fills final packed local shards without a full default-device copy. | Peak-residency and final-sharding tests for HDF5 and Zarr. |
| PORT-BHV-5 | api | `JaxGRMOperator` composes block products with host-level orchestration. | GRM remains ordinary composition and becomes outer-JIT/autodiff safe through the new primitives. | GRM product, gradient, and RHE integration tests. |
| PORT-BHV-6 | numerics | Existing benchmarks separate cold and warm RHE/product timings. | Benchmarks add packed representation, IR size, padding, transfer, and per-device memory metrics. | Opt-in benchmark tables on arm64, x86_64, and GPU. |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory`
- Investigation completion: yes
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence | Status |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | `jax-focused` operator state | One block contributes graph arrays as individual dynamic PyTree leaves. | `src/linear_dag/core/jaxlinarg/operator.py:134` | confirmed |
| PORT-INV-2 | `jax-focused` placement | Constructors build each block under its assigned default device and validate exclusive residence. | `src/linear_dag/core/jaxlinarg/wrapper.py:305`, `src/linear_dag/core/jaxlinarg/wrapper.py:619` | confirmed |
| PORT-INV-3 | `jax-focused` execution | Multi-device products use Python orchestration around one exact-shape JIT per device range; outer bound JIT is documented as unsafe. | `src/linear_dag/core/jaxlinarg/wrapper.py:101`, `src/linear_dag/core/jaxlinarg/wrapper.py:376` | confirmed |
| PORT-INV-4 | `genoio` execution | `shard_map` maps only the dense argument while graph blocks are referenced from closure branches. | `c271a9a:src/linear_dag/core/jaxlinarg/wrapper.py:330` | discrepancy |
| PORT-INV-5 | `genoio` transforms | Regression coverage intentionally wraps the captured operator in outer `jax.jit`. | `c271a9a:tests/jax/test_wrapper.py:394` | discrepancy |
| PORT-INV-6 | removed bucketing | Earlier padding grouped each block by `(max_nodes, max_nnz)` for cache reuse, not aggregate state sharding. | `d47b419^:src/linear_dag/core/jaxlinarg/padding.py:24` | confirmed |
| PORT-INV-7 | current transforms | `custom_vjp` explicitly disables solve forward mode. | `src/linear_dag/core/jaxlinarg/operator.py:660` | missing |
| PORT-INV-8 | dependency boundary | The project currently pins `jax>=0.10,<0.11`; JAX 0.11.0 is the first release with the documented HiJAX derivative helpers and requires Python 3.12, NumPy 2.1, and SciPy 1.15 or newer. | `pyproject.toml:6`; JAX 0.11.0 package metadata | confirmed |

## External Research Findings (When Triggered)
| Claim ID | Claim | Source URL | Source Type | Access Date | Confidence |
| --- | --- | --- | --- | --- | --- |
| EXT-1 | HiJAX can define a high-level primitive, aggregate type, sharding contract, and distinct forward/reverse transform rules, but the API is experimental. | https://docs.jax.dev/en/latest/hijax_types.html | official documentation | 2026-08-13 | high |
| EXT-2 | HiJAX expansion still lowers aggregate values to ordinary arrays; a one-equation high-level jaxpr does not itself guarantee small lowered IR or graph residency. | https://docs.jax.dev/en/latest/ffi.html | official documentation | 2026-08-13 | high |
| EXT-3 | `shard_map` expresses per-device computation and explicit collectives and composes with JAX transformations. | https://docs.jax.dev/en/latest/notebooks/shard_map.html | official documentation | 2026-08-13 | high |
| EXT-4 | Building one global JAX array from addressable shards requires compatible local shard shapes, motivating aggregate per-device padding. | https://docs.jax.dev/en/latest/_autosummary/jax.make_array_from_single_device_arrays.html | official documentation | 2026-08-13 | high |
| EXT-5 | `custom_partitioning` callback identities can interfere with persistent compilation-cache reuse, so it is not the initial partitioning mechanism. | https://docs.jax.dev/en/latest/persistent_compilation_cache.html | official documentation | 2026-08-13 | high |
| EXT-6 | Experimental JAX APIs do not carry the compatibility guarantees of public APIs. | https://docs.jax.dev/en/latest/api_compatibility.html | official documentation | 2026-08-13 | high |
| EXT-7 | JAX 0.11.0 is the first release with the documented HiJAX derivative helpers and requires Python 3.12 or newer. | https://docs.jax.dev/en/latest/changelog.html#jax-0-11-0-july-16-2026 | official release notes/package metadata | 2026-08-13 | high |

## Mathematical Sanity Checks
- Summary: The representation does not change $X$. If devices own disjoint variant sets $V_d$, then $XW = \sum_d X_{:,V_d}W_{V_d,:}$ and $X^TY$ is the logical-order concatenation/permutation of $X_{:,V_d}^TY$. Padding is algebraically inert when descriptors and valid lengths mask it. Because each product is linear in its dense operand, its JVP is the same product applied to the tangent and its VJP is the adjoint product applied to the cotangent.
- Blocking issues: none in the algebra. The prototype must prove that packing preserves flip correction, nonunique-index compression, and exact logical order.
- Accepted risks: collective order may produce small floating-point differences from host-sequential accumulation. Tolerances remain dtype-specific and compare against both Cython and exact-ragged JAX references.

Detailed artifacts:
- `.plans/design-plans/artifacts/2026-08-13-jax-packed-sharded-lineararg/model-symbol-table.md`
- `.plans/design-plans/artifacts/2026-08-13-jax-packed-sharded-lineararg/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: a memory-conscious JAX LinearARG suitable for production pipelines and arbitrary differentiable research losses, with JIT supported whether or not differentiation is applied.
- Chosen strategy: packed globally sharded graph arrays as the foundational representation; explicit functional operator calls; `shard_map` for per-device execution; a private HiJAX primitive/type layer for transformation rules; existing pure-JAX, CPU-FFI, and accelerator kernels for local solves.
- Why this strategy: packing is the only considered approach that simultaneously gives XLA fixed-shape explicit operands and preserves one physical graph shard per device. HiJAX improves operation identity and transformation rules but cannot repair closure capture or ragged physical storage by itself. `custom_partitioning` and opaque external handles add cache, lifetime, export, and portability costs that are unnecessary for the first prototype.

## Solver Translation Feasibility
- Summary: feasible. Existing local kernels already accept array graph state and dense right-hand sides. Packing changes how those arrays are grouped, sharded, and indexed, not the triangular solve. Local descriptors can drive a static loop over packed blocks, while masks exclude padding.
- Blocking constraints: equal local shapes introduce padding; logical variant reconstruction introduces dense-result communication; HiJAX version stability is uncertain; FFI and accelerator paths must accept the chosen local packed layout or use bounded local views.
- Custom-solver rationale: the LinearARG triangular solve is domain-specific and already implemented. Generic dense, sparse, root-finding, and optimization libraries do not supply this representation or its distributed transform rules.

Detailed artifact:
- `.plans/design-plans/artifacts/2026-08-13-jax-packed-sharded-lineararg/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract: HDF5, Zarr, or in-memory LinearARG blocks plus a concrete single-host mesh produce validated host packing metadata and a fixed set of globally sharded device arrays. Source schemas remain unchanged. Loading is streamed by assigned shard; full graph state is never first materialized on the default device.
- Rejection rules: reject inconsistent sample counts/dtypes, invalid CSC structure, incompatible variant metadata, insufficient capacities, padding above the configured bound after whole-block rebalancing, unsupported mesh topology, or unavailable explicitly requested backend. A padding-bound override must be explicit and visible in diagnostics.

### Pipeline
- Contract: public functional products accept an explicit `JaxLinearARG` and rank-one or rank-two logical dense arrays. Bound eager methods delegate to the same operations. GRM and RHE code compose these operations without accessing graph buffers.
- Validation-first checks: shape, dtype policy, mesh axis, packed descriptor ranges, valid lengths, logical variant permutation, backend availability, and graph sharding are validated at construction. Traced calls see static, already-valid metadata.

### Numerics
- Contract: each local kernel implements forward and transpose products over one packed graph shard. The global primitive defines collectives, logical-order conversion, JVP, transpose/VJP, batching, rematerialization, dead-code elimination, and graph-zero behavior. Graph inputs are never differentiable.
- Result/status semantics: public calls return ordinary JAX arrays or raise before compilation for invalid construction/call boundaries. Explicit unsupported transforms or graph differentiation raise actionable errors. Traced kernels do not encode Python exceptions.

### Egress
- Contract: public sample- and variant-space outputs have the same exact logical shapes and ordering as `LinearARG`. Internal padding is never visible. Optional sharding is represented by normal JAX array sharding, not by altered biological indexing.
- Output/exit-code mapping: not applicable; this is a library contract. CLI/RHE paths retain their current failure semantics.

## Data Conversion and Copy Strategy
| Source | Target | Copy mode | Rationale |
| --- | --- | --- | --- |
| HDF5/Zarr block datasets | assigned host shard staging buffers | single-copy fallback | Durable formats are block-oriented; each source block is read once into its assigned aggregate shard rather than into a default-device operator. |
| assigned host shard buffers | globally sharded packed JAX arrays | single host-to-device copy per shard | `jax.make_array_from_single_device_arrays` assembles global values from compatible local arrays without replicating graph state. |
| in-memory `LinearARG` blocks | assigned host shard staging buffers | single-copy fallback | CSC and metadata arrays are normalized once and concatenated into their final shard layout. |
| logical dense variant/sample arrays | local dense kernel operands | replicated or sharded dense transfer | Dense operands/results may communicate; graph arrays may not. The benchmark reports these bytes separately. |
| packed local variant result | exact logical variant array | one dense-result reconstruction | Required to preserve the public `(n_variants, k)` contract; padding and device ordering remain internal. |

## Multi-Input Reconciliation Contract
Not applicable. This representation consumes already aligned LinearARG blocks. Phenotype/covariate reconciliation remains in the existing association and heritability layers.

## Validation Strategy
- Boundary checks: durable-source metadata, mesh shape/axis, backend support, dtype policy, aggregate capacities, padding ratio, and non-empty graph state.
- Shape/range/domain checks: every descriptor range lies inside its field's valid length; graph indices refer to valid local nodes; variant/sample maps are complete and non-overlapping; logical variant permutation is bijective; padding masks are inert.
- Multi-input alignment checks: not applicable at this layer. RHE integration continues to validate IID alignment before conversion to arrays.
- Failure semantics: invalid packing or placement raises before JIT. Explicit graph differentiation and unsupported transform/version combinations fail with messages that identify the supported functional API. Automatic backend selection may use the documented portable fallback; an explicit unavailable backend fails fast.

## Testing and Verification Strategy
- TDD scope: packing invariants, primitive contracts, transformation rules, ingress, and migration behavior are implemented test-first. Every functionality phase owns tests for its acceptance criteria.
- Regression strategy: compare packed results against both Cython and exact-ragged `jax-focused` operators across existing fixtures, flipped variants, block-count/shape variation, rank-one/rank-two operands, and float32/float64 where enabled. Inspect jaxpr constants, operand counts, array shardings, StableHLO collectives, resident bytes, and padding rather than relying only on numerical outputs.
- Verification commands: use `pytest -p no:capture` for all tests, including forced two-device CPU runs with `XLA_FLAGS=--xla_force_host_platform_device_count=2`; run Ruff and `ty` through the project environment; keep large-data and performance runs opt-in through `--runbench`.

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Packed representation spike
**Goal:** Prove that heterogeneous source blocks can become a fixed set of explicitly sharded arrays with bounded aggregate padding and one physical graph copy.

**Covered ACs:** `jax-packed-sharded-lineararg.AC1.*`, structural portions of `jax-packed-sharded-lineararg.AC2.*`.

**Components:**
- A cohesive packing component within `src/linear_dag/core/jaxlinarg/` owns pure capacity selection, whole-block balance scoring, descriptor construction, padding masks, logical variant mappings, and invariant validation.
- `src/linear_dag/core/jaxlinarg/ingress.py` stages canonical block arrays into assigned host shard buffers and constructs global JAX arrays with explicit graph-axis sharding.
- Memory/IR instrumentation reports unpadded bytes, padded bytes, per-device resident bytes, PyTree/lowered operand count, and graph-sized constants.

**Dependencies:** Existing `jax-focused` ingress, block metadata, and exact-ragged operator as oracles.

**Done when:** two-device CPU fixtures and the production benchmark file construct a valid packed state; numerical unpacking recovers all source arrays; graph residency and padding tests cover the ACs assigned to this phase.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Explicit pure-JAX sharded products
**Goal:** Execute correct forward and reverse products from packed state under outer JIT without graph closure capture.

**Covered ACs:** `jax-packed-sharded-lineararg.AC2.*`, `jax-packed-sharded-lineararg.AC3.*`, `jax-packed-sharded-lineararg.AC4.*`.

**Components:**
- Project-owned functional `matmat` and `rmatmat` boundaries accept the packed operator explicitly and retain bound eager convenience methods.
- `shard_map` local execution uses explicit graph-field `in_specs`, existing pure-JAX solve kernels, explicit forward reduction, and reverse logical-order reconstruction.
- Jaxpr and StableHLO regression tests enforce the operand, constant, sharding, and collective contracts alongside numerical parity.

**Dependencies:** Phase 1.

**Done when:** JIT-compiled forward/reverse products match both numerical oracles, contain no graph-sized constants, preserve final graph residency, and have tests for each covered AC.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Private HiJAX transformation boundary
**Goal:** Give packed products explicit forward/reverse transformation rules while keeping experimental JAX APIs out of the public surface.

**Covered ACs:** `jax-packed-sharded-lineararg.AC5.*`, `jax-packed-sharded-lineararg.AC7.3`, `jax-packed-sharded-lineararg.AC7.4`.

**Components:**
- A private compatibility component owns the HiJAX high-level graph type, product primitives, lower-type sharding, symbolic-zero graph tangent, JVP/VJP/transpose, batching, rematerialization, and dead-code-elimination rules.
- `pyproject.toml` and `uv.lock` move to the validated JAX compatibility range only after the pure packed spike establishes a version-independent baseline.
- Contract tests isolate upstream API changes and verify that public imports and call signatures contain no HiJAX types.

**Dependencies:** Phase 2 and the approved Python 3.12/JAX 0.11.0 dependency migration.

**Done when:** the full primitive transformation unit matrix passes for dense operands, graph differentiation is rejected, public API inspection shows no HiJAX exposure, and tests cover every assigned AC.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Composed autodiff and GRM/RHE integration
**Goal:** Support realistic user losses and the existing heritability pipeline through nested JAX transformations.

**Covered ACs:** `jax-packed-sharded-lineararg.AC5.*`, `jax-packed-sharded-lineararg.AC6.*`.

**Components:**
- Transform-composition coverage includes outer `jit`, `grad`, `value_and_grad`, JVP, nested/higher-order derivatives, `vmap`, `scan`, and `remat` with the operator passed explicitly.
- `src/linear_dag/core/jaxlinarg/grm.py` composes the new products without a separate graph primitive and preserves symmetric-adjoint behavior.
- The JAX RHE integration consumes the new GRM through the same IID-aligned pipeline boundary and remains comparable with the NumPy/Cython estimator.

**Dependencies:** Phase 3.

**Done when:** representative nonlinear losses, multi-device GRM gradients, and RHE smoke/integration tests pass under the supported transform compositions and cover all assigned ACs.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Accelerated local backends
**Goal:** Reuse CPU FFI and accelerator kernels without weakening the packed sharding or derivative contracts.

**Covered ACs:** backend-specific portions of `jax-packed-sharded-lineararg.AC3.*`, `jax-packed-sharded-lineararg.AC5.*`, and `jax-packed-sharded-lineararg.AC7.*`.

**Components:**
- Existing CPU-FFI and accelerator kernels adapt to local packed views behind the same project-owned operation contract.
- Companion forward/transpose primitives implement derivative rules for every backend; unsupported backend/transform combinations fail at selection rather than during lowering.
- Backend-parametrized tests reuse the same algebraic, transformation, sharding, and residency assertions as the pure-JAX path.

**Dependencies:** Phases 3 and 4; existing kernel availability.

**Done when:** every available backend passes its assigned numerical, transform, memory, and failure-path ACs without graph replication.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Streaming ingress and coexistence migration
**Goal:** Make packed operators constructible from supported durable formats while retaining a safe fallback to the exact-ragged implementation.

**Covered ACs:** ingress portions of `jax-packed-sharded-lineararg.AC1.*` and `jax-packed-sharded-lineararg.AC7.*`.

**Components:**
- HDF5, Zarr, and in-memory constructors stream into final assigned shards and expose packing diagnostics; the prototype packed carrier remains private until promotion.
- Promotion makes the packed representation the multi-block implementation of public `JaxLinearARG`, while keeping `JaxParallelOperator` available as the compatibility/fallback facade and leaving durable schemas unchanged.
- Documentation explains the explicit-operator compiled API, eager bound methods, safe compilation helper, backend selection, and migration/fallback behavior.

**Dependencies:** Phases 2 through 5.

**Done when:** all ingress sources satisfy peak/final residency and round-trip parity tests, fallback behavior is covered, and the public documentation matches the supported contract.
<!-- END_PHASE_6 -->

<!-- START_PHASE_7 -->
### Phase 7: Cross-platform gates and promotion decision
**Goal:** Decide whether the packed path can replace the exact-ragged path using reproducible evidence.

**Covered ACs:** `jax-packed-sharded-lineararg.AC8.*` and all earlier ACs as release regressions.

**Components:**
- Opt-in benchmarks compare cold compilation, warm products, GRM/RHE timing, dense communication, padding, peak/final graph residency, and IR size against `jax-focused`, `genoio`, and NumPy/Cython baselines.
- Validation runs cover arm64 CPU, x86_64 CPU, forced two-device CPU, and an available GPU, with JAX persistent-cache behavior recorded.
- A migration review either promotes the packed path, extends coexistence with named blockers, or rejects it without removing the exact-ragged implementation.

**Dependencies:** Phases 1 through 6.

**Done when:** the agreed quantitative gates pass on the reference environments and the migration decision, compatibility range, and retained fallback are documented.
<!-- END_PHASE_7 -->

## Simulation And Inference-Consistency Validation
- In scope: no
- Rationale: This work changes a linear-operator representation, not an inferential model or data-generating process.
- Simulate entrypoint/signature: n/a
- Inputs: existing deterministic oracle fixtures and real benchmark blocks.
- Outputs: algebraic products, derivatives, shardings, IR, and memory/performance metrics.
- Seed/RNG policy: existing randomized numerical tests use explicit JAX keys or NumPy generators; performance inputs are fixed per run.

### Assumption Alignment
| Inference Assumption | Simulation Rule | Mismatch Risk | Mitigation |
| --- | --- | --- | --- |
| Not applicable | No synthetic inferential validation is required. | Representation errors could still alter RHE results. | Differential GRM/RHE tests compare packed, exact-ragged, and NumPy/Cython paths. |

### Planned Validation Experiments
| Experiment ID | Type | Success Criterion | Notes |
| --- | --- | --- | --- |
| SIM-1 | n/a | No inference simulation required. | Algebraic and integration validation is specified above. |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | Aggregate equal-shape packing wastes too much memory for skewed graph fields. | High | Measure per-field padding in Phase 1; rebalance whole blocks by bytes/work, fail above the default bound, and retain an explicit override plus the exact-ragged fallback. | Implementation lead |
| R2 | Reverse logical-order reconstruction or dense collectives erase performance gains. | High | Benchmark communication separately; retain internal packed dense results between compatible operations; consider output sharding/custom partitioning only with HLO evidence. | Implementation lead |
| R3 | HiJAX changes across JAX releases. | High | Keep it private behind one compatibility component, pin a tested range, run upstream/nightly compatibility checks, and preserve a version-independent packed/shard-map core. | Implementation lead |
| R4 | High-level primitive IR is small but expanded StableHLO still scales with source block count. | High | Pack blocks as descriptor data and enforce fixed component/body-size tests at constant capacities; inspect both high jaxpr and StableHLO. | Implementation lead |
| R5 | CPU FFI cannot consume packed local views without copying. | Medium | Permit bounded local slicing/view metadata; if unavoidable copies exceed the memory gate, keep FFI on the exact-ragged path until its ABI changes. | Implementation lead |
| R6 | Explicit operator arguments are less convenient than bound methods. | Low | Keep eager methods and provide a safe compilation helper that supplies the operator dynamically. | Implementation lead |
| Q1 | Should forward sample results default to replication, sample sharding, or follow downstream constraints? | Open | Prototype replicated `psum` and `psum_scatter`; select from measured communication and composability, while preserving the exact logical array contract. | Implementation lead |
| R7 | The reference padding and performance thresholds may not generalize to every dataset or platform. | Medium | Use the approved 25% padding, 65% per-device residency, and 20% warm-runtime limits as default promotion gates; require an explicit documented override rather than silently relaxing them. | Maintainer |

## Additional Considerations
**Module granularity:** packing is a justified cohesive boundary because capacity selection, descriptor construction, and invariants form a reusable pure subsystem. Ingress stays in the existing I/O module, kernels stay in the existing backend package, and the HiJAX seam is one private compatibility component. Passive descriptor/config/result types remain colocated with the component that owns them; this design does not introduce separate `types`, `schemas`, or `helpers` modules.

**Compatibility:** implementation raises the package floor to Python 3.12 and pins JAX/JAXlib 0.11.0, the first release with the documented HiJAX derivative helpers. NumPy and SciPy minimums are raised to at least 2.1 and 1.15. Because HiJAX is experimental and JAX uses effort-based versioning, the supported JAX range remains exact until the compatibility suite passes on another release.

**Export and serialization:** durable reconstruction continues through HDF5/Zarr. Compiled/exported artifacts are not promised to embed graph buffers or survive graph-file removal. Opaque process-local handle registries are intentionally excluded.

**Future partitioning:** `custom_partitioning` is a contingency for a demonstrated `shard_map` limitation, not a parallel implementation track. Adoption requires a specific HLO defect, cache measurements, and the same transform/residency gates.

## Acceptance Criteria

### jax-packed-sharded-lineararg.AC1: Packing preserves state with bounded memory
- **jax-packed-sharded-lineararg.AC1.1 Success:** The packed graph exposes the same fixed set of array components for datasets with different source block counts; only shapes, descriptor values, and valid lengths vary.
- **jax-packed-sharded-lineararg.AC1.2 Success:** Unpacking valid rows reconstructs every canonical source graph field and the exact logical sample/variant mappings.
- **jax-packed-sharded-lineararg.AC1.3 Success:** On the production benchmark, aggregate packed graph bytes excluding separately reported descriptor metadata are no more than 1.25 times the unpadded canonical graph bytes.
- **jax-packed-sharded-lineararg.AC1.4 Success:** On a balanced two-device production load, maximum graph residency on either device is no more than 0.65 times the unpadded total graph bytes, and every graph array resides only on its assigned device.
- **jax-packed-sharded-lineararg.AC1.5 Success:** During HDF5 and Zarr ingress, graph residency on the default device never exceeds that device's final assigned shard plus one source block of staging data.
- **jax-packed-sharded-lineararg.AC1.6 Failure:** If whole-block rebalancing cannot satisfy the configured padding limit, construction fails with diagnostics unless the caller supplies an explicit override; the prototype does not claim to subdivide one source graph block.

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

### jax-packed-sharded-lineararg.AC5: Dense operands support composable JAX transformations
- **jax-packed-sharded-lineararg.AC5.1 Success:** Forward JVPs for `matmat` and `rmatmat` equal the corresponding product applied to the dense tangent.
- **jax-packed-sharded-lineararg.AC5.2 Success:** VJPs equal the companion adjoint product and match analytical and finite-difference references.
- **jax-packed-sharded-lineararg.AC5.3 Success:** `jit`, `jit(grad)`, `grad(jit)`, `value_and_grad`, nested/higher-order derivatives, `vmap`, `scan`, and `remat` pass their documented transform-composition cases with the operator explicit.
- **jax-packed-sharded-lineararg.AC5.4 Success:** Symbolic-zero and dead-code paths do not retain graph cotangents or duplicate graph residuals.
- **jax-packed-sharded-lineararg.AC5.5 Failure:** Requests to differentiate graph topology, edge values, allele metadata, or packing state fail with an actionable message identifying graph state as opaque.

### jax-packed-sharded-lineararg.AC6: GRM, RHE, and research losses compose normally
- **jax-packed-sharded-lineararg.AC6.1 Success:** Packed multi-device GRM products and reverse-mode gradients match the existing symmetric GRM reference.
- **jax-packed-sharded-lineararg.AC6.2 Success:** With fixed probe vectors and seed, JAX RHE outputs match the NumPy/Cython estimator within the existing numerical tolerance.
- **jax-packed-sharded-lineararg.AC6.3 Success:** A representative nonlinear loss with learnable parameters around a LinearARG product runs under outer JIT with JVP, gradient, and a higher-order derivative check without a hand-coded loss gradient.
- **jax-packed-sharded-lineararg.AC6.4 Failure:** IID/phenotype/covariate alignment errors retain the existing fatal pipeline behavior before conversion to packed numerical arrays.

### jax-packed-sharded-lineararg.AC7: Backends, ingress, and public API remain compatible
- **jax-packed-sharded-lineararg.AC7.1 Success:** Pure-JAX, CPU-FFI, and each available accelerator backend pass the same numerical, transform, sharding, and graph-residency contracts.
- **jax-packed-sharded-lineararg.AC7.2 Failure:** An explicitly requested unavailable or transform-incompatible backend fails before lowering; `Backend.AUTO` uses only its documented fallback.
- **jax-packed-sharded-lineararg.AC7.3 Success:** HDF5 and Zarr inputs require no schema change and reconstruct the same logical operator as current readers.
- **jax-packed-sharded-lineararg.AC7.4 Success:** Public package signatures, annotations, PyTree inspection, and documentation expose no HiJAX types or primitives.
- **jax-packed-sharded-lineararg.AC7.5 Success:** The target public `JaxLinearARG` supports single- and multi-block packed datasets after promotion; the exact-ragged `JaxParallelOperator` remains constructible and covered as the compatibility/fallback path.
- **jax-packed-sharded-lineararg.AC7.6 Success:** Package metadata, lockfile, classifiers, and test matrices require Python 3.12 or newer, JAX/JAXlib 0.11.0, NumPy 2.1 or newer, and SciPy 1.15 or newer; the compatibility suite must pass before widening the exact JAX pin.

### jax-packed-sharded-lineararg.AC8: Promotion is evidence-gated across platforms
- **jax-packed-sharded-lineararg.AC8.1 Success:** Correctness, transform, IR, and residency suites pass on arm64 CPU, x86_64 CPU, forced two-device CPU, and an available GPU before promotion.
- **jax-packed-sharded-lineararg.AC8.2 Success:** Benchmarks report cold construction/compilation separately from warm execution and include dense communication, padding, peak/final graph residency, and IR metrics.
- **jax-packed-sharded-lineararg.AC8.3 Success:** At $K\in\{4,20\}$, warm packed-product median runtime is no more than 1.20 times the `jax-focused` median for the equivalent backend and device count on reference environments.
- **jax-packed-sharded-lineararg.AC8.4 Success:** Benchmark output retains NumPy/Cython comparisons for products and RHE but does not conflate cold and warm ratios.
- **jax-packed-sharded-lineararg.AC8.5 Failure:** If any promotion gate fails, the packed path remains experimental, the exact-ragged path remains available, and the blocker is recorded instead of silently weakening the gate.

## Glossary

- **Aggregate padding**: Unused capacity added to device shards so each local slice of a packed field has the same static shape.
- **Closure capture**: Referencing graph arrays from a compiled Python function's enclosing scope, which makes them constants instead of explicit runtime operands.
- **Descriptor**: Packed metadata identifying valid ranges, source-block boundaries, and logical index mappings within concatenated shard buffers.
- **Exact-ragged representation**: The current layout in which each source block keeps its natural shape and executes through Python-orchestrated per-device programs.
- **Graph residency**: The physical devices and memory locations holding graph arrays, including whether those arrays have been replicated.
- **Graph shard**: The packed portion of graph state assigned exclusively to one device.
- **HiJAX**: An experimental JAX facility used privately here to define an aggregate graph type, primitive operations, and transformation rules.
- **Jaxpr**: JAX's intermediate program representation, inspected to verify that graph arrays are inputs rather than closed-over constants.
- **JVP**: A forward-mode Jacobian-vector product; for these linear operators, it applies the same product to the dense input tangent.
- **Logical variant order**: The public variant ordering expected by callers, reconstructed from device-local packed order after reverse products.
- **`matmat`**: The forward LinearARG product $XW$, mapping variant-space values to sample-space values.
- **Packed graph state**: A fixed collection of globally sharded arrays formed by concatenating ragged source blocks and recording their boundaries as data.
- **`psum`**: A collective that sums per-device sample-space contributions and replicates the result.
- **`psum_scatter`**: A collective that sums per-device contributions while distributing portions of the result across devices.
- **`rmatmat`**: The adjoint LinearARG product $X^T Y$, mapping sample-space values to variant-space values.
- **Safe compilation helper**: `compile_matmat` or `compile_rmatmat`, which provides bound-method convenience while still passing the operator dynamically to compiled code.
- **`shard_map`**: A JAX construct defining a per-device computation with explicit input shardings and collectives.
- **StableHLO**: The lowered compiler representation inspected to verify collectives, operand shardings, and the absence of graph broadcasts.
- **Valid length**: The boundary between meaningful packed entries and inert padding within a shard field.
- **VJP**: A reverse-mode vector-Jacobian product; here it invokes the companion adjoint LinearARG operation.
- **Zero-only tangent**: A differentiation contract declaring that graph state cannot vary and therefore has no nonzero tangent.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-08-13 | N/A | Draft | Plan created | |
| 2026-08-13 | Draft | In Review | Architecture and acceptance criteria documented for readiness validation. | Codex |
| 2026-08-13 | In Review | Approved for Implementation | Maintainer approved the acceptance criteria and the approval readiness gate passed. | Maintainer / Codex |
| 2026-08-13 | Approved for Implementation | Approved for Implementation | Maintainer required Python 3.12/JAX 0.11.0 and removed graph splitting from the prototype after implementation investigation. | Maintainer / Codex |
