# Packed Sharded JAX LinearARG Test Requirements

## Purpose and status

This is a pre-implementation test contract for the seven-phase packed JAX LinearARG plan. It maps every atomic acceptance-criteria case to the task that owns it, the planned automated evidence, an executable verification command, and the required outcome.

The mapping is complete at the plan level: all 40 atomic AC cases have an automated test or evidence evaluator. The planned files do not all exist yet and no implementation test execution is claimed here. Commands become required after their owning tasks land.

This increment ends with a private `_PackedJaxLinearARG` candidate and a promotion decision. It does not rename or export the candidate, replace the public exact `JaxLinearARG`/`JaxParallelOperator`, or reroute the CLI. A `promote` evidence result authorizes a separate approved public-migration plan; it does not make that migration part of this test contract.

## Task ID convention

Task IDs use `P<phase>-T<task>`. For example, `P3-T2` means Phase 3, Task 2 in `phase_03.md`.

## Verification command registry

Every pytest invocation includes the repository-required `-p no:capture` option. Paths created by later phases are prospective until their owning task is implemented.

| ID | Command | Purpose |
| --- | --- | --- |
| V-PACK | `uv run pytest -p no:capture tests/jax/test_packing.py tests/jax/test_ingress.py` | Packing, lossless reconstruction, validation, and single-device ingress. |
| V-PACK-2D | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_packing.py tests/jax/test_ingress.py` | Forced two-device placement, sharding, and streaming residency. |
| V-PACK-PROD | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/bench/test_parallel_benchmarks.py --runbench --linarg-h5-path /absolute/path/to/representative-lineararg.h5 --linarg-parallel-processes 2 --linarg-benchmark-k 4 20` | Production padding and per-device graph-residency gates. |
| V-PRODUCT | `uv run pytest -p no:capture tests/jax/test_packed_products.py` | Local/single-device forward and transpose products, validation, and IR checks. |
| V-PRODUCT-2D | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_packed_products.py` | Multi-device products, collectives, shard ownership, and StableHLO checks. |
| V-HIJAX | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_hijax.py tests/jax/test_packed_products.py` | Private HiJAX type, primitive, AD, batching, DCE, remat, compatibility, and lowered-contract tests. |
| V-GRM | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_grm_operator.py` | Packed/exact GRM parity, gradients, JVP, IR, and fallback behavior. |
| V-COMPOSE | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_transform_composition.py` | Nonlinear loss and composed `jit`/AD/`vmap`/`scan`/`remat` matrix. |
| V-RHE | `uv run pytest -p no:capture tests/core/test_alignment.py tests/association/test_rhe.py tests/association/test_heritability_jax.py` | Shared IID failures and packed JAX versus NumPy/Cython RHE parity. |
| V-RHE-2D | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/association/test_heritability_jax.py -k 'packed or explicit or matches'` | Explicit-operand packed RHE on two devices. |
| V-FFI-BUILD | `LINEAR_DAG_REQUIRE_FFI_CPU=1 uv build` | Native extension build with exact and packed CPU-FFI targets. |
| V-FFI-ABI | `JAX_ENABLE_X64=1 uv run pytest -p no:capture tests/jax/test_kernels_ffi_cpu.py` | Exact/packed ABI, descriptor safety, float32/float64, and registration tests. |
| V-FFI-PACKED | `XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 uv run pytest -p no:capture tests/jax/test_packed_products.py tests/jax/test_hijax.py tests/jax/test_operator_ffi_cpu.py -k 'ffi or backend'` | Packed CPU-FFI numerical, transform, IR, and residency matrix. |
| V-BACKEND | `uv run pytest -p no:capture tests/jax/test_backend_resolution.py tests/jax/test_ffi_fallback.py tests/jax/test_operator_ffi_cpu.py tests/jax/test_kernels_ffi_cpu.py` | Representation-aware exact/packed backend resolution and early failures. |
| V-COEXIST | `uv run pytest -p no:capture tests/jax/test_coexistence.py tests/jax/test_wrapper.py tests/jax/test_grm_operator.py tests/cli/test_cli.py` | Private/public isolation, exact-ragged compatibility, and unchanged CLI routing. |
| V-DEPS | `uv lock && uv sync --python 3.12 && uv run --python 3.12 python -c "import jax, numpy, scipy; assert jax.__version__ == '0.11.0'; assert tuple(map(int, numpy.__version__.split('.')[:2])) >= (2, 1); assert tuple(map(int, scipy.__version__.split('.')[:2])) >= (1, 15)"` | Python/JAX/NumPy/SciPy metadata and resolved-environment contract. |
| V-JAX-SUITE | `uv run --python 3.12 pytest -p no:capture tests/jax --ignore=tests/jax/bench` | Migrated JAX suite under the supported Python floor. |
| V-DOCS | `uv run --extra docs mkdocs build --strict` | Documentation references and published-contract checks. |
| V-PROMO-UNIT | `uv run pytest -p no:capture tests/jax/bench/test_promotion_harness.py tests/jax/bench/test_parallel_benchmarks.py tests/jax/bench/test_rhe_benchmarks.py` | Evidence schema, matching, gate evaluator, command builder, and non-opt-in benchmark units. |
| V-PROMO-LOCAL | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/bench/test_promotion_benchmarks.py --runbench --jax-enforce-promotion-gates --jax-promotion-output /tmp/linear-dag-jax-promotion-local.json --linarg-h5-path /absolute/path/to/representative-lineararg.h5 --linarg-parallel-processes 2 --linarg-benchmark-k 4 20 --rhe-benchmark-num-matvecs 4 20` | Isolated packed/exact/NumPy-Cython local performance, memory, IR, and numerical evidence. |
| V-RUNNER | `bash scripts/run_jax_promotion.sh --repo-root "$PWD" --hdf5-path /absolute/path/to/representative-lineararg.h5 --output-dir /absolute/path/to/evidence-output --platform-label <arm64-cpu\|x86_64-cpu\|forced-two-device-cpu\|gpu> --device-count <N>` | Fresh-cache/reused-cache cross-machine evidence generation. The implementation must preserve these explicit argument semantics. |
| V-DECISION | `uv run pytest -p no:capture tests/jax/bench/test_promotion_harness.py tests/jax/test_coexistence.py tests/cli/test_cli.py` | Evidence validation, decision recomputation, and no leaked public promotion. |
| V-RELEASE | `uv run pytest -p no:capture tests/jax tests/association/test_heritability_jax.py tests/association/test_rhe.py tests/core/test_alignment.py tests/cli/test_cli.py` | Full non-opt-in JAX/RHE/alignment/CLI regression suite. |
| V-RELEASE-2D | `XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest -p no:capture tests/jax/test_packing.py tests/jax/test_packed_products.py tests/jax/test_hijax.py tests/jax/test_transform_composition.py tests/jax/test_grm_operator.py tests/association/test_heritability_jax.py` | Final two-device correctness, transforms, IR, and residency regression. |
| V-RELEASE-X64 | `JAX_ENABLE_X64=1 uv run pytest -p no:capture tests/jax` | Final x64 dtype regression. |
| V-STATIC | `uv run ruff check src tests scripts && uv run ty check src tests && git diff --check` | Formatting, static typing, and whitespace. |

## AC-to-task-to-test traceability

### AC1: Packing preserves state with bounded memory

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-1.1 | AC1.1 | P1-T1, P1-T2, P1-T3 | Parameterized structural/invariant tests in `tests/jax/test_packing.py`; ingress integration in `tests/jax/test_ingress.py`. | V-PACK, V-PACK-2D | One fixed ordered component/schema set is used across one, two, and many source blocks; only component shapes, values, descriptors, and valid lengths change. |
| TR-1.2 | AC1.2 | P1-T2, P1-T3, P6-T1 | Round-trip/oracle tests in `tests/jax/test_packing.py` and HDF5/in-memory integration tests in `tests/jax/test_ingress.py`. | V-PACK, V-PACK-2D | Pack/unpack reproduces every canonical graph field byte-for-byte and restores exact sample and logical variant mappings, including flips, optional fields, nonunique rows, and physical/logical reorderings. |
| TR-1.3 | AC1.3 | P1-T1, P1-T4, P7-T2 | Pure byte-accounting unit tests plus opt-in production benchmark in `tests/jax/test_packing.py`, `tests/jax/bench/test_parallel_benchmarks.py`, and `tests/jax/bench/test_promotion_benchmarks.py`. | V-PACK, V-PACK-PROD, V-PROMO-LOCAL | Descriptor bytes are reported separately and `padded_graph_bytes / canonical_graph_bytes <= 1.25` on the representative production dataset without an override. |
| TR-1.4 | AC1.4 | P1-T3, P1-T4, P7-T2 | Forced-two-device sharding/residency integration and production benchmark tests in packing/ingress/benchmark modules. | V-PACK-2D, V-PACK-PROD, V-PROMO-LOCAL | Every graph leaf resides exclusively on its assigned device; maximum device graph bytes are at most `0.65 * canonical_graph_bytes`. Process RSS is not accepted as the residency oracle. |
| TR-1.5 | AC1.5 | P6-T1, P7-T2 | Instrumented streaming ingress tests in `tests/jax/test_ingress.py` plus promotion memory records. | V-PACK-2D, V-PROMO-LOCAL | HDF5 and in-memory construction never put a full graph on the default device; accounted default-device ingress peak is bounded by its final shard plus one live source block. The downstream real-Zarr gate is recorded, not claimed by group mocks. |
| TR-1.6 | AC1.6 | P1-T1, P1-T2, P6-T1 | Negative boundary/property tests in `tests/jax/test_packing.py` and `tests/jax/test_ingress.py`. | V-PACK, V-PACK-2D | Unsatisfied whole-block padding bounds raise diagnostic `ValueError`; an explicit override permits excess padding but never bypasses structural validation or claims graph subdivision. |

### AC2: Compiled programs treat graph state as explicit data

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-2.1 | AC2.1 | P2-T3, P3-T4, P4-T2 | Recursive jaxpr constant inspection in `tests/jax/test_packed_products.py`, `tests/jax/test_hijax.py`, and `tests/jax/test_transform_composition.py`. | V-PRODUCT-2D, V-HIJAX, V-COMPOSE | Explicit functional calls and transformed calls report zero graph-sized closed-over constants; graph arrays appear as dynamic operands. |
| TR-2.2 | AC2.2 | P2-T3, P4-T2 | Differential structural IR tests at fixed capacities and varying source-block counts. | V-PRODUCT-2D, V-COMPOSE | Packed operand count, recursive equation structure, and StableHLO operation count do not grow with logical source-block count, allowing DCE of genuinely unused fields. |
| TR-2.3 | AC2.3 | P2-T3 | Lowered sharding/StableHLO inspection in `tests/jax/test_packed_products.py`. | V-PRODUCT-2D | Graph operands retain graph-axis/manual shardings; no graph all-gather or graph broadcast occurs. Scalar/mask broadcasts are not falsely rejected. |
| TR-2.4 | AC2.4 | P2-T2, P2-T3, P3-T3 | Safe-helper numerical and lowering tests in product/HiJAX modules. | V-PRODUCT-2D, V-HIJAX | `compile_matmat` and `compile_rmatmat` match functional results and pass the carrier as a dynamic executable argument while satisfying TR-2.1 through TR-2.3. |
| TR-2.5 | AC2.5 | P2-T2, P6-T3 | Docstring/source assertions plus strict documentation build. | V-PRODUCT-2D, V-COEXIST, V-DOCS | Documentation identifies raw bound-method closure capture as unsupported for memory guarantees, avoids tracer-inspection promises, and directs users to explicit-operator calls or safe helpers. |

### AC3: Packed products preserve LinearARG numerics

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-3.1 | AC3.1 | P2-T1, P5-T2 | Differential numerical tests against `tests/jax/oracle.py`, Cython `LinearARG`, and exact-ragged JAX in `tests/jax/test_packed_products.py`. | V-PRODUCT, V-PRODUCT-2D, V-FFI-PACKED | Packed `matmat` matches both references within the established dtype-specific tolerances for pure JAX and available packed CPU FFI. |
| TR-3.2 | AC3.2 | P2-T1, P5-T2 | Differential transpose-product and exact-order tests in `tests/jax/test_packed_products.py`. | V-PRODUCT, V-PRODUCT-2D, V-FFI-PACKED | Packed `rmatmat` matches both references and returns exactly `(n_variants, K)` in logical variant order. |
| TR-3.3 | AC3.3 | P2-T1, P5-T2 | Parameterized contract tests covering rank, flips, compression, and dtype. | V-PRODUCT, V-PRODUCT-2D, V-FFI-ABI, V-RELEASE-X64 | Rank-one/rank-two, flipped, nonunique/compressed, float32, and x64-enabled float64 cases retain current operator semantics. |
| TR-3.4 | AC3.4 | P2-T1, P5-T1 | Negative pre-execution validation and native ABI safety tests. | V-PRODUCT, V-FFI-ABI | Bad ranks/shapes, unsupported dtypes, descriptor spans, graph indices, and non-bijective mappings fail before solve execution; malformed direct FFI inputs return errors rather than crashing. |

### AC4: Multi-device execution preserves graph ownership

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-4.1 | AC4.1 | P2-T2, P5-T2 | One-versus-two-device differential integration tests. | V-PRODUCT, V-PRODUCT-2D, V-FFI-PACKED | Single-device and forced-two-device outputs match the same logical references for both products and supported CPU backends. |
| TR-4.2 | AC4.2 | P2-T2, P2-T3, P5-T2 | Local shard-index/device inspection plus lowered-operand tests. | V-PRODUCT-2D, V-FFI-PACKED | Each `shard_map` body receives only one assigned graph shard and operation-required dense values; graph leaves remain on their original devices after execution. |
| TR-4.3 | AC4.3 | P2-T2, P2-T3, P5-T2 | StableHLO collective/custom-call inspection. | V-PRODUCT-2D, V-FFI-PACKED | Forward uses compatible `psum_scatter` or replicated `psum`; reverse communicates dense samples/results only; no graph collective is present. |
| TR-4.4 | AC4.4 | P2-T2, P5-T2 | Edge-case multi-device fixtures with uneven blocks, empty assignments, and an explicitly overridden skewed fixture. | V-PACK-2D, V-PRODUCT-2D, V-FFI-PACKED | Uneven/empty assignments retain valid descriptors/shardings and numerical parity; the skewed case needs an explicit padding override. |
| TR-4.5 | AC4.5 | P2-T2 | Negative construction/lowering tests in `tests/jax/test_packed_products.py`. | V-PRODUCT-2D | Missing/wrong mesh axes, incompatible output shardings, non-single-host meshes, and invalid reduce-scatter shapes fail with actionable errors before unsafe execution. |

### AC5: Dense operands support composable JAX transformations

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-5.1 | AC5.1 | P3-T3, P5-T2 | Exact linearity/JVP oracle tests in `tests/jax/test_hijax.py` and backend-parameterized product tests. | V-HIJAX, V-FFI-PACKED | JVPs equal the same packed product applied to the dense tangent for `matmat` and `rmatmat`; graph tangent remains zero-only. |
| TR-5.2 | AC5.2 | P3-T3, P5-T2 | Analytical adjoint plus finite-difference VJP tests. | V-HIJAX, V-COMPOSE, V-FFI-PACKED | Dense VJPs bind the companion product, agree with analytical adjoints, and match deterministic finite differences within dtype-specific tolerance. |
| TR-5.3 | AC5.3 | P3-T3, P3-T4, P4-T1, P4-T2, P5-T2 | Transform-unit and nonlinear composition suites in HiJAX, GRM, and transform test modules. | V-HIJAX, V-GRM, V-COMPOSE, V-FFI-PACKED | `jit`, `jit(grad)`, `grad(jit)`, `value_and_grad`, higher order, dense `vmap`, invariant-graph `scan`, and `remat` all produce reference results with the operator explicit. |
| TR-5.4 | AC5.4 | P3-T2, P3-T3, P3-T4, P4-T1, P4-T2, P5-T2 | Symbolic-zero, DCE, residual-inspection, and rematerialization tests. | V-HIJAX, V-GRM, V-COMPOSE, V-FFI-PACKED | Inactive dense tangents work; no graph cotangent is emitted; unused results do not retain graph residual copies or introduce graph constants/collectives. |
| TR-5.5 | AC5.5 | P3-T2, P3-T3, P4-T2 | Negative graph-AD/batching tests in `tests/jax/test_hijax.py` and `tests/jax/test_transform_composition.py`. | V-HIJAX, V-COMPOSE | Nonzero differentiation or mapped axes for topology, graph values, allele metadata, or packing state raise an actionable opaque-graph error. |

### AC6: GRM, RHE, and research losses compose normally

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-6.1 | AC6.1 | P4-T1, P5-T2 | GRM numerical/AD integration against dense, Cython, and exact-ragged references. | V-GRM, V-FFI-PACKED | Packed multi-device GRM products, JVPs, and reverse gradients preserve centered/weighted symmetric behavior for supported backends. |
| TR-6.2 | AC6.2 | P4-T3, P7-T2 | Fixed-seed/fixed-probe RHE differential integration and benchmark parity. | V-RHE, V-RHE-2D, V-PROMO-LOCAL | At `num_matvecs` 4 and 20 where configured, packed JAX estimates match NumPy/Cython within the existing estimator tolerance and preserve reordered-phenotype behavior. |
| TR-6.3 | AC6.3 | P4-T2, P5-T2 | Deterministic nonlinear research-loss transform suite. | V-COMPOSE, V-FFI-PACKED | The specified `tanh` loss runs under outer JIT with JVP, gradient, and a second-order check without a loss-specific handwritten gradient. |
| TR-6.4 | AC6.4 | P4-T3 | Failure-first alignment/pipeline tests in `tests/core/test_alignment.py`, `tests/association/test_rhe.py`, and `tests/association/test_heritability_jax.py`. | V-RHE | Zero overlap, missing columns/IIDs, invalid diploid multiplicity/intercept, and all-missing phenotypes fail before graph products or array conversion, consistently across NumPy and JAX paths. |

### AC7: Backends, ingress, and API compatibility

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-7.1 | AC7.1 | P5-T1, P5-T2, P5-T3, P7-T3 | Shared backend matrix spanning numerical, transform, sharding, and residency suites. | V-FFI-BUILD, V-FFI-ABI, V-FFI-PACKED, V-RUNNER | Pure JAX and available CPU FFI pass the same contracts. GPU evidence uses portable pure JAX. No accelerator-specific backend is advertised; any future backend must join this matrix before exposure. |
| TR-7.2 | AC7.2 | P5-T3 | Representation-aware backend resolution unit/integration tests. | V-BACKEND | Explicit unavailable/non-CPU/incomplete packed FFI requests fail before lowering. `AUTO` silently selects exact FFI only when exact targets exist, packed FFI only when packed targets exist, otherwise pure JAX, and records the resolved backend. |
| TR-7.3 | AC7.3 | P6-T1, P6-T3 | HDF5 schema/reconstruction integration, generic-group labeling assertions, and strict docs tests. | V-PACK, V-PACK-2D, V-DOCS | Existing HDF5 root/block layouts reconstruct identical logical operators without schema changes. Generic group fixtures never count as real Zarr support; downstream `genoio` integration remains a named blocking gate before any Zarr claim. |
| TR-7.4 | AC7.4 | P3-T2, P3-T4, P6-T2, P6-T3, P7-T5 | Public export/signature/annotation/PyTree/source inspection plus docs build. | V-HIJAX, V-COEXIST, V-DOCS, V-RELEASE | No public import, signature, annotation, documentation example, or public PyTree contract exposes HiJAX or the private packed carrier. |
| TR-7.5 | AC7.5 | P6-T2, P6-T3, P7-T4, P7-T5 | Private target-contract matrix plus public coexistence/CLI regressions. | V-COEXIST, V-DECISION, V-RELEASE | The private candidate supplies target single/multi-block constructors, properties, products, and safe helpers; current public exact classes remain constructible and tested. Candidate renaming/export is absent and explicitly deferred. |
| TR-7.6 | AC7.6 | P3-T1, P3-T4, P7-T5 | Dependency resolution, metadata/source inspection, build, and supported-runtime suites. | V-DEPS, V-JAX-SUITE, V-FFI-BUILD, V-STATIC | Metadata, lockfile, classifiers, Hatch matrix, and Ruff target consistently require Python `>=3.12,<3.15`, JAX/JAXlib 0.11.0, NumPy `>=2.1`, and SciPy `>=1.15`; no test widens the JAX pin. |

### AC8: Promotion is evidence-gated across platforms

| Test ID | AC case | Owning task IDs | Planned automated test type and path | Commands | Required behavior |
| --- | --- | --- | --- | --- | --- |
| TR-8.1 | AC8.1 | P7-T3, P7-T4 | Portable runner command-generation tests plus schema-validated evidence from independent arm64 CPU, x86_64 CPU, forced-two-device CPU, and available GPU runs. | V-PROMO-UNIT, V-RUNNER, V-DECISION | Correctness, transform, IR, and residency suites pass in each required environment before a `promote` result. Missing hardware evidence is `missing` and forces coexistence rather than being skipped. |
| TR-8.2 | AC8.2 | P7-T1, P7-T2, P7-T3 | Evidence-schema units and isolated opt-in product/RHE benchmark subprocesses. | V-PROMO-UNIT, V-PROMO-LOCAL, V-RUNNER | Records separate construction, lowering, compilation, first execution, and warm execution where supported, with null reasons otherwise; communication, padding, peak/final residency, constants, operands, and IR metrics are present. |
| TR-8.3 | AC8.3 | P7-T1, P7-T2, P7-T4 | Exact-match gate evaluator and isolated production benchmark evidence. | V-PROMO-UNIT, V-PROMO-LOCAL, V-DECISION | For both products and `K` 4/20, every promotable packed warm median is at most 1.20 times the isolated retained exact-ragged median from the same candidate commit, dataset, backend, dtype, device count, and cache policy. |
| TR-8.4 | AC8.4 | P7-T1, P7-T2, P7-T4 | Phase-aware product/RHE ratio matching and report cross-checks. | V-PROMO-UNIT, V-PROMO-LOCAL, V-DECISION | NumPy/Cython product and RHE comparisons remain in evidence, but cold/construction/first-execution measurements are never used as warm denominators or labeled compilation when inseparable. |
| TR-8.5 | AC8.5 | P7-T1, P7-T3, P7-T4, P7-T5 | Gate-aggregation units, decision-report recomputation, public-isolation regression, and documentation checks. | V-PROMO-UNIT, V-DECISION, V-COEXIST, V-DOCS | Any failed or missing gate yields `continue_coexistence` or `reject`, names every blocker, keeps packed private, retains exact public classes/CLI routing, and does not weaken the quantitative threshold. A full pass records `promote` but still requires a separate public-migration plan. |

## Test-quality requirements

- Use behavior and algebraic oracles rather than asserting private call counts, except where call interception is the observable safety boundary (for example, proving failure occurs before an FFI call or graph product).
- Packing tests must exercise round-trip, bijection, inert-padding, complete-assignment, and byte-accounting properties over constrained deterministic synthetic blocks. Hypothesis is optional; do not add it solely for nominal examples. If used, constrain generated CSC structures at the strategy boundary and retain explicit empty/single/skewed examples.
- Numerical tests must compare against independent Cython/dense/exact-ragged references. Do not reimplement the packed algorithm as the expected-value calculation.
- AD tests must cover primals, JVPs, VJPs, batching, JIT composition, and at least one higher-order check. Finite differences supplement exact linear/adjoint identities rather than replacing them.
- Multi-device and benchmark tests must run in isolated processes when device count, allocator state, persistent cache, or graph residency could contaminate another case. Do not use arbitrary sleeps; synchronize actual JAX results.
- All stochastic RHE checks use explicit fixed seeds/probe vectors and report tolerance. Performance inputs are fixed and keyed by dataset fingerprint.
- Negative boundary tests must prove rejection before traced numerical execution. Traced kernels use JAX/FFI-compatible error channels rather than Python exceptions.

## Human and external verification plan

Most correctness judgments are automated. The following steps are genuinely external or governance-only and cannot be established by a unit test in one checkout.

| Human ID | Related ACs | Why human/external action is required | Executable steps | Required result |
| --- | --- | --- | --- | --- |
| H-1 | AC8.1-AC8.5 | The test runner cannot provision independent CPU architectures or an available GPU, attest the physical machine, or transfer the representative dataset. | On clean checkouts of the same candidate commit, run V-RUNNER separately for `arm64-cpu`, `x86_64-cpu`, `forced-two-device-cpu`, and `gpu`; use the same dataset fingerprint; copy only normalized JSON/log/checksum artifacts into the plan evidence directory; run V-DECISION on the aggregate. | Every artifact validates and the evaluator either proves all gates pass or records missing/failed evidence and selects coexistence/rejection. No platform is silently marked not applicable. |
| H-2 | AC7.5, AC8.5 | Public API migration is an approval/governance decision intentionally outside this increment, even if automated evidence says `promote`. | Review `promotion-decision.md` against its JSON IDs and recomputed ratios; if the result is `promote`, open and approve a separate design/implementation plan covering class rename, exports, exact-block compatibility, CLI routing, documentation, and deprecation behavior; do not edit the public facade in this phase. | Current increment retains the public facade. Any future migration has explicit maintainer approval and its own acceptance criteria. |

## Deferred integration gate

Real Zarr verification is not human-only, but it is deferred to a later `genoio` integration because this branch has no durable Zarr reader. Before any Zarr-support claim, that later work must add automated reconstruction, schema-parity, peak-residency, sharding, and transform tests against the real `genoio` reader. Duck-typed group fixtures satisfy only generic reader-unit coverage and cannot close AC7.3's downstream gate.

## Preconditions for evidence runs

- Clean candidate checkout with the exact commit recorded in every result.
- Python `>=3.12,<3.15`, JAX/JAXlib 0.11.0, and the locked dependency set.
- Representative production HDF5 file supplied by absolute path; identical SHA-256/size/block-count/logical-shape fingerprint on every machine.
- CPU evidence that claims FFI has a successful V-FFI-BUILD and records exact/packed target availability plus BLAS/native-tuning configuration.
- Explicit device count and XLA flags. GPU evidence uses pure JAX because this branch has no accelerator-specific backend.
- Runner-owned output/cache directories with enough space for normalized evidence and optional external XLA dumps.

## Coverage summary and unresolved gaps

- Atomic AC cases: 40.
- AC cases mapped to implementation tasks: 40.
- AC cases mapped to planned automated evidence: 40.
- Human/external verification items: 2.
- Plan-level traceability gaps: none.
- Execution evidence: not collected; this is a pre-implementation contract.
- Expected execution-time blockers: missing representative dataset or any required platform produces named missing evidence and forces coexistence under AC8.5.
- Deferred by approved scope: real `genoio` Zarr integration and all public packed-class promotion changes.
