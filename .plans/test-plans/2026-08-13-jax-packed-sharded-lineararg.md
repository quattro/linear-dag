# Human Test Plan: JAX Packed-Sharded LinearARG

## Preconditions

- Use a clean checkout of candidate commit `0195c3dc418abbce1b636d29da26e50730222ea4` or its reviewed descendant.
- Do not stage the representative HDF5 file or machine-local cache/output directories.
- Use Python 3.12–3.14 with the committed lockfile, JAX/JAXlib 0.11.0, NumPy 2.1+, and SciPy 1.15+.
- Supply one representative production HDF5 file by absolute path. Record its SHA-256, size, block count, sample count, and variant count.
- For CPU FFI evidence, build with `LINEAR_DAG_REQUIRE_FFI_CPU=1`.
- Provision independent arm64 CPU, x86_64 CPU, forced-two-device CPU, and GPU environments. GPU uses pure JAX.
- Create a dedicated output directory outside the repository for every evidence run.

## Phase Checks

| Step | Action | Expected Result |
|---|---|---|
| H1 | Confirm `git rev-parse HEAD`, `git status --short`, Python/JAX versions, platform architecture, and visible devices before each run. | Every artifact identifies one clean candidate commit and the intended physical platform. Untracked benchmark data is not committed. |
| H2 | Fingerprint the representative HDF5 file on every machine. | SHA-256, byte size, block count, and logical shape match exactly across machines. |
| H3 | On forced-two-device CPU, run the production packing/product workflow and inspect the generated evidence metrics. | Padding ratio is separate from descriptor bytes; per-device graph residency and dense communication are present; graph constants are zero; process RSS is not substituted for device residency. |
| H4 | Inspect packed `matmat` and `rmatmat` results against retained exact-ragged and NumPy/Cython outputs for K=4 and K=20. | Shapes and logical variant order match; numerical status passes; all rows use the same dataset and workload identity. |
| H5 | Inspect construction, lowering, compilation, first-execution, and warm-execution rows. | Supported phases are separate. Unsupported phases are null with reasons; first-call time is not labeled compilation. |
| H6 | Inspect the public Python facade and CLI help after the run. | Public `JaxLinearARG` and `JaxParallelOperator` remain exact/exact-ragged; no private packed or HiJAX type is exported; `--jax-backend` retains its existing route. |
| H7 | Review HDF5 reconstruction and documentation claims. | Existing HDF5 schema is unchanged. Generic group fixtures are not described as real Zarr support; the downstream `genoio` gate remains explicit. |
| H8 | Run `scripts/run_jax_promotion.sh` separately for arm64 CPU, x86_64 CPU, forced-two-device CPU, and GPU, collecting fresh and reused cache states. | Each run emits schema-valid JSON, redacted logs, command results, and checksums. Missing hardware is recorded as missing rather than skipped. |
| H9 | Recompute the aggregate decision and compare it with `promotion-decision.md`. | JSON evidence IDs, ratios, blockers, and Markdown agree. Any failed or missing gate yields `continue_coexistence` or `reject`. |
| H10 | If a future aggregate says `promote`, perform a maintainer governance review before changing code. | No facade, export, class-name, constructor, documentation, or CLI migration occurs without a separate approved implementation plan. |

## End-to-End Scenarios

| Scenario | Steps | Expected Result |
|---|---|---|
| Production forced-two-device validation | Run the portable runner with the representative HDF5 file, `--platform-label forced-two-device-cpu`, and `--device-count 2`; inspect fresh and reused artifacts. | Correctness, transform, IR, numerical, padding, residency, and communication outcomes are persisted with evidence IDs. Both cache states use the same commit and dataset fingerprint. |
| Cross-platform completeness | Repeat the runner on arm64 CPU, x86_64 CPU, and GPU; aggregate all artifacts. | The evaluator requires every platform/cache/workload key. Absent x86_64 or GPU evidence remains an explicit blocker. |
| Performance failure handling | Supply valid evidence containing a packed/exact warm ratio above 1.20. | The report names the exact packed and baseline evidence IDs, preserves the threshold, chooses coexistence/rejection, and leaves packed private. |
| Missing-attestation handling | Remove one validation, structural, platform, or workload attestation and recompute. | The gate becomes `missing`; it is not inferred from narrative logs or marked not applicable; the decision is not `promote`. |
| HDF5-to-research workflow | Construct packed and exact operators from the same HDF5 source, run products, GRM, and fixed-probe RHE, then inspect public CLI routing. | Numerical results agree within tolerances, alignment failures remain fatal before products, and the CLI continues using exact-ragged. |
| Future full-pass governance | If all required machine evidence and quantitative gates eventually pass, recompute and review the report. | `promote` authorizes planning only. Public migration begins only through a separate approved plan. |

## Traceability

| AC ID | Automated Evidence | Human Step |
|---|---|---|
| AC1.1–AC1.2 | Packing schema and lossless reconstruction tests | H2, H7 |
| AC1.3–AC1.5 | Byte accounting, placement, staging, production-gate, and evidence-schema tests | H3, forced-two-device scenario |
| AC1.6 | Padding rejection, diagnostics, and override-with-validation tests | H3 |
| AC2.1–AC2.4 | Recursive jaxpr, StableHLO, sharding, and safe-helper tests | H3–H5 |
| AC2.5 | Docstring and documentation source tests | H6 |
| AC3.1–AC3.4 | Product oracle, dtype/rank, logical-order, validation, and FFI safety tests | H4 |
| AC4.1–AC4.5 | Forced-two-device ownership, collective, empty/skewed, and mesh-failure tests | H3, forced-two-device scenario |
| AC5.1–AC5.5 | HiJAX JVP/VJP, batching, symbolic-zero, composition, and opaque-graph rejection tests | H4 |
| AC6.1–AC6.4 | GRM, nonlinear loss, fixed-probe RHE, and fail-before-product alignment tests | HDF5-to-research scenario |
| AC7.1–AC7.2 | Pure-JAX/FFI backend matrix and resolution tests | H1, H8 |
| AC7.3 | HDF5 reconstruction and documentation-label tests | H7 |
| AC7.4–AC7.6 | Public isolation, coexistence, CLI, metadata, lockfile, and docs tests | H1, H6 |
| AC8.1–AC8.2 | Runner/platform/cache/schema/phase tests | H5, H8 |
| AC8.3–AC8.4 | Exact-match ratio and phase-matching tests | H4, H5, H9 |
| AC8.5 | Failure/missing aggregation and committed-decision recomputation | H6, H9, H10 |

## Current Decision

The current decision is `continue_coexistence`. Historical packed warm-product ratios exceed the 1.20 threshold, x86_64/GPU evidence is missing, and final-candidate benchmark and validation attestations are absent. Packed and HiJAX implementations therefore remain private; public exact-ragged operators and CLI routing remain unchanged.
