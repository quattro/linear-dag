# Solver Feasibility Matrix

## Context
- Plan slug: `jax-packed-sharded-lineararg`
- Generated date: `2026-08-13`

| Candidate | Problem Fit | AD Compatibility | Memory/Sharding Fit | Status/Error Mapping | Feasible | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Existing pure-JAX triangular solve | Direct LinearARG local product. | Composable once wrapped by explicit JVP/VJP rules. | Accepts local arrays; suitable as portable packed-shard kernel. | Existing backend exceptions/fallback policy. | yes | Baseline local kernel and numerical oracle. |
| Existing CPU FFI triangular solve | Direct LinearARG local product on CPU. | Requires companion primitive rules; graph remains nondifferentiable. | Feasible if local packed views avoid unbounded copies. | Explicit unavailable backend fails or documented AUTO fallback applies. | conditional | ABI/layout adaptation measured after pure-JAX prototype. |
| Existing accelerator kernel | Direct LinearARG local product on supported accelerator. | Requires the same companion rules as CPU/pure JAX. | Natural local-shard execution if descriptor layout is supported. | Explicit unavailable backend fails fast. | conditional | Validated only on actual supported hardware. |
| Lineax | Generic linear operator/solver composition. | JAX-native. | Does not define packed ragged graph storage or local triangular kernel. | Lineax solver result conventions. | no for core | Possible future adapter over the completed project-owned operations. |
| Optimistix | Root finding/least-squares/minimization. | JAX-native. | Does not address operator representation or graph sharding. | Optimistix result conventions. | no | Wrong problem category. |
| HiJAX + `shard_map` | High-level operation identity, transform rules, and per-device expansion. | Supports explicit JVP/VJP, batching, remat, and zero tangents. | Feasible only when lower value is the packed explicitly sharded array set. | Private compatibility boundary converts unsupported cases to actionable errors. | yes | Preferred orchestration/transform layer; not the storage solution. |
| `custom_partitioning` | Compiler partitioning of a custom operation. | Orthogonal rules still required. | Could express specialized partitioning, but adds cache/callback risk. | Lowering-time failures are harder to make actionable. | deferred | Adopt only for a measured `shard_map` limitation. |
| Opaque external graph handle + FFI | Avoid graph arrays as JAX operands. | Every transform needs explicit callback/FFI support. | Avoids buffer operands but adds registry/lifetime/export/multi-process hazards. | Handle expiration and process-local errors. | no | Rejected as the primary public representation. |
| Exact-ragged host orchestration (`jax-focused`) | Correct current multi-device products. | Reverse mode works outside outer JIT. | Best current no-padding residency, but not outer-JIT compatible. | Existing documented contract. | fallback | Retained as memory/performance oracle through migration. |

## Decision
- Preferred solver path: packed globally sharded graph state, `shard_map` global execution, existing local triangular kernels, and a private HiJAX transformation layer.
- Reason: this is the only evaluated path that addresses physical graph ownership, fixed compiled operands, full dense-operand autodiff, and public JAX composition without changing the scientific operator.
- Benchmark or validation requirement before implementation: the pure-JAX packing spike must show no graph-sized constants, a fixed component count independent of source block count, bounded aggregate padding, one graph shard per device, correct forward/reverse products, and StableHLO collectives consistent with the design before HiJAX or accelerated backends are added.
