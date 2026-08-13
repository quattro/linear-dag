# Model Symbol Table

## Context
- Plan slug: `jax-packed-sharded-lineararg`
- Generated date: `2026-08-13`

| Symbol | Meaning | Domain/Support | Shape/Type | Defined In Source | Notes |
| --- | --- | --- | --- | --- | --- |
| $X$ | Logical genotype matrix represented by a LinearARG. | Real-valued linear operator with sample rows and variant columns. | $(N, M)$ | `src/linear_dag/core/lineararg.py` | Never materialized densely by this design. |
| $N$ | Number of samples. | Positive integer. | scalar static metadata | `LinearARG.shape[0]` | Shared across all source blocks. |
| $M$ | Total number of logical variants. | Positive integer. | scalar static metadata | `LinearARG.shape[1]` / wrapper offsets | Public variant results use this exact length. |
| $D$ | Number of devices on the graph mesh axis. | Positive integer; single host in scope. | scalar static metadata | packed constructor mesh | Shapes and compilation specialize to $D$. |
| $K$ | Number of dense right-hand sides or traits. | Positive integer. | scalar shape dimension | call-time dense operand | Rank-one inputs are treated as $K=1$. |
| $V_d$ | Ordered logical variant indices owned by graph shard $d$. | Disjoint sets whose union is $\{0,\ldots,M-1\}$. | variable logical length; padded physically | packing descriptors | Ownership may reorder variants physically. |
| $G_d$ | Packed graph state assigned to device $d$. | Validated CSC-derived arrays plus descriptors and masks. | fixed local capacities per field | planned packing component | A device stores only $G_d$, not all $G$. |
| $C_f$ | Per-device physical capacity for packed field $f$. | At least the maximum valid assigned length for field $f$. | scalar static metadata | packing policy | Aggregate padding is $D C_f-\sum_d L_{d,f}$. |
| $L_{d,f}$ | Valid unpadded length of field $f$ on device $d$. | Integer in $[0,C_f]$. | descriptor scalar | packed valid-length arrays | Masks padding from local computation. |
| $W$ | Dense variant-space operand to `matmat`. | Floating-point JAX array. | $(M,K)$ | public operation contract | May communicate; is not graph state. |
| $Y$ | Dense sample-space operand to `rmatmat` or output of `matmat`. | Floating-point JAX array. | $(N,K)$ | public operation contract | May be replicated or sample-sharded. |
| $\dot W,\dot Y$ | Forward-mode tangents of dense operands. | Same dtype/support as primal operands. | $(M,K)$ or $(N,K)$ | HiJAX JVP rules | Graph tangent is symbolic zero. |
| $\bar W,\bar Y$ | Reverse-mode cotangents of dense operands/results. | Same dtype/support as primal operands. | $(M,K)$ or $(N,K)$ | HiJAX VJP/transpose rules | No graph cotangent is constructed. |
| $P$ | Physical-to-logical variant mapping/permutation. | Bijection over valid packed variant rows. | packed integer descriptor | packing component | Used to gather inputs and restore exact output order. |

## Checks
- [x] No undefined symbols.
- [x] No conflicting symbol reuse.
- [x] Support/domain constraints are explicit.
