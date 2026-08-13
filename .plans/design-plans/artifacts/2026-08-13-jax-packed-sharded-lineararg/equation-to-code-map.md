# Equation To Code Map

## Context
- Plan slug: `jax-packed-sharded-lineararg`
- Generated date: `2026-08-13`

| Equation ID | Equation | Intended Computation | Target Component | Test Target | Status |
| --- | --- | --- | --- | --- | --- |
| EQ-1 | $Y=XW$ | Public forward genotype product with exact sample ordering. | project-owned `matmat` primitive and bound facade | packed-vs-Cython and packed-vs-exact-ragged forward parity | pending |
| EQ-2 | $U=X^TY$ | Public transpose product with exact logical variant ordering. | project-owned `rmatmat` primitive and reverse unpacking | packed-vs-Cython and packed-vs-exact-ragged reverse parity | pending |
| EQ-3 | $XW=\sum_{d=0}^{D-1}X_{:,V_d}W_{V_d,:}$ | Device-local forward contributions and explicit sample-space reduction. | `shard_map` forward expansion | StableHLO collective and multi-device parity | pending |
| EQ-4 | $X^TY=P^{-1}[X_{:,V_0}^TY;\ldots;X_{:,V_{D-1}}^TY]$ | Device-local transpose contributions, unpadding, and logical-order restoration. | `shard_map` reverse expansion | permutation bijection and multi-device reverse parity | pending |
| EQ-5 | $D(XW)[\dot W]=X\dot W$ | Forward-mode rule for `matmat` with fixed graph state. | private HiJAX JVP/linearization rule | JVP analytical and finite-difference comparison | pending |
| EQ-6 | $(D(XW))^*[\bar Y]=X^T\bar Y$ | Reverse-mode rule for `matmat`. | private HiJAX VJP/transpose rule | VJP/grad adjoint comparison | pending |
| EQ-7 | $D(X^TY)[\dot Y]=X^T\dot Y$ | Forward-mode rule for `rmatmat`. | private HiJAX JVP/linearization rule | reverse-operation JVP comparison | pending |
| EQ-8 | $(D(X^TY))^*[\bar W]=X\bar W$ | Reverse-mode rule for `rmatmat`. | private HiJAX VJP/transpose rule | reverse-operation VJP comparison | pending |
| EQ-9 | $Kz=c^{-1}XD_\alpha X^Tz$ | GRM action composed from public forward/reverse products and fixed weighting. | `src/linear_dag/core/jaxlinarg/grm.py` | packed GRM/RHE parity and gradient tests | pending |
| EQ-10 | $\rho_f=(D C_f)/(\sum_d L_{d,f})$ | Per-field aggregate padding ratio. | packing diagnostics | padding-bound and rejection tests | pending |

## Checks
- [x] Objective sign and optimization direction are not applicable; this is a linear-operator representation.
- [x] Product and derivative rules map to concrete component boundaries.
- [x] Every mapped equation has a corresponding test target.
