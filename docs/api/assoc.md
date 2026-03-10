# Association testing

`linear-dag` provides covariate-adjusted association scans that operate on `LinearOperator` genotype backends.

At a high level, association testing consists of:

1. Aligning phenotype and covariate rows to genotype identifiers.
2. Residualizing phenotypes on covariates.
3. Computing per-variant effect sizes and standard errors in per-trait columns.

!!! note

    The first covariate must be an intercept (all ones). Non-HWE paths require
    genotype inputs with explicit individual-node and heterozygote support.


::: linear_dag.run_gwas
    options:
        show_bases: true

---

::: linear_dag.get_gwas_beta_se
options:
show_bases: true

---
