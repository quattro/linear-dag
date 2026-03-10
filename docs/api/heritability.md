# Heritability estimation

`linear-dag` implements randomized Haseman-Elston regression for SNP heritability estimation.

At a high level, the estimator:

1. Aligns phenotype/covariate rows with GRM identifiers.
2. Residualizes phenotypes on covariates.
3. Uses randomized probes to estimate trace terms such as $\mathrm{tr}(K)$ and $\mathrm{tr}(K^2)$.
4. Solves moment equations for genetic and environmental variance components.
5. Reports `s2g`, `s2e`, and `h2g` with standard errors for each phenotype.

!!! note

    The first covariate must be an intercept (all ones), and `num_matvecs`
    must satisfy estimator-specific minimum values.


::: linear_dag.association.heritability.randomized_haseman_elston
    options:
        show_bases: true
