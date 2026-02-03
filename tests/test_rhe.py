from pathlib import Path

import numpy as np
import polars as pl
import pytest

from linear_dag.association.heritability import randomized_haseman_elston
from linear_dag.association.simulation import simulate_phenotype
from linear_dag.association.util import residualize_phenotypes
from linear_dag.core.operators import get_diploid_operator
from linear_dag.core.parallel_processing import GRMOperator, ParallelOperator

TEST_DATA_DIR = Path(__file__).parent / "testdata"
SEED = 0  # todo: should we parameterize this somehow?


def _haseman_elston(grm, data, pheno_cols, covar_cols):
    phenotypes = data.select(pheno_cols).collect().to_numpy()
    covariates = data.select(covar_cols).collect().to_numpy()
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)

    yresid = residualize_phenotypes(phenotypes, covariates, is_missing)
    yresid /= np.sqrt(np.sum(yresid**2, axis=0) / num_nonmissing)  # ||y_resid||^2 == num_nonmissing

    C = np.sum(grm.dot(yresid) * yresid, axis=0)
    E = np.sum(yresid * yresid, axis=0)
    N = grm.shape[0]
    grm_trace = np.trace(grm)
    grm_sq_trace = np.trace(grm @ grm)

    # construct linear equations to solve
    LHS = np.array([[grm_sq_trace, grm_trace], [grm_trace, N]])
    RHS = np.vstack([C, E])
    # print(f"HE | LHS = {LHS} | RHS = {RHS}")
    solution = np.linalg.solve(LHS, RHS)

    s2g = solution[0, :] * grm_trace
    s2e = solution[1, :] * N
    heritability = s2g / (s2g + s2e)

    return heritability


@pytest.mark.parametrize("h2g", (0.05, 0.5, 0.9))
@pytest.mark.parametrize("estimator", ("hutchinson", "hutch++", "xnystrace"))
@pytest.mark.parametrize("sampler", ("normal", "sphere", "rademacher"))
@pytest.mark.parametrize("num_matvecs", (25, 10, 5))
def test_rhe(h2g, estimator, sampler, num_matvecs):
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    rng = np.random.default_rng(SEED)

    # Load as genotype operator to simulate data
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as genotypes:
        n, m = genotypes.shape
        fraction_causal = 1
        dgenotypes = get_diploid_operator(genotypes.normalized) / np.sqrt(2.0)

        # Simulate phenotype
        y, beta = simulate_phenotype(
            dgenotypes,
            heritability=h2g,
            fraction_causal=fraction_causal,
            return_beta=True,
            num_traits=2,
            seed=rng,
        )

        # Create phenotype DataFrame with covariates
        df_pheno = pl.DataFrame(
            {
                "iid": genotypes.iids.unique(maintain_order=True).to_list(),
                "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
                "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
            }
        ).with_columns(
            intercept=pl.lit(1),
            covar1=rng.random(size=(n // 2,)),
        )

        # Extract genotype matrix to compute GRM the ol' fashioned way, for comparison
        G = dgenotypes @ np.eye(dgenotypes.shape[1])
        af = G.mean(axis=0) / 2.0
        G_std = (G - 2 * af) / np.sqrt(2 * af * (1.0 - af))
        G_std[np.isnan(G_std)] = 0
        grm_direct = G_std @ G_std.T / G.shape[1]

    pheno_cols = ["phenotype1", "phenotype2"]
    covar_cols = ["intercept", "covar1"]
    # Re-load using the genotype operator
    with GRMOperator.from_hdf5(hdf5_path, num_processes=2, alpha=-1.0) as grm:
        # Our randomized estimator
        observed = randomized_haseman_elston(
            grm,
            df_pheno.lazy(),
            pheno_cols,
            covar_cols,
            num_matvecs,
            estimator,
            sampler,
            seed=rng,
        )
        # Compare with exact computation using GRM matrix
        observed = observed.get_column("h2g").to_numpy()
        expected = _haseman_elston(grm_direct, df_pheno.lazy(), pheno_cols, covar_cols)
        print(f"h2g = {h2g} | estimator = {estimator} | sampler = {sampler} | num_matvecs = {num_matvecs}")
        print(f"\tobserved = {observed} | expected = {expected}")

    return
