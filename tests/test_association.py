from pathlib import Path

import numpy as np
import polars as pl
import pytest

from linear_dag.association.gwas import (
    run_gwas,
    simple_gwas,
    get_gwas_beta_se,
    _format_sumstats,
)
from linear_dag.association.util import impute_missing_with_mean, residualize_phenotypes
from linear_dag.association.simulation import simulate_phenotype
from linear_dag.core.operators import get_diploid_operator, get_inner_merge_operators
from linear_dag.core.lineararg import list_blocks, LinearARG
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag import read_vcf

TEST_DATA_DIR = Path(__file__).parent / "testdata"
    

np.random.seed(0)

def test_simulation_and_gwas():
    """
    Test that run_gwas can recover a simulated causal effect.
    """
    # 1. Setup
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    heritability = 0.5

    # 2. Simulation
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as genotypes:
        n, m = genotypes.shape
        fraction_causal = 1

        y, beta = simulate_phenotype(
            get_diploid_operator(genotypes),
            heritability=heritability,
            fraction_causal=fraction_causal,
            return_beta=True,
            num_traits=2,
        )

        # y[::3, 0] = np.nan

        # Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame({
            "iid": genotypes.iids.unique(maintain_order=True).to_list(),
            "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
            "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
        }).with_columns(
            intercept=pl.lit(1), 
            covar1=np.random.rand(n // 2),
            covar2=y[:, 0] + np.random.rand(n // 2),
        )
        pheno_df_dropNA = pheno_df.filter(pl.col("phenotype1").is_not_null())

        # 3. GWAS
        pheno_cols = ["phenotype1", "phenotype2"]
        covar_cols = ["intercept"]#, "covar1", "covar2"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df_dropNA.lazy(),
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
            in_place_op=True,
        ).collect()

        # 4. Assertions
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        beta = gwas_results.select("phenotype1_BETA").to_numpy().copy()
        se = gwas_results.select("phenotype1_SE").to_numpy().copy()

        ident = np.eye(n)
        gt_mat = ident @ genotypes
        left_op, _ = get_inner_merge_operators(
            pheno_df.select("iid").cast(pl.Utf8).to_series(), genotypes.iids
        )

        beta_simple, se_simple = simple_gwas(gt_mat, left_op.T @ pheno_df.select("phenotype1").to_numpy(),
                                        left_op.T @ pheno_df.select(covar_cols).to_numpy())

        assert np.allclose(beta, beta_simple, atol=1e-6)

        print(f"SE: {se[:5]}")
        print(f"SE simple: {se_simple[:5]}")
        assert np.allclose(se, se_simple, atol=1e-6)

        # 5. GWAS not assuming HWE
        pheno_cols = ["phenotype1", "phenotype2"]
        covar_cols = ["intercept", "covar1", "covar2"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df.lazy(),
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
            in_place_op=True,
            assume_hwe=False,
        ).collect()
        beta = gwas_results.select("phenotype1_BETA").to_numpy().copy()

        ident = np.eye(n//2)
        gt_mat = ident @ get_diploid_operator(genotypes)
        beta_simple, se_simple = simple_gwas(gt_mat, pheno_df.select("phenotype1").to_numpy(),
                                        pheno_df.select(covar_cols).to_numpy())

        num_carriers = np.sum(gt_mat>=1, axis=0)
        assert np.all(num_carriers.ravel() == genotypes.number_of_carriers().ravel())

        # 6. Assertions
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32

        diff = (beta-beta_simple)**2
        print(f"Max diff: {np.max(diff)}")
        print(f"Median diff: {np.median(diff)}")
        AF = genotypes.allele_frequencies.ravel()
        print(f"AF where diff > .1: {AF[diff.ravel() > .1]}")
        assert np.allclose(beta, beta_simple, atol=1e-6)


if __name__ == "__main__":
    test_simulation_and_gwas()