from pathlib import Path

import numpy as np
import polars as pl

from linear_dag.association.gwas import run_gwas
from linear_dag.association.simulation import simulate_phenotype
from linear_dag.core.operators import get_diploid_operator
from linear_dag.core.parallel_processing import ParallelOperator

TEST_DATA_DIR = Path(__file__).parent / "testdata"


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
        )

        # Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame({"iid": genotypes.iids.unique().to_list(), "phenotype": y.flatten()}).with_columns(
            intercept=pl.lit(1), covar1=pl.lit(np.random.rand(n // 2))
        )

        # 3. GWAS
        pheno_col = "phenotype"
        covar_cols = ["intercept", "covar1"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df.lazy(),
            pheno_cols=[pheno_col],
            covar_cols=covar_cols,
        ).collect()

        # 4. Assertions
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert "phenotype_LOG10P" in gwas_results.columns
