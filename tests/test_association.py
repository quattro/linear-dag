from pathlib import Path

import numpy as np
import polars as pl
import pytest

from linear_dag.association.gwas import (
    run_gwas,
    get_gwas_beta_se,
    _format_sumstats,
)
from linear_dag.association.util import impute_missing_with_mean, residualize_phenotypes
from linear_dag.association.simulation import simulate_phenotype
from linear_dag.core.operators import get_diploid_operator, get_inner_merge_operators
from linear_dag.core.lineararg import list_blocks, LinearARG
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
            num_traits=2,
        )

        # Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame({"iid": genotypes.iids.unique().to_list(), 
            "phenotype1": y[:, 0].flatten(), "phenotype2": y[:, 1].flatten()}).with_columns(
            intercept=pl.lit(1), 
            covar1=np.random.rand(n // 2),
            covar2=y[:, 0] + np.random.rand(n // 2),
        )

        # 3. GWAS
        pheno_cols = ["phenotype1", "phenotype2"]
        covar_cols = ["intercept", "covar1", "covar2"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df.lazy(),
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
            in_place_op=True,
        ).collect()

        # 4. Assertions
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        assert gwas_results.select("A1FREQ").dtypes[0] == pl.Float32

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

        # 6. Assertions
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        assert gwas_results.select("A1_CARRIER_FREQ").dtypes[0] == pl.Float32
        assert gwas_results.select((pl.col("A1_CARRIER_FREQ") <= 2*pl.col("A1FREQ")).all()).to_series()[0]


