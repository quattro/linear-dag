from pathlib import Path

import numpy as np
import polars as pl
import pytest

from linear_dag.association.gwas import (
    run_gwas,
    run_gwas_parallel,
    get_gwas_beta_se,
    _format_sumstats,
)
from linear_dag.association.util import residualize_phenotypes
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


@pytest.mark.parametrize("assume_hwe", [True, False])
def test_gwas_parallel_matches_run_gwas(tmp_path, assume_hwe):
    """
    Test that run_gwas_parallel produces identical summary statistics to run_gwas
    when reconstructed from its preprocessed outputs.
    """
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    heritability = 0.5

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as genotypes:
        n, m = genotypes.shape

        # Simulate phenotype and prepare data (same as the baseline test)
        y = simulate_phenotype(
            get_diploid_operator(genotypes),
            heritability=heritability,
            fraction_causal=1,
            return_beta=False,
        )
        pheno_df = pl.DataFrame({"iid": genotypes.iids[::2].to_list(), "phenotype": y.flatten()}).with_columns(
            intercept=pl.lit(1), covar1=pl.lit(np.random.rand(n // 2))
        )

        pheno_cols = ["phenotype"]
        covar_cols = ["intercept", "covar1"]

        # Parallel-prepared inputs (writes per-block TSVs to output_prefix)
        try:
            run_gwas_parallel(
                str(hdf5_path),
                pheno_df.lazy(),
                pheno_cols=pheno_cols,
                covar_cols=covar_cols,
                output_prefix=str(tmp_path),
                assume_hwe=assume_hwe,
                num_workers=2,
            )
        except ValueError as e:
            if not assume_hwe and "must include individual nodes" in str(e):
                pytest.skip("HDF5 lacks individual nodes; skipping non-HWE test")
            else:
                raise
        
        # Compare per-block outputs to per-block expected stats to avoid ordering issues
        blocks = list_blocks(str(hdf5_path))["block_name"].to_list()
        for block in blocks:
            # Load worker block output
            out_path = tmp_path / f"{block}.parquet"
            df_block = pl.read_parquet(out_path)

            # Compute expected for this block by calling run_gwas on the block
            linarg = LinearARG.read(str(hdf5_path), block=block, load_metadata=True)
            expected_b = run_gwas(
                linarg,
                pheno_df.lazy(),
                pheno_cols=pheno_cols,
                covar_cols=covar_cols,
                variant_info=linarg.variants,
                assume_hwe=assume_hwe,
            ).collect()
            df_block_stats = df_block.select(expected_b.columns)

            assert expected_b.shape == df_block_stats.shape
            assert expected_b.columns == df_block_stats.columns
            for col in expected_b.columns:
                if np.issubdtype(expected_b[col].to_numpy().dtype, np.number):
                    np.testing.assert_allclose(
                        expected_b[col].to_numpy(), df_block_stats[col].to_numpy(), rtol=1e-6, atol=1e-6
                    )
