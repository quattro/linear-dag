import numpy as np
import polars as pl
import pytest

from linear_dag.association.gwas import get_gwas_beta_se, run_gwas, simple_gwas
from linear_dag.association.simulation import simulate_phenotype
from linear_dag.core.operators import get_diploid_operator, get_inner_merge_operators
from linear_dag.core.parallel_processing import ParallelOperator
from scipy.sparse.linalg import aslinearoperator


def test_gwas_hwe(linarg_h5_path):
    """
    Test that run_gwas can recover a simulated causal effect.
    """
    # 1. Setup
    hdf5_path = linarg_h5_path
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

        # 3. Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame(
            {
                "iid": genotypes.iids.unique(maintain_order=True).to_list(),
                "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
                "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
            }
        ).with_columns(
            intercept=pl.lit(1),
            covar1=np.random.rand(n // 2),
            covar2=y[:, 0] + np.random.rand(n // 2),
        )

        # 4. GWAS
        pheno_cols = ["phenotype1", "phenotype2"]
        covar_cols = ["intercept"]  # , "covar1", "covar2"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df.lazy(),
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
            in_place_op=True,
        ).collect()
        beta = gwas_results.select("phenotype1_BETA").to_numpy().copy()
        se = gwas_results.select("phenotype1_SE").to_numpy().copy()

        # 5. Simple GWAS
        ident = np.eye(n)
        gt_mat = ident @ genotypes
        left_op, _ = get_inner_merge_operators(pheno_df.select("iid").cast(pl.Utf8).to_series(), genotypes.iids)
        beta_simple, se_simple = simple_gwas(
            gt_mat,
            left_op.T @ pheno_df.select("phenotype1").to_numpy(),
            left_op.T @ pheno_df.select(covar_cols).to_numpy(),
            ploidy=2,
        )
        # 6. Assertions
        np.nan_to_num(beta_simple, copy=False)
        np.nan_to_num(se_simple, copy=False)
        np.nan_to_num(beta, copy=False)
        np.nan_to_num(se, copy=False)
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        assert np.allclose(beta, beta_simple, atol=1e-6)
        se = se[beta != 0]
        se_simple = se_simple[beta != 0]
        assert np.allclose(se, se_simple, atol=1e-6)


def test_gwas_no_hwe(linarg_h5_path):
    """
    Test that run_gwas can recover a simulated causal effect.
    """
    # 1. Setup
    hdf5_path = linarg_h5_path
    heritability = 0.5
    np.random.seed(42)

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

        # 3. Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame(
            {
                "iid": genotypes.iids.unique(maintain_order=True).to_list(),
                "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
                "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
            }
        ).with_columns(
            intercept=pl.lit(1),
            covar1=np.random.rand(n // 2),
            covar2=y[:, 0] + np.random.rand(n // 2),
        )

        # 4. GWAS not assuming HWE
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
        se = gwas_results.select("phenotype1_SE").to_numpy().copy()

        # 5. Simple GWAS
        ident = np.eye(n // 2)
        gt_mat = ident @ get_diploid_operator(genotypes)
        beta_simple, se_simple = simple_gwas(
            gt_mat, pheno_df.select("phenotype1").to_numpy(), pheno_df.select(covar_cols).to_numpy()
        )

        # 6. Assertions
        np.nan_to_num(beta_simple, copy=False)
        np.nan_to_num(se_simple, copy=False)
        np.nan_to_num(beta, copy=False)
        np.nan_to_num(se, copy=False)
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        assert np.allclose(beta, beta_simple, atol=1e-3)
        se = se[beta != 0]
        se_simple = se_simple[beta != 0]
        assert np.allclose(se, se_simple, atol=1e-3)


def test_gwas_missingness(linarg_h5_path):
    """
    Test that run_gwas can recover a simulated causal effect.
    """
    # 1. Setup
    hdf5_path = linarg_h5_path
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

        y[::2, 0] = np.nan

        # 3. Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame(
            {
                "iid": genotypes.iids.unique(maintain_order=True).to_list(),
                "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
                "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
            }
        ).with_columns(
            intercept=pl.lit(1),
            covar1=np.random.rand(n // 2),
            covar2=y[:, 0] + np.random.rand(n // 2),
        )

        # 4. GWAS not assuming HWE
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
        se = gwas_results.select("phenotype1_SE").to_numpy().copy()

        # 5. Simple GWAS
        ident = np.eye(n // 2)
        gt_mat = ident @ get_diploid_operator(genotypes)
        beta_simple, se_simple = simple_gwas(
            gt_mat, pheno_df.select("phenotype1").to_numpy(), pheno_df.select(covar_cols).to_numpy()
        )

        # 6. Assertions
        np.nan_to_num(beta_simple, copy=False)
        np.nan_to_num(se_simple, copy=False)
        np.nan_to_num(beta, copy=False)
        np.nan_to_num(se, copy=False)
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        se = se[beta != 0]
        se_simple = se_simple[beta != 0]
        assert np.isclose(np.median(se**2), np.median(se_simple**2), atol=1e-1)
        assert np.isclose(np.median(beta**2), np.median(beta_simple**2), atol=1e-1)


def test_gwas_recompute_AC(linarg_h5_path):
    """
    Test that run_gwas can handle missingness with HWE assumption.
    """
    # 1. Setup
    hdf5_path = linarg_h5_path
    heritability = 0.5

    # 2. Simulation
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, max_num_traits=16) as genotypes:
        n, m = genotypes.shape
        fraction_causal = 1

        y, beta = simulate_phenotype(
            get_diploid_operator(genotypes),
            heritability=heritability,
            fraction_causal=fraction_causal,
            return_beta=True,
            num_traits=2,
        )

        y[::2, 0] = np.nan

        # 3. Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame(
            {
                "iid": genotypes.iids.unique(maintain_order=True).to_list(),
                "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
                "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
            }
        ).with_columns(
            intercept=pl.lit(1),
        )

        # 4. GWAS with recompute_AC
        pheno_cols = ["phenotype1", "phenotype2"]
        covar_cols = ["intercept"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df.lazy(),
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
            in_place_op=False,
            assume_hwe=True,
            recompute_AC=True,
        ).collect()
        beta = gwas_results.select("phenotype1_BETA").to_numpy().copy()
        se = gwas_results.select("phenotype1_SE").to_numpy().copy()

        # 5. Simple GWAS
        ident = np.eye(n)
        gt_mat = ident @ genotypes
        left_op, _ = get_inner_merge_operators(pheno_df.select("iid").cast(pl.Utf8).to_series(), genotypes.iids)
        beta_simple, se_simple = simple_gwas(
            gt_mat,
            left_op.T @ pheno_df.select("phenotype1").to_numpy(),
            left_op.T @ pheno_df.select(covar_cols).to_numpy(),
            ploidy=2,
        )

        # 6. Assertions
        np.nan_to_num(beta_simple, copy=False)
        np.nan_to_num(se_simple, copy=False)
        np.nan_to_num(beta, copy=False)
        np.nan_to_num(se, copy=False)
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        assert np.allclose(beta, beta_simple, atol=1e-6)
        se = se[beta != 0]
        se_simple = se_simple[beta != 0]
        assert np.allclose(se, se_simple, atol=1e-6)


def test_gwas_recompute_AC_no_hwe(linarg_h5_path):
    """
    Test that run_gwas can handle missingness without HWE assumption.
    """
    # 1. Setup
    hdf5_path = linarg_h5_path
    heritability = 0.5

    # 2. Simulation
    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2, max_num_traits=16) as genotypes:
        n, m = genotypes.shape
        fraction_causal = 1

        y, beta = simulate_phenotype(
            get_diploid_operator(genotypes),
            heritability=heritability,
            fraction_causal=fraction_causal,
            return_beta=True,
            num_traits=2,
        )

        # y[::2, 0] = np.nan

        # 3. Create phenotype DataFrame with covariates
        pheno_df = pl.DataFrame(
            {
                "iid": genotypes.iids.unique(maintain_order=True).to_list(),
                "phenotype1": pl.Series(y[:, 0].flatten(), dtype=pl.Float64).cast(pl.Float64).fill_nan(None),
                "phenotype2": pl.Series(y[:, 1].flatten()).fill_nan(None),
            }
        ).with_columns(
            intercept=pl.lit(1),
        )

        # 4. GWAS with recompute_AC
        pheno_cols = ["phenotype1", "phenotype2"]
        covar_cols = ["intercept"]
        gwas_results = run_gwas(
            genotypes,
            pheno_df.lazy(),
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
            in_place_op=False,
            assume_hwe=False,
            recompute_AC=True,
        ).collect()
        beta = gwas_results.select("phenotype1_BETA").to_numpy().copy()
        se = gwas_results.select("phenotype1_SE").to_numpy().copy()

        # 5. Simple GWAS
        ident = np.eye(n // 2)
        gt_mat = ident @ get_diploid_operator(genotypes)
        beta_simple, se_simple = simple_gwas(
            gt_mat, pheno_df.select("phenotype1").to_numpy(), pheno_df.select(covar_cols).to_numpy()
        )

        # 6. Assertions
        np.nan_to_num(beta_simple, copy=False)
        np.nan_to_num(se_simple, copy=False)
        np.nan_to_num(beta, copy=False)
        np.nan_to_num(se, copy=False)
        assert isinstance(gwas_results, pl.DataFrame)
        assert gwas_results.shape[0] == m
        assert gwas_results.select("phenotype1_BETA").dtypes[0] == pl.Float32
        assert gwas_results.select("phenotype1_SE").dtypes[0] == pl.Float32
        assert np.median(np.abs(beta - beta_simple)) < 1e-4
        assert np.max(np.abs(beta - beta_simple)) < 1
        assert np.median(np.abs(se - se_simple)) < 1e-4
        assert np.max(np.abs(se - se_simple)) < 1


class _InvalidNonHWEGenotypes:
    n_individuals = 2
    iids = pl.Series(["id1", "id1", "id2", "id2"])


def test_run_gwas_non_hwe_requires_heterozygote_counter():
    data = pl.DataFrame(
        {
            "iid": ["id1", "id2"],
            "phenotype1": [0.1, -0.2],
            "intercept": [1.0, 1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="If assume_hwe is False, genotypes must expose n_individuals, iids, and number_of_heterozygotes\\(\\).",
    ):
        run_gwas(
            genotypes=_InvalidNonHWEGenotypes(),
            data=data.lazy(),
            pheno_cols=["phenotype1"],
            covar_cols=["intercept"],
            assume_hwe=False,
        )


def test_get_gwas_beta_se_returns_four_arrays():
    genotypes = aslinearoperator(
        np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    right_op = aslinearoperator(np.eye(4, dtype=np.float32))
    y_resid = np.array([[0.1], [1.2], [-0.5], [0.7]], dtype=np.float32)
    covariates = np.ones((4, 1), dtype=np.float32)

    beta, var_numerator, var_denominator, allele_counts = get_gwas_beta_se(
        genotypes,
        right_op,
        y_resid,
        covariates,
    )

    assert beta.shape == (3, 1)
    assert var_numerator.shape == (1,)
    assert var_denominator.shape == (3, 1)
    assert allele_counts.shape == (3, 1)
