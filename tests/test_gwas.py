import numpy as np
from linear_dag.association import simulate_phenotype, get_gwas_beta_se, run_gwas
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag.core.operators import get_diploid_operator
import linear_dag as ld
import polars as pl
import h5py

def main():
    run_test("data/test/test_chr21_small_all.h5", 
                missingness = np.array([0]*5 + [0.99]*5),  #np.zeros(10),#
                heritability=0,
                traits_with_correlated_covariates = np.arange(0,10,2))


def run_test(linarg_path: str, 
            missingness: np.ndarray = np.zeros(10),
            heritability: float = 0,
            traits_with_correlated_covariates = None
            ):
    with ParallelOperator.from_hdf5(linarg_path) as genotypes:
        print(genotypes.shape)

        genotypes_diploid = get_diploid_operator(genotypes)
        n, m = genotypes_diploid.shape
        num_traits = 10

        y, beta = simulate_phenotype(genotypes_diploid, heritability=heritability, num_traits=num_traits, fraction_causal=0.01, return_beta=True)
        covariates = y[:, traits_with_correlated_covariates].copy()
        covariates += np.random.randn(*covariates.shape)
        print(f"covariates: {covariates[:10, :2]}")

        # Add missingness
        for i in range(num_traits):
            mask = np.random.random(n) < missingness[i]
            y[mask, i] = np.nan

        covariates = np.hstack([np.ones((n, 1)), covariates])
        num_covars = covariates.shape[1]

        # Prepare Polars DataFrame input using actual IIDs
        pheno_cols = [f"pheno_{i}" for i in range(num_traits)]
        covar_cols = [f"covar_{i}" for i in range(num_covars)] # Constant is covar_0

        diploid_iids = genotypes.iids.unique(maintain_order=True)

        data_dict = {"iid": diploid_iids}
        data_dict.update({pheno_cols[i]: y[:, i] for i in range(num_traits)})
        data_dict.update({covar_cols[i]: covariates[:, i] for i in range(num_covars)})

        data_lazy = pl.DataFrame(data_dict).lazy()

        # --- Call run_gwas ---
        print("Running GWAS...")
        results_lazy = run_gwas(
            genotypes=genotypes,
            data=data_lazy,
            pheno_cols=pheno_cols,
            covar_cols=covar_cols,
        )

        af = genotypes.allele_frequencies

        results_df = results_lazy.collect()
        print("GWAS finished. Results shape:", results_df.shape)
        # bad = results_df.filter(pl.col('trait') == 'pheno_0', pl.col('z')**2 > 100)
        # if not bad.is_empty():
        #     print('bad rows:')
        #     print(bad)

        # --- Assertions ---
        assert isinstance(results_df, pl.DataFrame)
        assert results_df.shape[0] == m * num_traits
        # Check essential GWAS columns + variant info columns
        expected_cols = ['variant_index', 'beta', 'se', 'n', 'trait', 'z', 'pval']

        missing_cols = set(expected_cols) - set(results_df.columns)
        assert not missing_cols, f"Missing columns in results: {missing_cols}"

        assert results_df['trait'].dtype == pl.Enum # Check Enum type
        assert results_df['n'].dtype == pl.Float32 # Check n type
        assert not results_df.is_empty()

        # Check basic stats
        mean_chisq = results_df.group_by('trait').agg(pl.col('z').pow(2).mean().alias('mean_chisq'))
        print("\nMean ChiSq per trait:")
        print(mean_chisq)
        # Basic sanity check on chisq values
        assert mean_chisq['mean_chisq'].min() > 0.01

        expected_max_chisq_ub = np.max(beta**2 * (af * (1-af)).reshape(-1,1)) * n * 2 + 10
        assert mean_chisq['mean_chisq'].max() < expected_max_chisq_ub

        # Check p-value range
        assert results_df['pval'].min() >= 0.0
        assert results_df['pval'].max() <= 1.0

        # Optional: Add more sophisticated checks comparing results to true beta if needed

if __name__ == "__main__":
    main()
    # test_gwas()
    # test_run_gwas() # Add call here if running script directly