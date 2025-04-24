import numpy as np
from linear_dag.association import simulate_phenotype, get_gwas_betas
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag.core.lineararg import diploid_operator

linarg_path = "data/test/test_chr21_100_haplotypes.h5"
def test_gwas():
    
    with ParallelOperator.from_hdf5(linarg_path) as genotypes:
        genotypes_diploid = diploid_operator(genotypes)
        genotypes_diploid_normalized = diploid_operator(genotypes.normalized) / np.sqrt(2)
        n, m = genotypes_diploid.shape
        p_missing = np.array([0, 0.5, 0, 0.1, 0.5, 0.9])
        num_traits = len(p_missing)

        y, beta = simulate_phenotype(genotypes_diploid, heritability=0, num_traits=num_traits, fraction_causal=0.01, return_beta=True)
        for i in range(num_traits):
            is_missing = np.random.rand(n) < p_missing[i]
            y[is_missing, i] = np.nan

        print(f"mean missingness: {np.mean(np.isnan(y), axis=0)}")
        # indicator = np.zeros((m,2))
        # indicator[0,0] = 1
        # indicator[1,1] = 1
        # covariates = genotypes.normalized @ indicator 
        covariates = y[:, :2]
        covariates += np.random.randn(*covariates.shape)
        covariates = np.hstack([covariates, np.ones((n, 1))])
        
        betas = get_gwas_betas(genotypes_diploid_normalized, y, covariates)
        z_scores = betas * np.sqrt(np.sum(~np.isnan(y)))
    
    print(f"Mean chisq: {np.mean(z_scores**2, axis=0)}")
    if np.any(beta != 0):
        print(f"Mean chisq causal betas: {np.mean(z_scores.ravel()[beta.ravel() != 0]**2)}")

if __name__ == "__main__":
    test_gwas()
    