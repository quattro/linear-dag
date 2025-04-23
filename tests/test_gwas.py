import numpy as np
from linear_dag.association import simulate_phenotype, get_gwas_betas
from linear_dag.core.parallel_processing import ParallelOperator

linarg_path = "data/test/test_chr21_100_haplotypes.h5"
def test_gwas():
    
    with ParallelOperator.from_hdf5(linarg_path) as genotypes:
        n, m = genotypes.shape
        y, beta = simulate_phenotype(genotypes, heritability=0.5, num_traits=10, fraction_causal=0.01, return_beta=True)
        betas = get_gwas_betas(genotypes.normalized, y, np.ones((genotypes.num_samples, 1)))
        z_scores = betas * np.sqrt(n)
    
    print(f"Mean chisq: {np.mean(z_scores**2)}")
    if np.any(beta != 0):
        print(f"Mean chisq causal betas: {np.mean(z_scores.ravel()[beta.ravel() != 0]**2)}")

if __name__ == "__main__":
    test_gwas()
    