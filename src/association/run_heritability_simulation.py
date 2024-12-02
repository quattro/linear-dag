from heritability import randomized_haseman_elston
from simulation import simulate_phenotype
from linear_dag import LinearARG
import numpy as np

def main():
    path = "/Users/loconnor/Dropbox/linearARG/linearg_shared/data/linarg/1kg_chr20_1000000_2000000.npz"
    linarg = LinearARG.read(path)
    print(linarg.shape)

    h2, alpha, pi, traits = .2, -.5, .01, 100
    y = np.zeros((linarg.shape[0], traits))
    for i in range(traits):
        y[:,i] = simulate_phenotype(linarg, h2, alpha, pi)

    num_random_vectors = 20
    h2_est = randomized_haseman_elston(linarg, y, B=num_random_vectors, alpha=alpha)
    print(np.mean(h2_est), np.std(h2_est))

if __name__ == '__main__':
    main()