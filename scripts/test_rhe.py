import argparse as ap
import sys

import numpy as np

from linear_dag import (
    LinearARG,
    randomized_haseman_elston,
)
from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator


def simulate_phenotype(
    linarg: LinearARG,
    heritability: float,
    alpha: float = 0.0,
    fraction_causal: float = 1.0,
    seed=None,
):
    """
    Simulates quantitative phenotypes
        y = X * beta + epsilon
    with specified heritability, AF-dependent architecture, and polygenicity
    :param linarg: genotypes
    :param heritability: between 0 and 1
    :param alpha: between -1 and 0; cf. Schoech et al 2019 Nat Commun
    :param fraction_causal: between 0 and 1
    :return: vector of phenotype values
    """

    N, M = linarg.shape
    generator = np.random.default_rng(seed=seed)
    beta = generator.standard_normal(size=(M,))

    heterozygosity = 2 * linarg.allele_frequencies * (1 - linarg.allele_frequencies)
    beta[heterozygosity == 0] = 0
    heterozygosity[heterozygosity == 0] = 1
    beta = beta * np.sqrt(heterozygosity**alpha)
    is_causal = generator.uniform(size=(M,)) < fraction_causal
    beta = beta * is_causal

    y_bar = linarg @ beta
    y_bar -= np.mean(y_bar)
    y_bar /= np.std(y_bar)
    y_bar *= np.sqrt(heritability)
    y = y_bar + generator.standard_normal(size=(N,)) * np.sqrt(1 - heritability)

    return y


def _rhe(linarg, ys, B, alpha=-0.5, seed=None):
    n = linarg.shape[0]

    generator = np.random.default_rng(seed=seed)

    heterozygosity = linarg.allele_frequencies * (1 - linarg.allele_frequencies)
    heterozygosity[heterozygosity == 0] = 1

    # Genetic relatedness matrix
    sigmasq = heterozygosity ** (1 + alpha)
    Z = linarg.normalized
    K = Z @ aslinearoperator(diags(sigmasq / np.sum(sigmasq))) @ Z.T

    grm_trace = grm_sq_trace = 0.0
    for b in range(B):
        x = generator.standard_normal(size=(n,))
        Kx = K @ x
        grm_trace += np.dot(x, Kx)
        grm_sq_trace += np.dot(Kx, Kx)

    grm_trace /= B
    grm_sq_trace /= B

    # center and standardize
    ys = ys - np.mean(ys, axis=0)
    ys = ys / np.std(ys, axis=0)

    # compute y_j' K y_j for each y_j \in y
    C = np.sum(K.matmat(ys) ** 2, axis=0)

    # construct linear equations to solve
    LHS = np.array([[grm_sq_trace, grm_trace], [grm_trace, n]])
    RHS = np.vstack([C, n * np.ones_like(C)])
    solution = np.linalg.solve(LHS, RHS)
    heritability = solution[0, :] / (solution[0, :] + solution[1, :])

    return heritability


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("-B", "--num-vec", type=int, default=10)
    argp.add_argument("-a", "--alpha", type=float, default=-0.5)
    argp.add_argument("-m", "--method", choices=["hutchinson", "hutch++", "xtrace", "xnystrace"], default="hutchinson")
    argp.add_argument("-d", "--distribution", choices=["normal", "sphere", "rademacher"], default="normal")
    argp.add_argument("-s", "--seed", type=int, default=None)
    argp.add_argument("-o", "--output", type=ap.FileType("w"), default=sys.stdout)

    args = argp.parse_args(args)

    path = "../data/linearg_shared/data/linarg/1kg_chr20_1000000_2000000.npz"
    linarg = LinearARG.read(path)
    print(linarg.shape)

    h2, pi, traits = 0.2, 0.01, 100
    y = np.zeros((linarg.shape[0], traits))
    generator = np.random.default_rng(seed=args.seed)
    for i in range(traits):
        y[:, i] = simulate_phenotype(linarg, h2, args.alpha, pi, seed=generator)

    h2_est = randomized_haseman_elston(
        linarg,
        y,
        B=args.num_vec,
        alpha=args.alpha,
        trace_est=args.method,
        sampler=args.distribution,
        seed=generator,
    )

    mean_h2 = np.mean(h2_est)
    std_h2 = np.std(h2_est)
    mse = np.sqrt(np.mean(h2 - h2_est) ** 2)
    args.output.write(f"mean estimate (sd): {mean_h2} ({std_h2}) | mse = {mse}\n")

    h2_est = _rhe(
        linarg,
        y,
        B=args.num_vec,
        alpha=args.alpha,
        seed=generator,
    )
    mean_h2 = np.mean(h2_est)
    std_h2 = np.std(h2_est)
    mse = np.sqrt(np.mean(h2 - h2_est) ** 2)
    args.output.write(f"classic mean estimate (sd): {mean_h2} ({std_h2}) | mse = {mse}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
