import linear_dag as ld
import numpy as np

from scipy.io import mmread


def main():
    # genotypes = mmread("/Users/loconnor/Dropbox/linearArg/linearg_shared/genotypes.mtx")
    genotypes = mmread("/Users/nicholas/Dropbox/collab/nick-luke/linearg_shared/genotypes.mtx")
    genotypes = genotypes.tocsc()
    ploidy = np.max(genotypes).astype(int)
    genotypes = ld.apply_maf_threshold(genotypes, 0.001)
    genotypes, flipped = ld.flip_alleles(genotypes, ploidy)
    linarg = ld.Linarg.from_genotypes(genotypes, ploidy)
    linarg = linarg.find_recombinations()

    assert linarg.A.shape == (40270, 40270)
    assert linarg.A.nnz == 310345

    print(str(linarg))
    print("OK")


if __name__ == "__main__":
    main()
