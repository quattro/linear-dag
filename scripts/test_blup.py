import linear_dag as ld
import numpy as np
from time import time

def main():
    np.random.seed(0)
    path = "/Users/lukeoconnor/Desktop/1kg_chr20_1000000_50000000"
    linarg = ld.LinearARG.read(path + ".npz", path + ".pvar", path + ".psam")

    y, y_bar = ld.association.simulate_phenotype(linarg=linarg, heritability=0.1, return_genetic_component=1)

    t = time()
    estimated_h2 = ld.association.randomized_haseman_elston(linarg, y, num_matvecs=10, trace_est="xnystrace")
    print(f"Problem size: {linarg.shape}, RHE time taken: {time() - t}, estimated heritability: {estimated_h2}")

    t = time()
    blup = ld.association.blup(linarg=linarg, heritability=estimated_h2[0], y=y)
    corr_BLUP_y_bar = np.corrcoef(y_bar, blup)[0,1]
    print(f"Problem size: {linarg.shape}, BLUP time taken: {time() - t}, BLUP correlation: {corr_BLUP_y_bar}")

if __name__ == "__main__":
    main()