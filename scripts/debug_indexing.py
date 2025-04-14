import numpy as np
import time
import linear_dag as ld
import sys
import os
from scipy.sparse import csr_matrix, csc_matrix, eye
import h5py
import numpy as np


def main():
    linarg_path = 'data/linearg_shared/data/1kg_chr/1kg_chr1'
    linarg_path = 'data/1kg_commonsnps_chr22_46177037_47876022'
    # linarg_path = 'data/linearg_shared/data/1kg_commonsnps/1kg_chr22'

    t1 = time.time()
    linarg = ld.LinearARG.read(f'{linarg_path}.npz', f'{linarg_path}.pvar', f'{linarg_path}.psam')
    t2 = time.time()
    print(f'Time to load LinearARG: {t2 - t1:.3f} seconds')

    A_csr = csr_matrix(linarg.A)
    row_degree = np.diff(A_csr.indptr)
    col_degree = np.diff(linarg.A.indices)
    nodes = np.where(row_degree == 0)[0]
    for node in nodes:
        if node not in linarg.variant_indices and node not in linarg.sample_indices:
            print(f"Node {node} has no column neighbors and has row degree {row_degree[node]}")

    


if __name__ == "__main__":
    main()