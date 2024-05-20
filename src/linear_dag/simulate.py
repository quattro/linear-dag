from dataclasses import dataclass

import networkx as nx
import numpy as np

from numpy.typing import NDArray
from scipy.io import mmwrite
from scipy.sparse import csc_matrix

from .lineararg import LinearARG
from .one_summed import construct_1_summed_DAG_slow


@dataclass
class Simulate(LinearARG):
    A_haplo: NDArray
    haplotypes: NDArray
    genotypes: NDArray

    @staticmethod
    def simulate_example(*, example: str = "2-1", ns: int = 10):
        # Initial ARG (not one-summed)
        if example == "2-1":
            A_haplo = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 1]]

        elif example == "3-2-1":
            A_haplo = [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]

        elif example == "2-2-1":
            A_haplo = [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 1, 0, 0],
            ]

        elif example == "4-2-1":
            A_haplo = [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1],
            ]

        elif example == "4-2":
            A_haplo = [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
            ]
        else:
            raise ValueError("Valid examples are '4', '2-1', '3-2-1' and '2-2-1'")

        A_haplo = np.asarray(A_haplo)
        nh, nm = A_haplo.shape
        A_haplo = np.hstack((A_haplo, np.zeros((nh, nh - nm))))

        # Convert to linear ARG
        G = nx.from_numpy_array(A_haplo, create_using=nx.DiGraph)
        G = construct_1_summed_DAG_slow(G)
        A_haplo = nx.to_numpy_array(G)
        print(A_haplo)

        # Set of possible haplotypes
        haplotypes = np.linalg.inv(np.eye(nh) - A_haplo)
        haplotypes = haplotypes[:, :nm]

        # Sample with replacement from the possible haplotypes
        random_indices = np.random.choice(nh, size=ns, replace=True)
        genotypes = csc_matrix(haplotypes[random_indices, :])

        # Samples then variants
        A = np.vstack((A_haplo[random_indices, :nm], A_haplo[:, :nm]))
        A = np.hstack((np.zeros((nh + ns, nh + ns - nm)), A))

        sample_indices = np.arange(ns)
        variant_indices = np.arange(ns, ns + nm)
        flip = np.zeros(nm, dtype=bool)

        return Simulate(
            A_haplo=A_haplo,
            haplotypes=haplotypes,
            genotypes=genotypes,
            sample_indices=sample_indices,
            variant_indices=variant_indices,
            A=A,
            flip=flip,
        )

    def write_genotypes(self, filename):
        mmwrite(filename + ".mtx", csc_matrix(self.genotypes))
