from dataclasses import dataclass

import networkx as nx
import numpy as np

from numpy.typing import NDArray
from scipy.io import mmwrite
from scipy.sparse import csc_matrix, csr_matrix

from .lineararg import LinearARG
from .one_summed import construct_1_summed_DAG_slow


@dataclass
class Simulate(LinearARG):
    A_ancestral: NDArray
    ancestral_haplotypes: NDArray
    sample_haplotypes: NDArray

    @staticmethod
    def simulate_example(*, example: str = "2-1", ns: int = 10):

        # A_ancestral is a number-of-haplotypes by number-of-mutations matrix,
        # where the mutation in column j occurs on the haplotype in row j. If
        # there are more rows than columns, then rows >= num_columns are haplotypes
        # that do not have mutations.
        if example == "2-1":
            A_ancestral = [[0, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [0, 1, 1]]

        elif example == "3-2-1":
            A_ancestral = [
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
            A_ancestral = [
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
            A_ancestral = [
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
            A_ancestral = [
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

        A_ancestral = np.asarray(A_ancestral)
        nh, nm = A_ancestral.shape
        assert nh >= nm, "ARG should have at least as many haplotypes as mutations."
        A_ancestral = np.hstack((A_ancestral, np.zeros((nh, nh - nm))))

        # Convert to linear ARG
        G = nx.from_numpy_array(A_ancestral, create_using=nx.DiGraph)
        G = construct_1_summed_DAG_slow(G)
        A_ancestral = nx.to_numpy_array(G)

        # Set of possible haplotypes
        haplotypes = np.linalg.inv(np.eye(nh) - A_ancestral)
        haplotypes = haplotypes[:, :nm]

        # Sample with replacement from the possible haplotypes
        random_indices = np.random.choice(nh, size=ns, replace=True)
        sample_haplotypes = csc_matrix(haplotypes[random_indices, :])

        # Samples then ancestral haplotypes
        identity = np.eye(nh)
        A = np.vstack((identity[random_indices, :], A_ancestral[:, :]))
        A = np.hstack((np.zeros((nh + ns, ns)), A))

        sample_indices = np.arange(ns)
        variant_indices = np.arange(ns, ns + nm)
        flip = np.zeros(nm, dtype=bool)

        return Simulate(
            A_ancestral=A_ancestral,
            ancestral_haplotypes=haplotypes,
            sample_haplotypes=sample_haplotypes,
            sample_indices=sample_indices,
            variant_indices=variant_indices,
            A=csr_matrix(A),
            flip=flip,
        )

    def write_genotypes(self, filename):
        mmwrite(filename + ".mtx", csc_matrix(self.sample_haplotypes))
