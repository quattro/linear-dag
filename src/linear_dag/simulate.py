import networkx as nx
import numpy as np

from scipy.io import mmwrite
from scipy.sparse import csc_matrix

from .linarg import Linarg
from .one_summed import construct_1_summed_DAG_slow


class Simulate(Linarg):
    def simulate_example(self, example: str = "2-1", ns: int = 10):
        # Initial ARG (not one-summed)
        if example == "2-1":
            self.A_haplo = [[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]

        elif example == "3-2-1":
            self.A_haplo = [
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
            self.A_haplo = [
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
        else:
            raise ValueError("Valid examples are '2-1', '3-2-1' and '2-2-1'")

        self.A_haplo = np.asarray(self.A_haplo)
        nh, nm = self.A_haplo.shape
        self.A_haplo = np.hstack((self.A_haplo, np.zeros((nh, nh - nm))))

        # Convert to linear ARG
        G = nx.from_numpy_array(self.A_haplo, create_using=nx.DiGraph)
        G = construct_1_summed_DAG_slow(G)
        self.A_haplo = nx.to_numpy_array(G)

        # Set of possible haplotypes
        self.haplotypes = np.linalg.inv(np.eye(nh) - self.A_haplo)
        self.haplotypes = self.haplotypes[:, :nm]

        # Sample with replacement from the possible haplotypes
        random_indices = np.random.choice(nh, size=ns, replace=True)
        self.genotypes = csc_matrix(self.haplotypes[random_indices, :])

        # Samples then variants
        self.A = np.vstack((self.A_haplo[random_indices, :nm], self.A_haplo[:, :nm]))
        self.A = np.hstack((np.zeros((nh + ns, nh + ns - nm)), self.A))

        self.samples = np.arange(ns)
        self.variants = np.arange(ns, ns + nm)

    def linarg(self) -> Linarg:
        copy = Linarg()
        copy.genotypes = csc_matrix(self.genotypes)
        copy.variants = np.copy(self.variants)
        copy.samples = np.copy(self.samples)
        return copy

    def write_genotypes(self, filename):
        mmwrite(filename + ".mtx", csc_matrix(self.genotypes))
