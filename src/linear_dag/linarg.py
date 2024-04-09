# linarg.py
from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from scipy.io import mmwrite
from scipy.sparse import csc_matrix, csr_matrix, eye, hstack, triu, vstack

from .solve import spinv_triangular
from .trios import Trios


def _construct_A(genotypes: csc_matrix, ploidy: int = 1) -> csc_matrix:
    for n in range(1, ploidy + 1):
        X_carrier = genotypes >= n
        X_carrier = X_carrier.astype(np.int32)
        R_carrier = X_carrier.transpose().dot(X_carrier)

        temp = R_carrier.copy()
        for i in range(R_carrier.shape[0]):
            row_data = R_carrier.data[R_carrier.indptr[i] : R_carrier.indptr[i + 1]]
            temp.data[temp.indptr[i] : temp.indptr[i + 1]] = row_data >= R_carrier[i, i]

        if n == 1:
            haplotypes = temp
        else:
            haplotypes = haplotypes.multiply(temp)

    haplotypes.eliminate_zeros()
    ties = haplotypes.multiply(haplotypes.transpose())
    haplotypes = haplotypes - triu(ties, k=1)
    haplotypes.eliminate_zeros()

    row_counts = np.diff(haplotypes.indptr)
    triangular_order = np.argsort(-row_counts)
    original_order = np.argsort(triangular_order)
    haplotypes = haplotypes[triangular_order, :][:, triangular_order].astype(np.int32)
    haplotypes = haplotypes.tocsr()
    haplotypes.sort_indices()
    indptr, indices, data = spinv_triangular(haplotypes.indptr, haplotypes.indices, haplotypes.data)

    A = csr_matrix((data, indices, indptr))
    A = A[original_order, :][:, original_order]
    A.eliminate_zeros()

    return A


@dataclass
class Linarg:
    A: csr_matrix
    sample_indices: NDArray
    variant_indices: NDArray

    @staticmethod
    def from_genotypes(genotypes: csc_matrix, ploidy: int = 1) -> "Linarg":
        n, m = genotypes.shape
        A_haplo = _construct_A(genotypes, ploidy)

        # Fit samples to the matrix
        A_sample = genotypes @ A_haplo
        A_haplo = eye(m) - A_haplo

        # Concatenate
        zeros_matrix = csr_matrix((m + n, n))
        vertical_stack = vstack([A_sample, A_haplo])
        A = hstack([zeros_matrix, vertical_stack])
        A = A.astype(np.int32)
        A.eliminate_zeros()
        A = csr_matrix(A)

        # Indices in A of the samples and variants
        samples_idx = np.arange(n)
        variants_idx = np.arange(n, n + m)

        return Linarg(A, samples_idx, variants_idx)

    @property
    def shape(self):
        n = len(self.sample_indices)
        m = len(self.variant_indices)
        return n, m

    @property
    def nnz(self):
        return self.A.nnz

    def __str__(self):
        return f"A: shape {self.A.shape}, nonzeros {self.A.nnz}"

    def write(self, filename: str):
        # Write samples
        with open(filename + ".samples.txt", "w") as f_samples:
            for sample in self.samples:
                f_samples.write(f"{sample}\n")

        # Write variants
        with open(filename + ".mutations.txt", "w") as f_variants:
            for variant in self.variants:
                f_variants.write(f"{variant}\n")

        # Write the matrix A
        mmwrite(filename + ".mtx", self.A)

    def compute_hierarchy(self, haplotypes: csc_matrix) -> NDArray:
        A_csr = self.A
        n_sample, m_sample = self.shape
        rank = np.zeros(self.nnz)
        H = haplotypes - eye(haplotypes.shape[0])
        for child in range(self.A.shape[0]):
            # Restrict haplotype matrix to parents of that child
            parents = A_csr.indices[A_csr.indptr[child] : A_csr.indptr[child + 1]]
            parents_as_indices = parents - n_sample
            haplotype_submatrix = H[parents_as_indices, :][:, parents_as_indices]
            X = csc_matrix(eye(len(parents)))
            while np.any(X.data):
                X = haplotype_submatrix @ X
                rank[parents_as_indices] += np.diff(X.indptr) > 0

        return rank

    def find_recombinations(self, ranked: bool = False) -> "Linarg":
        trio_list = Trios(2 * self.A.nnz)  # TODO what should n be?
        if ranked:
            rank = self.compute_hierarchy()
            trio_list.convert_matrix_ranked(self.A.data, self.A.indices, self.A.indptr, self.A.indptr.shape[0], rank)
        else:
            trio_list.convert_matrix(self.A.data, self.A.indices, self.A.indptr, self.A.indptr.shape[0])

        trio_list.find_recombinations()

        # Verify the trio list remains valid
        trio_list.check_properties(-1)

        # Compute new edge list
        edges = trio_list.fill_edgelist()
        num_nodes = np.max(edges[:, 0:2]) + 1

        A = csr_matrix((edges[:, 2], (edges[:, 1], edges[:, 0])), shape=(num_nodes, num_nodes))

        return Linarg(A, self.sample_indices, self.variant_indices)
