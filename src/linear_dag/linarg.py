# linarg.py
from dataclasses import dataclass

import numpy as np

from numpy.typing import NDArray
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix, eye, hstack, triu, vstack
from scipy.sparse.linalg import spsolve_triangular

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
    haplotypes = haplotypes.tocsc()
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
    flip: NDArray

    @staticmethod
    def from_genotypes(genotypes: csc_matrix, ploidy: int = 1, flip: NDArray = None) -> "Linarg":
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

        if flip is None:
            flip = np.zeros(len(variants_idx), dtype=bool)

        return Linarg(A, samples_idx, variants_idx, flip)

    @staticmethod
    def from_file(filename: str) -> "Linarg":
        # Read samples
        sample_list = []
        with open(filename + ".samples.txt", "r") as f:
            header = next(f)
            assert header == "index\n"
            for line in f:
                sample_list.append(int(line) - 1)

        # Read variants
        variant_list = []
        flip_list = []
        with open(filename + ".mutations.txt", "r") as f:
            header = next(f)
            assert header == "index,flip\n"
            for line in f:
                index, flip = [int(n) for n in line.split(",")]
                variant_list.append(index - 1)
                flip_list.append(flip)

        # Read the matrix A
        A = csr_matrix(mmread(filename + ".mtx"))

        return Linarg(
            A,
            np.asarray(sample_list, dtype=int),
            np.asarray(variant_list, dtype=int),
            np.asarray(flip_list, dtype=bool),
        )

    @property
    def shape(self):
        n = len(self.sample_indices)
        m = len(self.variant_indices)
        return n, m

    @property
    def nnz(self):
        return self.A.nnz

    @property
    def ndim(self):
        return 2

    def __str__(self):
        return f"A: shape {self.A.shape}, nonzeros {self.A.nnz}"

    def __matmul__(self, other: NDArray) -> NDArray:
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and {other.shape}."
            )

        other[self.flip] *= -1
        v = np.zeros((self.A.shape[0], other.shape[1]))
        v[self.variant_indices, :] = other
        x = spsolve_triangular(eye(self.A.shape[0]) - self.A, v)
        return x - np.sum(other[self.flip])

    def __rmatmul__(self, other: NDArray) -> NDArray:
        if other.shape[1] != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. " f"Inputs had size {other.shape} and {self.shape}."
            )

        v = np.zeros((other.shape[0], self.A.shape[1]))
        v[:, self.sample_indices] = other
        x = spsolve_triangular(eye(self.A.shape[1]) - self.A.T, v.T, lower=False)
        x = x[self.variant_indices]
        x[self.flip] = np.sum(other) - x[self.flip]
        return x.T

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.matmul:
            # Identify the position of `self` in inputs
            if inputs[0] is self:  # self is the left operand
                return self.__matmul__(inputs[1])
            elif inputs[1] is self:  # self is the right operand
                return self.__rmatmul__(inputs[0])
        return NotImplemented

    def __getitem__(self, key: tuple[slice, slice]) -> "Linarg":
        rows, cols = key
        row_indices = range(*rows.indices(self.shape[0]))
        col_indices = range(*cols.indices(self.shape[1]))
        return Linarg(
            self.A, self.sample_indices[row_indices], self.variant_indices[col_indices], self.flip[col_indices]
        )

    def copy(self) -> "Linarg":
        return Linarg(self.A.copy(), self.variant_indices.copy(), self.sample_indices.copy(), self.flip.copy())

    # def rows(self, idx: slice):
    #     row_indices = range(*idx.indices(self.shape[0]))
    #     n = len(row_indices)
    #     x = np.zeros((n, self.shape[0]))
    #     for i, index in enumerate(row_indices):
    #         x[i, self.sample_indices[index]] = 1
    #     return x * self.data

    def write(self, filename: str):
        # Write samples
        with open(filename + ".samples.txt", "w") as f_samples:
            for sample in self.sample_indices:
                f_samples.write(f"{sample + 1}\n")

        # Write variants
        with open(filename + ".mutations.txt", "w") as f_variants:
            for variant in self.variant_indices:
                f_variants.write(f"{variant + 1}\n")

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

    def make_triangular(self) -> NDArray:
        import networkx as nx

        G = nx.from_scipy_sparse_array(self.A, create_using=nx.DiGraph)
        order = nx.topological_sort(G.reverse(copy=False))
        order = np.asarray(list(order))

        self.A = self.A[order, :][:, order]
        inv_order = np.argsort(order)
        self.variant_indices = inv_order[self.variant_indices]
        self.sample_indices = inv_order[self.sample_indices]
        return order

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
