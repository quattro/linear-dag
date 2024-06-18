# lineararg.py
from dataclasses import dataclass

import bed_reader as br
import cyvcf2 as cv
import numpy as np

from numpy.typing import NDArray
from scipy.io import mmread, mmwrite
from scipy.sparse import block_diag, csc_matrix, csr_matrix, eye, hstack, triu, vstack
from scipy.sparse.linalg import spsolve_triangular

from .solve import spinv_triangular
from .trios import Trios


def _construct_A(genotypes: csc_matrix, ploidy: int = 1) -> csc_matrix:
    more_than_one_carrier = np.diff(genotypes.indptr) > 1
    for n in range(1, ploidy + 1):
        X_carrier = genotypes[:, more_than_one_carrier] >= n
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

    A = csc_matrix((data, indices, indptr))
    A = A[original_order, :][:, original_order]

    # Add back variants with <=1 carrier
    one_or_zero_carriers = np.where(more_than_one_carrier == 0)[0]
    more_than_one_carrier = np.where(more_than_one_carrier == 1)[0]
    variant_ordering = np.argsort(np.concatenate((more_than_one_carrier, one_or_zero_carriers)))
    A = csc_matrix(block_diag((A, eye(len(one_or_zero_carriers)))))
    A = A[variant_ordering, :][:, variant_ordering]

    A.eliminate_zeros()

    return A


@dataclass
class LinearARG:
    A: csr_matrix
    sample_indices: NDArray
    variant_indices: NDArray
    flip: NDArray

    @staticmethod
    def from_genotypes(genotypes: csc_matrix, ploidy: int = 1, flip: NDArray = None) -> "LinearARG":
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

        return LinearARG(A, samples_idx, variants_idx, flip)

    @staticmethod
    def from_file(filename: str) -> "LinearARG":
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
            header = next(f).split(",")
            assert header[-2:] == ["index", "flip\n"]
            for line in f:
                fields = line.split(",")
                variant_list.append(int(fields[-2]) - 1)
                flip_list.append(int(fields[-1]))

        # Read the matrix A
        A = csr_matrix(mmread(filename + ".mtx"))

        return LinearARG(
            A,
            np.asarray(sample_list, dtype=int),
            np.asarray(variant_list, dtype=int),
            np.asarray(flip_list, dtype=bool),
        )

    @staticmethod
    def from_plink(prefix: str) -> "LinearARG":
        # TODO: handle missing data
        with br.open_bed(f"{prefix}.bed") as bed:
            genotypes = bed.read_sparse(dtype="int8")

            return LinearARG.from_genotypes(genotypes)

    @staticmethod
    def from_vcf(path: str) -> "LinearARG":
        vcf = cv.VCF(path, gts012=True)
        data = []
        idxs = []
        ptrs = [0]
        # TODO: handle missing data
        for var in vcf:
            (idx,) = np.where(var.gt_types != 0)
            gts = var.gt_types[idx]
            data.append(gts)
            idxs.append(idx)
            ptrs.append(ptrs[-1] + len(idx))

        data = np.array(data)
        idxs = np.array(idxs)
        ptrs = np.array(ptrs)
        genotypes = csc_matrix((data, idxs, ptrs))

        return LinearARG.from_genotypes(genotypes)

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
        if other.ndim == 1:
            other = other.reshape(-1, 1)
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and {other.shape}."
            )

        v = np.zeros((self.A.shape[0], other.shape[1]))
        v[self.variant_indices, :] = (other.T * (-1) ** self.flip).T
        x = spsolve_triangular(eye(self.A.shape[0]) - self.A, v)
        return x[self.sample_indices] + np.sum(other[self.flip])

    def __rmatmul__(self, other: NDArray) -> NDArray:
        if other.ndim == 1:
            other = other.reshape(1, -1)
        if other.shape[1] != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. " f"Inputs had size {other.shape} and {self.shape}."
            )

        v = np.zeros((other.shape[0], self.A.shape[1]))
        v[:, self.sample_indices] = other
        x = spsolve_triangular(eye(self.A.shape[1]) - self.A.T, v.T, lower=False)
        x = x[self.variant_indices]
        if np.any(self.flip):
            x[self.flip] = np.sum(other, axis=1) - x[self.flip]  # TODO what if other is a matrix?
        return x.T

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.matmul:
            # Identify the position of `self` in inputs
            if inputs[0] is self:  # self is the left operand
                return self.__matmul__(inputs[1])
            elif inputs[1] is self:  # self is the right operand
                return self.__rmatmul__(inputs[0])
        return NotImplemented

    def __getitem__(self, key: tuple[slice, slice]) -> "LinearARG":
        rows, cols = key
        row_indices = range(*rows.indices(self.shape[0]))
        col_indices = range(*cols.indices(self.shape[1]))
        return LinearARG(
            self.A, self.sample_indices[row_indices], self.variant_indices[col_indices], self.flip[col_indices]
        )

    def copy(self) -> "LinearARG":
        return LinearARG(self.A.copy(), self.sample_indices.copy(), self.variant_indices.copy(), self.flip.copy())

    # def rows(self, idx: slice):
    #     row_indices = range(*idx.indices(self.shape[0]))
    #     n = len(row_indices)
    #     x = np.zeros((n, self.shape[0]))
    #     for i, index in enumerate(row_indices):
    #         x[i, self.sample_indices[index]] = 1
    #     return x * self.data

    def write(self, filename_prefix: str, variant_metadata: dict = None) -> None:
        """
        Writes LinearARG to a trio of files, filename_prefix + ['.mtx', '.samples.txt', '.mutations.txt'].
        :param filename_prefix: where to save.
        :param variant_metadata: dictionary of lists containing metadata to be saved in *.mutations.txt. Fields
        will be saved with keys as column headers and values as column values. Length of each list should equal number
        of variants.
        """
        if variant_metadata is None:
            variant_metadata = {"variant": list(range(1, len(self.variant_indices) + 1))}
        # Write samples
        with open(filename_prefix + ".samples.txt", "w") as f_samples:
            f_samples.write("index\n")
            for sample in self.sample_indices:
                f_samples.write(f"{sample + 1}\n")

        # Write variants
        with open(filename_prefix + ".mutations.txt", "w") as f_variants:
            for key in variant_metadata.keys():
                f_variants.write(f"{key},")
            f_variants.write("index,flip\n")
            for line in zip(self.variant_indices, self.flip, *variant_metadata.values()):
                index, flip, variant = line[0], line[1], line[2:]
                for value in variant:
                    f_variants.write(f"{value},")
                f_variants.write(f"{index + 1},{int(flip)}\n")

        # Write the matrix A
        mmwrite(filename_prefix + ".mtx", self.A)

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

    def unweight_slow(self) -> "LinearARG":
        M = self.A
        num_nodes = len(M.indptr) - 1
        index_counter = len(M.indices)
        new_indptr = np.concatenate((M.indptr, np.empty_like(M.indices, dtype=np.uintc)))
        new_indices = np.concatenate((M.indices.copy(), np.empty_like(M.indices, dtype=np.intc)))
        new_data = np.concatenate((M.data.copy(), np.empty_like(M.indices, dtype=np.intc)))
        new_nodes = dict()

        for entry in range(len(M.indices)):
            index, weight = M.indices[entry], M.data[entry]
            if weight == 1:
                continue
            if (index, weight) not in new_nodes:
                new_nodes[(index, weight)] = num_nodes
                new_indices[index_counter] = index
                new_data[index_counter] = weight
                num_nodes += 1
                index_counter += 1
                new_indptr[num_nodes] = index_counter

            new_indices[entry] = new_nodes[(index, weight)]
            new_data[entry] = 1

        M = csr_matrix(
            (new_data[:index_counter], new_indices[:index_counter], new_indptr[: num_nodes + 1]),
            shape=(num_nodes, num_nodes),
        )
        return LinearARG(M, self.sample_indices, self.variant_indices, self.flip)

    def unweight(self, handle_singletons_differently=False) -> "LinearARG":
        """
        Prepares an initial linear ARG for recombination finding by grouping out-edges by weight. If node u has
        multiple out-edges with weight k != 1, then a new node v is created with a single k-weighted edge (u,v)
        and a 1-weighted edge (v, w) for each neighbor w.
        :return:
        """
        from collections import defaultdict

        M = csc_matrix(self.A)
        num_nodes = len(M.indptr) - 1
        new_indptr = np.zeros(2 * num_nodes, dtype=np.uintc)
        new_indices = np.zeros(2 * len(M.indices), dtype=np.uintc)
        new_data = np.zeros(2 * len(M.indices), dtype=np.intc)
        original_nodes = np.zeros(num_nodes, dtype=np.uintc)
        element_index = 0
        node_index = 0
        for u in range(num_nodes):
            original_nodes[u] = node_index

            # Collect neighbors of i by weight
            weight_to_neighbors = defaultdict(list)
            for j in range(M.indptr[u], M.indptr[u + 1]):
                weight_to_neighbors[M.data[j]].append(M.indices[j])

            # Separately handle edges of weight 1, which do not produce a new node
            for neighbor in weight_to_neighbors[1]:
                new_data[element_index] = 1
                new_indices[element_index] = neighbor
                element_index += 1

            out_degree = len(weight_to_neighbors) + len(weight_to_neighbors[1]) - 1
            new_indptr[node_index + 1] = new_indptr[node_index] + out_degree
            node_index += 1

            # Create out-edges of u
            for edge_weight, neighbors_with_weight in weight_to_neighbors.items():
                if edge_weight == 1:
                    continue
                if handle_singletons_differently and len(neighbors_with_weight) == 1:
                    new_indices[element_index] = neighbors_with_weight[0]
                else:
                    # Create a new node
                    new_indices[element_index] = node_index
                    out_degree = len(neighbors_with_weight)
                    new_indptr[node_index + 1] = new_indptr[node_index] + out_degree
                    node_index += 1

                new_data[element_index] = edge_weight
                element_index += 1

            # Create out-edges of newly created nodes
            for edge_weight, neighbors_with_weight in weight_to_neighbors.items():
                if edge_weight == 1:
                    continue
                if handle_singletons_differently and len(neighbors_with_weight) == 1:
                    continue

                for neighbor in neighbors_with_weight:
                    new_data[element_index] = 1
                    new_indices[element_index] = neighbor
                    element_index += 1

        # new_indices are a combination of new and old node IDs, mostly old except for those with weight != 1
        new_indices[new_data == 1] = original_nodes[new_indices[new_data == 1]]

        # Reconstitute the adjacency matrix and convert back from CSC to CSR
        M = csc_matrix(
            (new_data[:element_index], new_indices[:element_index], new_indptr[: node_index + 1]),
            shape=(node_index, node_index),
        )
        M = csr_matrix(M)

        return LinearARG(M, original_nodes[self.sample_indices], original_nodes[self.variant_indices], self.flip)

    def make_triangular(self) -> "LinearARG":
        import networkx as nx  # TODO maybe re-implement without this

        G = nx.from_scipy_sparse_array(self.A, create_using=nx.DiGraph)
        order = nx.topological_sort(G.reverse(copy=False))
        order = np.asarray(list(order))
        inv_order = np.argsort(order)

        return LinearARG(
            self.A[order, :][:, order], inv_order[self.sample_indices], inv_order[self.variant_indices], self.flip
        )

    def find_recombinations(self, ranked: bool = False) -> "LinearARG":
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

        return LinearARG(A, self.sample_indices, self.variant_indices, self.flip)
