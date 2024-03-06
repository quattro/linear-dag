# linarg.py

import numpy as np

from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix, eye, hstack, triu, vstack

from .solve import spinv_triangular
from .trios import Trios  # Assuming Trios is accessible as a Python module


class Linarg:
    def __init__(self, genotype_matrix_mtx: str = None):
        self.trio_list: Trios = None
        self.genotypes: csc_matrix = csc_matrix((0, 0))
        self.A: csc_matrix = csc_matrix((0, 0))
        self.A_haplo: csc_matrix = csc_matrix((0, 0))
        self.variants: np.ndarray = np.array([])
        self.samples: np.ndarray = np.array([])
        self.flip: np.ndarray = np.array([])
        self.ploidy: int = 1
        self.af: np.ndarray = np.array([])

        if genotype_matrix_mtx is not None:
            self.genotypes = csc_matrix(mmread(genotype_matrix_mtx))
            self.samples = np.arange(self.genotypes.shape[0])
            self.variants = np.arange(self.genotypes.shape[1])
            self.flip = np.zeros(self.genotypes.shape[1])
            self.ploidy = np.max(self.genotypes)

    def print(self):
        print(f"genotypes: shape {self.genotypes.shape}, nonzeros {self.genotypes.nnz}")
        print(f"A: shape {self.A.shape}, nonzeros {self.A.nnz}")
        print(f"A_haplo: shape {self.A_haplo.shape}, nonzeros {self.A_haplo.nnz}")
        if self.trio_list is not None:
            tup = self.trio_list.get_num()
            print(f"trio_list properties: n {tup[0]}, num_cliques {tup[1]}")
            print(f"num_nodes {tup[2]}, num_trios {tup[3]}, num_edges {tup[4]}")

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

    def compute_af(self):
        column_sums = self.genotypes.sum(axis=0)
        # Convert column sums to a flat array (necessary for sparse matrices)
        column_sums = np.ravel(column_sums)
        self.af = column_sums / self.genotypes.shape[0] / self.ploidy
        return self.af

    def apply_maf_threshold(self, threshold: float = 0):
        # Calculate allele frequencies
        self.compute_af()

        # Calculate MAF (ensure p is a flat array for element-wise operations)
        maf = np.minimum(self.af, 1 - self.af)

        # Find indices where MAF is above the threshold
        maf_above_threshold_indices = np.where(maf > threshold)[0]

        # Keep only the columns of self.genotypes where MAF is above the threshold
        self.genotypes = self.genotypes[:, maf_above_threshold_indices]

        # Update the MAF array to reflect the columns kept
        self.af = self.af[maf_above_threshold_indices]

    def flip_alleles(self):
        # Calculate allele frequencies
        self.compute_af()
        self.flip = self.af > 0.5

        # list-of-columns format
        genotypes_lil = self.genotypes.T.tolil()

        for i in range(genotypes_lil.shape[0]):
            if self.flip[i]:
                self.af[i] = 1 - self.af[i]

                # Convert the row to dense, flip the alleles, and assign it back
                row_dense = genotypes_lil[i, :].toarray()
                flipped_row_dense = self.ploidy - row_dense
                genotypes_lil[i, :] = flipped_row_dense

        self.genotypes = genotypes_lil.T.tocsc()

    def compute_haplotypes(self):
        Xh = None
        for n in range(1, int(self.ploidy) + 1):
            X_carrier = self.genotypes >= n
            X_carrier = X_carrier.astype(np.int32)
            R_carrier = X_carrier.transpose().dot(X_carrier)

            temp = R_carrier.copy()
            for i in range(R_carrier.shape[0]):
                row_data = R_carrier.data[R_carrier.indptr[i] : R_carrier.indptr[i + 1]]
                temp.data[temp.indptr[i] : temp.indptr[i + 1]] = row_data >= R_carrier[i, i]

            if n == 1:
                Xh = temp
            else:
                Xh = Xh.multiply(temp)

        Xh.eliminate_zeros()
        Xh = Xh.tocsc()
        ties = Xh.multiply(Xh.transpose())
        Xh = Xh - triu(ties, k=1)
        Xh.eliminate_zeros()

        return Xh

    def invert_haplotype_matrix(self, X_haplo: csc_matrix):
        row_counts = np.diff(X_haplo.indptr)
        triangular_order = np.argsort(-row_counts)
        original_order = np.argsort(triangular_order)
        X_haplo = X_haplo[triangular_order, :][:, triangular_order].astype(np.int32)
        X_haplo = X_haplo.tocsr()
        X_haplo.sort_indices()
        indptr, indices, data = spinv_triangular(X_haplo.indptr, X_haplo.indices, X_haplo.data)
        self.A_haplo = csr_matrix((data, indices, indptr))
        self.A_haplo = self.A_haplo[original_order, :][:, original_order]
        self.A_haplo.eliminate_zeros()

    def form_initial_linarg(self):
        Xh = self.compute_haplotypes()
        # A = spinv_triangular(Xh_reordered)
        # lu = splu(Xh)
        # self.A_haplo = lu.solve_sparse(csc_matrix(b))
        # self.A_haplo = self.A_haplo.astype(np.int32)
        # self.A_haplo.eliminate_zeros()
        self.invert_haplotype_matrix(Xh)

        n, m = self.genotypes.shape

        # Fit samples to the matrix
        A_sample = self.genotypes @ self.A_haplo
        self.A_haplo = eye(m) - self.A_haplo

        # Concatenate
        zeros_matrix = csr_matrix((m + n, n))
        vertical_stack = vstack([A_sample, self.A_haplo])
        self.A = hstack([zeros_matrix, vertical_stack])
        self.A = self.A.astype(np.int32)
        self.A.eliminate_zeros()
        self.A = csr_matrix(self.A)

        # Indices in A of the samples and variants
        self.samples = range(self.genotypes.shape[0])
        self.variants = range(self.genotypes.shape[0], self.genotypes.shape[1] + self.genotypes.shape[0])

    def create_triolist(self):
        self.trio_list = Trios(2 * self.A.nnz)  # TODO what should n be?
        self.trio_list.convert_matrix(self.A.data, self.A.indices, self.A.indptr, self.A.indptr.shape[0])
        self.trio_list.check_properties(-1)

    def find_recombinations(self):
        if self.trio_list is None:
            self.create_triolist()

        c = self.trio_list.max_clique()
        while c > 0:
            self.trio_list.factor_clique(c)
            c = self.trio_list.max_clique()

        # Verify the trio list remains valid
        self.trio_list.check_properties(-1)

        # Compute new edge list
        edges = self.trio_list.fill_edgelist()
        num_nodes = np.max(edges[:, 0:2]) + 1
        self.A = csc_matrix((edges[:, 2], (edges[:, 1], edges[:, 0])), shape=(num_nodes, num_nodes))

        return edges
