# lineararg.py
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve_triangular

from .brick_graph import BrickGraph
from .brick_graph_py import BrickGraphPy
from .data_structures import DiGraph
from .genotype import read_vcf
from .linear_arg_inference import (
    add_samples_to_linear_arg,
    add_singleton_variants,
    infer_brick_graph_using_containment,
    linearize_brick_graph_adjacency,
    remove_undirected_edges,
)
from .one_summed_cy import linearize_brick_graph
from .recombination import Recombination
from .solve import topological_sort
from .trios import Trios


@dataclass
class LinearARG:
    A: csr_matrix
    sample_indices: NDArray
    variant_indices: NDArray
    flip: NDArray

    @staticmethod
    def from_genotypes(
        genotypes: csc_matrix,
        variants_flipped_ref_alt: Optional[np.ndarray] = None,
        find_recombinations: bool = True,
        make_triangular: bool = True,
    ):
        """
        Infers a linear ARG from a genotype matrix.
        :param genotypes: CSC matrix of 0-1 valued, phased genotypes; rows = samples, cols = variants
        :param variants_flipped_ref_alt: 0-1 valued array indicating which columns of the genotype matrix have had their
        ref and alt alleles flipped
        :param find_recombinations: whether to condense the graph by inferring recombination nodes
        :param make_triangular: whether to re-order rows and columns such that the adjacency matrix is triangular
        :return: linear ARG instance
        """
        if variants_flipped_ref_alt is None:
            variants_flipped_ref_alt = np.zeros(genotypes.shape[1], dtype=bool)
        if len(variants_flipped_ref_alt) != genotypes.shape[1]:
            raise ValueError
        if type(genotypes) is not csc_matrix:
            raise TypeError

        brick_graph, samples_idx, variants_idx = BrickGraph.from_genotypes(genotypes)

        recom = Recombination.from_graph(brick_graph)
        if find_recombinations:
            recom.find_recombinations()

        linear_arg_adjacency_matrix = linearize_brick_graph(recom)

        result = LinearARG(linear_arg_adjacency_matrix, samples_idx, variants_idx, variants_flipped_ref_alt)

        if make_triangular:
            result = result.make_triangular()

        return result

    @staticmethod
    def from_genotypes_old(
        genotypes: csc_matrix,
        ploidy: int = 1,
        flip: Optional[NDArray] = None,
        brick_graph_method: str = "old",
        recombination_method: str = "none",
    ) -> "LinearARG":
        # Infer an initial brick graph
        brick_graph_closure: csr_matrix
        if brick_graph_method.lower() == "old":
            brick_graph_closure = infer_brick_graph_using_containment(genotypes, ploidy)
        elif brick_graph_method.lower() == "new_slow":
            assert ploidy == 1, "new brick graph method assumes haploid samples"
            brick_graph_closure = BrickGraphPy.from_genotypes(genotypes).to_csr()
        elif brick_graph_method.lower() == "new":
            raise NotImplementedError("use from_genotypes_new instead")
        elif brick_graph_method.lower() == "trivial":
            brick_graph_closure = csr_matrix(eye(genotypes.shape[1]))
        else:
            raise ValueError(f"Unknown brick graph method {brick_graph_method}")
        brick_graph_closure = remove_undirected_edges(brick_graph_closure)

        linear_arg_adjacency_matrix = linearize_brick_graph_adjacency(brick_graph_closure)
        if brick_graph_method.lower() == "old":
            linear_arg_adjacency_matrix = add_singleton_variants(genotypes, linear_arg_adjacency_matrix)
        linear_arg_adjacency_matrix = add_samples_to_linear_arg(genotypes, linear_arg_adjacency_matrix)

        n, m = genotypes.shape
        samples_idx = np.arange(n)
        variants_idx = np.arange(n, m + n)
        if flip is None:
            flip = np.zeros(len(variants_idx), dtype=bool)
        result = LinearARG(linear_arg_adjacency_matrix, samples_idx, variants_idx, flip)

        if recombination_method.lower() == "after":
            result = result.unweight()
            result = result.find_recombinations()

        return result

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
        import bed_reader as br

        # TODO: handle missing data
        with br.open_bed(f"{prefix}.bed") as bed:
            genotypes = bed.read_sparse(dtype="int8")
            return LinearARG.from_genotypes(genotypes)

    @staticmethod
    def from_vcf(path: str) -> "LinearARG":
        genotypes, is_flipped, _ = read_vcf(path)
        return LinearARG.from_genotypes(genotypes, is_flipped)

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
        # TODO make this work with syntax like linarg[:100,] (works with linarg[:100,:])
        rows, cols = key
        return LinearARG(self.A, self.sample_indices[rows], self.variant_indices[cols], self.flip[cols])

    def copy(self) -> "LinearARG":
        return LinearARG(self.A.copy(), self.sample_indices.copy(), self.variant_indices.copy(), self.flip.copy())

    # def rows(self, idx: slice):
    #     row_indices = range(*idx.indices(self.shape[0]))
    #     n = len(row_indices)
    #     x = np.zeros((n, self.shape[0]))
    #     for i, index in enumerate(row_indices):
    #         x[i, self.sample_indices[index]] = 1
    #     return x * self.data

    def write(
        self,
        filename_prefix: str,
        chrom: str,
        positions: list,
        refs: list,
        alts: list,
        sample_filename: Optional[str] = None,
        iids: Optional[list] = None,
    ) -> None:
        """
        Writes LinearARG data in PLINK2 format, optionally writes sample information.
        :param filename_prefix: The base path and prefix used for output files.
        :param chrom: The chromosome number or identifier where the variants are located, e.g. "chr1".
        :param positions: A list of integers representing the genomic positions of each variant.
        :param refs: A list of strings representing the reference alleles for each variant.
        :param alts: A list of strings representing the alternate alleles for each variant.
        :param sample_filename: Optional filename for writing sample data. If provided, a .psam file will be generated
        containing sample identifiers.
        :param iids: Optional list of individual identifiers corresponding to samples. If provided, these identifiers
        are used in the .psam file.
        :return: None
        """
        if sample_filename:
            with open(f"{sample_filename}.psam", "w") as f_samples:
                f_samples.write("#IID IDX\n")
                if iids is None:
                    iids = [f"sample_{idx}" for idx in range(self.shape[0])]
                for i, iid in enumerate(iids):
                    f_samples.write(f"{iid} {self.sample_indices[i]}\n")

        with open(f"{filename_prefix}.pvar", "w") as f_variants:
            f_variants.write("##fileformat=PVARv1.0\n")
            f_variants.write('##INFO=<ID=IDX,Number=1,Type=Integer,Description="Variant Index">\n')
            f_variants.write('##INFO=<ID=FLIP,Number=1,Type=Integer,Description="Flip Information">\n')
            f_variants.write("#CHROM POS ID REF ALT INFO\n")
            for idx, pos, ref, alt, flip in zip(self.variant_indices, positions, refs, alts, self.flip):
                f_variants.write(f"{chrom} {pos} . {ref} {alt} IDX={idx};FLIP={int(flip)}\n")

        from scipy.sparse import save_npz

        save_npz(f"{filename_prefix}.npz", self.A)

    @staticmethod
    def read(
        adjacency_matrix_file: str, variants_file: str, samples_file: str
    ) -> Tuple["LinearARG", List[str], List[str]]:
        """
        Reads LinearARG data from provided PLINK2 formatted files.

        :param adjacency_matrix_file: Filename for the .npz file containing the adjacency matrix.
        :param variants_file: Filename for the .pvar file containing variant data.
        :param samples_file: Filename for the .psam file containing sample IDs.
        :return: A tuple containing the LinearARG object, list of variant IDs, and list of IIDs.
        """

        def parse_info(info_str: str) -> Tuple[int, int]:
            """Parses the INFO string to extract IDX and FLIP values."""
            info_parts = info_str.split(";")
            info_dict = {part.split("=")[0]: int(part.split("=")[1]) for part in info_parts if "=" in part}
            return info_dict.get("IDX", -1), info_dict.get("FLIP", 0)

        # Load the adjacency matrix
        from scipy.sparse import load_npz

        A = load_npz(adjacency_matrix_file)

        # Read variant data
        variants = []
        variant_ids = []
        flips = []
        with open(variants_file, "r") as f_var:
            headers = {}
            for line in f_var:
                if line.startswith("##"):
                    continue
                if not headers:
                    headers = {key: idx for idx, key in enumerate(line.strip().split())}
                    continue
                parts = line.strip().split()
                chrom = parts[headers["#CHROM"]]
                pos = parts[headers["POS"]]
                ref = parts[headers["REF"]]
                alt = parts[headers["ALT"]]
                idx, flip = parse_info(parts[headers["INFO"]])
                variant_ids.append(f"{chrom}_{pos}_{ref}_{alt}")
                variants.append(idx)
                flips.append(flip)

        variant_indices = np.array(variants)
        flip = np.array(flips)

        # Read sample data
        iids = []
        indices = []
        with open(samples_file, "r") as f_samp:
            headers = {}
            for line in f_samp:
                parts = line.strip().split()
                if line.startswith("#"):
                    # Identify headers and their positions
                    if not headers:
                        headers = {key: idx for idx, key in enumerate(parts) if key in ("IID", "IDX")}
                    continue
                # Read data according to identified headers
                iid = parts[headers["IID"]] if "IID" in headers else None
                idx = int(parts[headers["IDX"]]) if "IDX" in headers else -1
                iids.append(iid)
                indices.append(idx)
        sample_indices = np.array(indices)

        linear_arg = LinearARG(A, sample_indices, variant_indices, flip)

        return linear_arg, variant_ids, iids

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
        new_indptr = np.zeros(4 * num_nodes, dtype=np.uintc)
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
        order = np.asarray(topological_sort(self.A))
        inv_order = np.argsort(order)

        return LinearARG(
            self.A[order, :][:, order], inv_order[self.sample_indices], inv_order[self.variant_indices], self.flip
        )

    def find_recombinations(self, method="old") -> "LinearARG":
        if method == "old":
            trio_list = Trios(5 * self.A.nnz)  # TODO what should n be?
            edges = self.A == 1
            trio_list.convert_matrix(edges.indices, edges.indptr)
            trio_list.find_recombinations()
            new_edges = np.asarray(trio_list.fill_edgelist())
        else:
            gr = DiGraph.from_csc(self.A.transpose())
            recom = Recombination.from_graph(gr)
            recom.find_recombinations()
            new_edges = np.array(recom.edge_list(), dtype=np.dtype("int", "int"))

        new_edges = np.hstack((new_edges, np.ones((new_edges.shape[0], 1))))

        edges_with_nonone_weight = np.zeros((np.sum(self.A.data != 1), 3), dtype=np.intc)
        counter = 0
        for i in range(self.A.shape[0]):
            for entry in range(self.A.indptr[i], self.A.indptr[i + 1]):
                if self.A.data[entry] == 1:
                    continue
                j = self.A.indices[entry]
                weight = self.A.data[entry]
                edges_with_nonone_weight[counter, :] = [j, i, weight]
                counter += 1
        new_edges = np.vstack((new_edges, edges_with_nonone_weight))

        num_nodes = np.max(new_edges[:, 0:2]).astype(int) + 1
        A = csr_matrix((new_edges[:, 2], (new_edges[:, 1], new_edges[:, 0])), shape=(num_nodes, num_nodes))

        return LinearARG(A, self.sample_indices, self.variant_indices, self.flip)
