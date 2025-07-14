# lineararg.py

from dataclasses import dataclass
from functools import cached_property
from os import PathLike
from typing import Optional, Union

import h5py

# import bed_reader as br
import numpy as np
import numpy.typing as npt
import polars as pl

from scipy.sparse import csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import aslinearoperator, LinearOperator, spsolve_triangular

from linear_dag.core.solve import get_nonunique_indices_csc
from linear_dag.genotype import read_vcf

from .data_structures import DiGraph
from .linear_arg_inference import linear_arg_from_genotypes
from .one_summed_cy import linearize_brick_graph
from .solve import (
    add_at,
    spsolve_backward_triangular,
    spsolve_backward_triangular_matmat,
    spsolve_forward_triangular,
    spsolve_forward_triangular_matmat,
    topological_sort,
)

@dataclass
class LinearARG(LinearOperator):
    A: csc_matrix  # samples must be in descending order starting from the final row/col
    variant_indices: npt.NDArray[np.int32]
    flip: npt.NDArray[np.bool_]
    n_samples: np.int32
    n_individuals: Optional[np.int32] = None
    variants: Optional[pl.LazyFrame] = None
    iids: Optional[pl.Series] = None
    nonunique_indices: Optional[npt.NDArray[np.int32]] = None
    sex: Optional[npt.NDArray[np.int32]] = None  # determines how individual_indices are handled

    @property
    def individual_indices(self):
        if self.n_individuals is None:
            raise ValueError("The linear ARG does not have individual nodes. Try running add_individual_nodes first.")
        return np.arange(self.A.shape[0] - self.n_individuals, self.A.shape[0], dtype=np.int32)

    @property
    def sample_indices(self):
        if self.n_individuals is None:
            return np.arange(self.A.shape[0] - 1, self.A.shape[0] - self.n_samples - 1, -1, dtype=np.int32)
        else:
            return np.arange(
                self.A.shape[0] - self.n_individuals - 1,
                self.A.shape[0] - self.n_individuals - self.n_samples - 1,
                -1,
                dtype=np.int32,
            )

    @staticmethod
    def from_genotypes(
        genotypes: csc_matrix,
        flip: npt.NDArray[np.bool_],
        variant_info: pl.DataFrame = None,
        iids: Optional[list] = None,
        find_recombinations: bool = True,
        sex: Optional[npt.NDArray[np.int32]] = None,
        verbosity: int = 0,
    ):
        """
        Infers a linear ARG from a genotype matrix.
        :param genotypes: CSC matrix of 0-1 valued, phased genotypes; rows = samples, cols = variants
        ref and alt alleles flipped
        :param variant_info: polars dataframe containing required variant information, or none
        :param iids: Optional list of individual IDs corresponding to genotype rows.
        :param find_recombinations: whether to condense the graph by inferring recombination nodes
        :param make_triangular: whether to re-order rows and columns such that the adjacency matrix is triangular
        :return: linear ARG instance
        """
        A, flip, variants_idx, samples_idx, variant_info = linear_arg_from_genotypes(
            genotypes, flip, variant_info, find_recombinations, verbosity
        )
        A_filt, variants_idx_reindexed, samples_idx_reindexed = remove_degree_zero_nodes(A, variants_idx, samples_idx)
        A_tri, variants_idx_tri = make_triangular(A_filt, variants_idx_reindexed, samples_idx_reindexed)
        linarg = LinearARG(
            A_tri, variants_idx_tri, flip, len(samples_idx), None, variant_info.lazy(), iids=pl.Series(iids)
        )
        linarg.calculate_nonunique_indices()
        return linarg

    @staticmethod
    def from_vcf(
        path: Union[str, PathLike],
        phased: bool = True,
        region: Optional[str] = None,
        include_samples: Optional[list] = None,
        flip_minor_alleles: bool = False,
        return_genotypes: bool = False,
        verbosity: int = 0,
        maf_filter: float = None,
        snps_only: bool = False,
    ) -> Union[tuple, "LinearARG"]:
        genotypes, flip, v_info, iids = read_vcf(
            path, phased, region, flip_minor_alleles, maf_filter=maf_filter, remove_indels=snps_only
        )
        if genotypes is None:
            raise ValueError("No valid variants found in VCF")

        if phased:
            iids = [id_ for id_ in iids for _ in range(2)]

        if include_samples:
            genotypes = genotypes[include_samples, :]
            iids = [iids[i] for i in include_samples]

        result = LinearARG.from_genotypes(genotypes, flip, v_info, iids=iids, verbosity=verbosity)

        return (result, genotypes) if return_genotypes else result

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

    @property
    def dtype(self):
        return self.A.dtype

    @property
    def mean_centered(self):
        """
        Returns a linear operator representing the mean-centered genotype matrix
        """
        mean = aslinearoperator(np.ones((self.shape[0], 1), dtype=np.float32)) @ aslinearoperator(
            self.allele_frequencies
        )
        return self - mean

    @property
    def normalized(self):
        """
        Returns a linear operator representing the normalized genotype matrix
        whose columns have mean zero and variance one
        """
        pq = self.allele_frequencies * (1 - self.allele_frequencies)
        pq[pq == 0] = 1
        return self.mean_centered @ aslinearoperator(diags(pq**-0.5))

    @cached_property
    def allele_frequencies(self):
        return (np.ones(self.shape[0], dtype=np.int32) @ self) / self.shape[0]

    def number_of_carriers(self, individuals_to_include=None):
        if self.n_individuals is None:
            raise ValueError("The linear ARG does not have individual nodes. Try running add_individual_nodes first.")
        v = np.zeros(self.A.shape[0], dtype=np.float64)
        if individuals_to_include is None:
            v[self.individual_indices] = np.ones(self.n_individuals)
        else:
            v[self.individual_indices[individuals_to_include]] = np.ones(len(individuals_to_include))
        spsolve_backward_triangular(self.A, v)
        v = v[self.variant_indices]
        if np.any(self.flip):
            if individuals_to_include is None:
                v[self.flip] = self.n_individuals - v[self.flip]
            else:
                v[self.flip] = len(individuals_to_include) - v[self.flip]
        return v.astype(np.int64)

    def remove_samples(self, iids_to_remove: npt.ArrayLike):
        sample_mask = np.isin(self.iids, iids_to_remove)
        sample_indices_to_remove = np.where(sample_mask)[0]
        iids_to_keep = np.array(self.iids)[~sample_mask]

        individual_mask = np.isin(list(dict.fromkeys(self.iids)), iids_to_remove)  # deduplicate but keep order
        individual_indices_to_remove = np.where(individual_mask)[0]

        if self.n_individuals is not None:
            nodes_to_remove = np.concatenate(
                [self.sample_indices[sample_indices_to_remove], self.individual_indices[individual_indices_to_remove]]
            )
        else:
            nodes_to_remove = self.sample_indices[sample_indices_to_remove]

        A = self.A
        nodes_to_keep = np.setdiff1d(np.arange(A.shape[0]), nodes_to_remove)
        A = A.tocsr()
        A = A[nodes_to_keep, :].tocsc()  # efficient row slice, then convert back to CSC
        A = A[:, nodes_to_keep]

        linarg_samples_removed = LinearARG(A, self.variant_indices, self.flip, len(iids_to_keep))
        linarg_samples_removed.iids = pl.Series(iids_to_keep).cast(pl.Int64)

        if self.n_individuals is not None:
            linarg_samples_removed.n_individuals = self.n_individuals - len(iids_to_remove)
        if self.variants is not None:
            linarg_samples_removed.variants = self.variants
        if self.sex is not None:
            linarg_samples_removed.sex = self.sex[~individual_mask]
        return linarg_samples_removed

    def __str__(self):
        return f"A: shape {self.A.shape}, nonzeros {self.A.nnz}"

    def _matmat_scipy(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.ndim == 1:
            other = other.reshape(-1, 1)
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and {other.shape}."
            )

        v = np.zeros((self.A.shape[0], other.shape[1]), dtype=other.dtype)
        temp = (other.T * (-1) ** self.flip).T
        np.add.at(v, self.variant_indices, temp)
        x = spsolve_triangular(eye(self.A.shape[0]) - self.A, v)
        return x[self.sample_indices] + np.sum(other[self.flip], axis=0)

    def _matmat(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.ndim == 1:
            other = other.reshape(-1, 1)
        if other.shape[0] != self.shape[1]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {self.shape} and {other.shape}."
            )

        self.calculate_nonunique_indices()
        v = np.zeros((other.shape[1], self.num_nonunique_indices), dtype=other.dtype, order="F")

        if any(self.flip):
            temp = (other.T * (-1) ** self.flip.reshape(1, -1)).astype(other.dtype)
        else:
            temp = other.T

        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        add_at(v, variant_nonunique_indices, temp)
        spsolve_forward_triangular_matmat(self.A, v, self.nonunique_indices, int(self.sample_indices[-1]))
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        return v[:, sample_nonunique_indices].T + np.sum(other[self.flip], axis=0)

    def _rmatmat(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.ndim == 1:
            other = other.reshape(1, -1)
        if other.shape[0] != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {other.shape} and {self.shape}."
            )

        self.calculate_nonunique_indices()
        v = np.zeros((other.shape[1], self.num_nonunique_indices), dtype=other.dtype, order="F")
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        v[:, sample_nonunique_indices] = other.T
        spsolve_backward_triangular_matmat(self.A, v, self.nonunique_indices, int(self.sample_indices[-1]))
        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        v = v[:, variant_nonunique_indices]
        if np.any(self.flip):
            v[:, self.flip] = np.sum(other, axis=0)[:, np.newaxis] - v[:, self.flip]
        return v.T

    def _rmatmat_scipy(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.ndim == 1:
            other = other.reshape(1, -1)
        if other.shape[1] != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. Inputs had size {other.shape} and {self.shape}."
            )
        v = np.zeros((other.shape[0], self.A.shape[1]), dtype=other.dtype)
        v[:, self.sample_indices] = other

        x = spsolve_triangular(eye(self.A.shape[1]) - self.A.T, v.T, lower=False)

        x = x[self.variant_indices]
        if np.any(self.flip):
            raise NotImplementedError("_rmatmat_scipy not implemented for flipped variants")
            # x[self.flip] = np.sum(other, axis=0) - x[self.flip]  # TODO
        return x

    def _matvec(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.shape != (self.shape[1],) and other.shape != (self.shape[1], 1):
            raise ValueError(
                (
                    "Incorrect dimensions for matrix-vector multiplication.",
                    f" Inputs had size {self.shape} and {other.shape}.",
                )
            )

        v = np.zeros(self.A.shape[0], dtype=np.float64)
        temp = other.ravel().astype(np.float64) * ((-1) ** self.flip.ravel())
        np.add.at(v, self.variant_indices, temp)  # handles duplicate variant indices
        spsolve_forward_triangular(self.A, v)
        result = np.asarray(v[self.sample_indices]) + np.sum(other[self.flip])
        return result if other.ndim == 1 else result.reshape(-1, 1)

    def _rmatvec(self, other: npt.ArrayLike, individual=False) -> npt.NDArray[np.number]:
        if other.shape != (self.shape[0],) and other.shape != (self.shape[0], 1):
            raise ValueError(
                (
                    "Incorrect dimensions for vector-matrix multiplication.",
                    f" Inputs had size {other.shape} and {self.shape}.",
                )
            )
        v = np.zeros(self.A.shape[0], dtype=np.float64)
        v[self.sample_indices] = other.ravel().astype(np.float64)
        spsolve_backward_triangular(self.A, v)
        v = v[self.variant_indices]
        if np.any(self.flip):
            v[self.flip] = np.sum(other) - v[self.flip]
        return v if other.ndim == 1 else v.reshape(-1, 1)

    def copy(self) -> "LinearARG":
        # return LinearARG(self.A.copy(), self.sample_indices.copy(), self.variants.copy())
        pass

    def write(self, prefix: Union[str, PathLike], block_info: Optional[dict] = None, compression_option: str = "gzip"):
        """Writes LinearARG to disk.
        :param prefix: The base path and prefix used for output files.
        :param block_info: Optional dictionary containing:
            - 'chrom': Chromosome number
            - 'start': Start position
            - 'end': End position
        :param compression_option: Compression option for writing.
        :return: None
        """

        # write out DAG info
        if not str(prefix).endswith(".h5"):
            prefix = str(prefix) + ".h5"
        with h5py.File(prefix, "a") as f:
            if block_info:
                block_name = f"{block_info['chrom']}_{block_info['start']}_{block_info['end']}"
                destination = f.create_group(block_name)
                destination.attrs["chrom"] = block_info["chrom"]
                destination.attrs["start"] = block_info["start"]
                destination.attrs["end"] = block_info["end"]
            else:
                destination = f

            destination.attrs["n"] = self.A.shape[0]
            destination.attrs["n_samples"] = self.n_samples
            destination.attrs["n_variants"] = self.shape[1]
            destination.attrs["n_entries"] = self.nnz

            destination.create_dataset("indptr", data=self.A.indptr, compression=compression_option, shuffle=True)
            destination.create_dataset("indices", data=self.A.indices, compression=compression_option, shuffle=True)
            destination.create_dataset("data", data=self.A.data, compression=compression_option, shuffle=True)
            destination.create_dataset(
                "variant_indices", data=self.variant_indices, compression=compression_option, shuffle=True
            )
            destination.create_dataset("flip", data=self.flip, compression=compression_option, shuffle=True)

            if self.nonunique_indices is not None:
                destination.create_dataset(
                    "nonunique_indices", data=self.nonunique_indices, compression=compression_option, shuffle=True
                )
            if self.iids is not None and "iids" not in f.keys():
                str_iids = np.array(self.iids, dtype=h5py.string_dtype(encoding="utf-8"))
                f.create_dataset("iids", data=str_iids, compression=compression_option, shuffle=True)
            if self.n_individuals is not None:
                destination.attrs["n_individuals"] = self.n_individuals
            if self.variants is not None:
                for field in ["CHROM", "POS", "ID", "REF", "ALT"]:
                    variant_info = self.variants.collect()
                    if field == "POS":
                        destination.create_dataset(
                            field,
                            data=np.array(variant_info[field]).astype(int),
                            compression=compression_option,
                            shuffle=True,
                        )
                    else:
                        destination.create_dataset(
                            field,
                            data=np.array(variant_info[field]).astype("S"),
                            compression=compression_option,
                            shuffle=True,
                        )
        return

    @staticmethod
    def read(
        h5_fname: Union[str, PathLike],
        block: Optional[str] = None,
        load_metadata:bool = False,
    ) -> "LinearARG":
        """Reads LinearARG data from provided PLINK2 formatted files.

        :param h5_fname: The base path and prefix of the PLINK files.
        :return: A LinearARG object.
        """
        with h5py.File(h5_fname, "r") as file:
            f = file[block] if block else file
            A = csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=(f.attrs["n"], f.attrs["n"]))
            variant_indices = f["variant_indices"][:]
            flip = f["flip"][:]
            n_samples = f.attrs["n_samples"]
            n_individuals = f.attrs.get("n_individuals", None)
            nonunique_indices = f["nonunique_indices"][:] if "nonunique_indices" in f else None
            iids = f.get("iids")

            if load_metadata:
                v_dict = {field: f[field][:].astype(str) for field in ["CHROM", "POS", "ID", "REF", "ALT"]}
                v_info = pl.DataFrame(v_dict).with_columns([
                            pl.col("POS").cast(pl.Int32),
                ]).lazy()
            else:
                v_info = None

        return LinearARG(A, variant_indices, flip, n_samples, n_individuals, v_info, iids, nonunique_indices)

    def calculate_nonunique_indices(self) -> None:
        """Calculates and stores non-unique indices to facilitate memory-efficient matmat and rmatmat operations."""
        if self.nonunique_indices is None:
            if self.n_individuals is None:
                individual_indices = None
            else:
                individual_indices = self.individual_indices

            self.nonunique_indices = get_nonunique_indices_csc(
                self.A.indices,
                self.A.indptr,
                self.sample_indices,
                self.variant_indices,
                individual_indices=individual_indices,
            )
            self.nonunique_indices = np.asarray(self.nonunique_indices)
            print(f"Non-unique indices: {self.num_nonunique_indices} vs. {self.A.shape[0]}")

    @cached_property
    def num_nonunique_indices(self) -> Optional[int]:
        if self.nonunique_indices is None:
            return None
        return np.max(self.nonunique_indices) + 1

    def add_individual_nodes(self, sex: npt.NDArray[np.uint] = None) -> "LinearARG":
        """Creates a new LinearARG with indviduals added as nodes."""
        A, individual_indices = add_individuals_to_graph(self.A, self.sample_indices, sex=sex)
        individuals_graph = DiGraph.from_csr(A)  # edges are defined the other way around
        A = csc_matrix(linearize_brick_graph(individuals_graph))
        linarg = LinearARG(
            A,
            self.variant_indices,
            self.flip,
            len(self.sample_indices),
            len(individual_indices),
            self.variants,
            iids=self.iids,
            nonunique_indices=None,
            sex=self.sex,
        )
        linarg.calculate_nonunique_indices()

        return linarg


def list_blocks(h5_fname: Union[str, PathLike]) -> pl.DataFrame:
    """
    Lists the blocks (groups) and their attributes stored in an HDF5 file.
    Handles cases where attributes might be at the root if no blocks exist.

    Args:
        h5_fname: Path to the HDF5 file.

    Returns:
        A Polars DataFrame where each row corresponds to a block (group)
        or the root dataset in the HDF5 file. Columns include the block name
        (or 'root') and its attributes.
    """
    if not str(h5_fname).endswith(".h5"):
        h5_fname = str(h5_fname) + ".h5"
    block_data = []

    def parse_block_name(block_name):
        chrom, start, _ = block_name.split("_")
        return (int(chrom), int(start))

    with h5py.File(h5_fname, "r") as f:
        block_names = [b for b in list(f.keys()) if isinstance(f[b], h5py.Group) and b != "iids"]
        block_names = sorted(block_names, key=parse_block_name)

        if not block_names:
            return None
        else:
            for block_name in block_names:
                group = f[block_name]
                attrs = group.attrs
                block_info = {
                    "block_name": block_name,
                    "chrom": attrs.get("chrom"),
                    "start": attrs.get("start"),
                    "end": attrs.get("end"),
                    "n": attrs.get("n"),
                    "n_samples": attrs.get("n_samples"),
                    "n_variants": attrs.get("n_variants"),
                    "n_entries": attrs.get("n_entries"),
                }
                block_data.append(block_info)

    return pl.DataFrame(block_data)


def load_block_metadata(h5_fname, block_metadata):
    block_names = block_metadata.get_column("block_name").to_list()
    lazyframes = []
    with h5py.File(h5_fname, "r") as f:
        for block_name in block_names:
            block = f[block_name]
            v_dict = {field: block[field][:].astype(str) for field in ["CHROM", "POS", "ID", "REF", "ALT"]}
            v_info = (
                pl.DataFrame(v_dict)
                .with_columns(
                    [
                        pl.col("POS").cast(pl.Int32),
                    ]
                )
                .lazy()
            )
            lazyframes.append(v_info)
    return pl.concat(lazyframes)


def add_individuals_to_graph(
    A: csc_matrix,
    samples_idx: npt.NDArray[np.uint],
    sex: npt.NDArray[np.uint] = None,
) -> tuple:
    """
    Add individuals to the graph. Assumes that individuals are comprised of adjacent haplotypes in samples_idx.
    If sex is None, assumes that
    all individuals are diploid. Otherwise will only assign males a single haplotype.
    """
    A_csr = csr_matrix(A)
    indices_list = []
    indptr_list = [A_csr.indptr[-1]]

    if sex is None:
        haplotype_counts = np.full(len(samples_idx) // 2, 2, dtype=int)
    else:
        haplotype_counts = np.array([1 if s == 1 else 2 for s in sex], dtype=int)

    haplotype_offsets = np.concatenate(([0], np.cumsum(haplotype_counts)))

    for i in range(len(haplotype_counts)):
        start = haplotype_offsets[i]
        end = haplotype_offsets[i + 1]
        haps = samples_idx[start:end]
        indices_list.append(haps)
        indptr_list.append(indptr_list[-1] + len(haps))

    indices = np.concatenate([A_csr.indices] + indices_list)
    indptr = np.array(list(A_csr.indptr) + indptr_list[1:], dtype=np.int32)
    data = np.ones(len(indices), dtype=np.int32)

    n_nodes = len(indptr) - 1
    A_updated = csc_matrix(csr_matrix((data, indices, indptr), shape=(n_nodes, n_nodes)))
    individual_indices = np.arange(A.shape[0], A_updated.shape[0], dtype=np.int32)

    return A_updated, individual_indices


def remove_degree_zero_nodes(
    A: csc_matrix, variant_indices: npt.NDArray[np.uint], sample_indices: npt.NDArray[np.uint]
) -> tuple:
    """
    Removes degree-zero recombination nodes from the graph, while ensuring all nodes
    in variant_indices and sample_indices are retained. Returns the
    filtered graph and reindexed variant/sample indices.
    """
    node_degree = A.getnnz(axis=0) + A.getnnz(axis=1)
    nonzero_indices = set(np.where(node_degree > 0)[0])
    required_indices = set(variant_indices).union(sample_indices)
    indices_to_keep = np.array(sorted(nonzero_indices.union(required_indices)), dtype=int)

    index_map = -np.ones(A.shape[0], dtype=int)
    index_map[indices_to_keep] = np.arange(len(indices_to_keep))

    A_filt = A[indices_to_keep, :][:, indices_to_keep]
    variant_indices_reindexed = index_map[variant_indices]
    sample_indices_reindexed = index_map[sample_indices]

    return A_filt, variant_indices_reindexed, sample_indices_reindexed


def make_triangular(
    A: csc_matrix, variant_indices: npt.NDArray[np.uint], sample_indices: npt.NDArray[np.uint]
) -> tuple:
    """
    Triangularizes A by putting nodes in topological order (parents before children) such that
    sample/leaf nodes are in reverse order starting from the final row/column of the returned csc_matrix.
    Additionally, variant_indices are reindexed with respect to this new node ordering.
    """
    A_csr = csr_matrix(A)
    order = np.asarray(topological_sort(A_csr, nodes_to_ignore=set(sample_indices)))[: -len(sample_indices)]
    order = np.append(order, sample_indices[::-1])
    inv_order = np.argsort(order).astype(np.int32)

    A_triangular = A[order, :][:, order]
    variant_indices_reordered = inv_order[variant_indices]

    return A_triangular, variant_indices_reordered
