# lineararg.py
import gzip

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from os import linesep, PathLike
from typing import ClassVar, Optional, Union
import time

# import bed_reader as br
import numpy as np
import numpy.typing as npt
import polars as pl

import h5py
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix, diags, eye, load_npz, save_npz
from scipy.sparse.linalg import aslinearoperator, LinearOperator, spsolve_triangular

from linear_dag.genotype import read_vcf
from linear_dag.core.solve import get_nonunique_indices_csc
from .linear_arg_inference import linear_arg_from_genotypes
from .solve import topological_sort, \
            spsolve_forward_triangular_matmat, \
            spsolve_backward_triangular_matmat,\
            spsolve_forward_triangular,\
            spsolve_backward_triangular,\
            add_at


@dataclass
class VariantInfo:
    """Metadata about variants represented in the linear dag.

    **Attributes**
    """

    table: pl.DataFrame
    req_fields: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT"]

    def __post_init__(self):
        for req_col in self.req_fields:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in variant table")

    def copy(self):
        return VariantInfo(self.table.clone())

    def __getitem__(self, key):
        return VariantInfo(self.table[key])

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "VariantInfo":
        if path is None:
            raise ValueError("path argument cannot be None")
    
        var_table = defaultdict(list)
        with h5py.File(path, "r") as f:
            for field in cls.req_fields:
                if field == 'POS':
                    var_table[field] = f[field][:].astype(int) # cast as int to save memory
                else:
                    var_table[field] = f[field][:].astype(str)
        var_table = pl.DataFrame(var_table)
        return cls(var_table) # return class instance

    # def write(self, path: Union[str, PathLike]):
    #     open_f = gzip.open if str(path).endswith(".gz") else open
    #     with open_f(path, "wt") as pvar_file:
    #         pvar_file.write(f"##fileformat=PVARv1.0{linesep}")
    #         pvar_file.write(f'##INFO=<ID=IDX,Number=1,Type=Integer,Description="Variant Index">{linesep}')
    #         pvar_file.write(f'##INFO=<ID=FLIP,Number=0,Type=Flag,Description="Flip Information">{linesep}')
    #         pvar_file.write("\t".join([f"#{self.req_fields[0]}"] + self.req_fields[1:]) + linesep)
    #         pvar_file.flush()
    #         pvar_file.write(self.table.write_csv(include_header=False, separator="\t"))
    #     return

    # @classmethod
    # def from_open_bed(
    #     cls, bed: br.open_bed, indices: Optional[npt.ArrayLike] = None, is_flipped: Optional[npt.ArrayLike] = None
    # ):
    #     # doesn't really follow conventions for a class name...
    #     # req_fields: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", idx_field, flip_field]
    #     df = dict()
    #     df["CHROM"] = bed.chromosome
    #     df["POS"] = bed.bp_position
    #     df["ID"] = bed.sid
    #     df["REF"] = bed.allele_1
    #     df["ALT"] = bed.allele_2
    #     if indices is None:
    #         df[cls.idx_col] = -1 * np.ones(len(bed.bp_position), dtype=int)
    #     else:
    #         if len(indices) != len(bed.bp_position):
    #             raise ValueError("Length of indices does not match number of variants")
    #         df[cls.idx_col] = indices
    #     if is_flipped is None:
    #         df[cls.flip_field] = np.zeros(len(bed.iid)).astype(bool)
    #     else:
    #         if len(is_flipped) != len(bed.bp_position):
    #             raise ValueError("Length of is_flipped flags does not match number of variants")
    #         df[cls.flip_field] = is_flipped

    #     return cls(pl.DataFrame(df))


@dataclass
class LinearARG(LinearOperator):
    A: csc_matrix # samples must be in descending order starting from the final row/col
    variant_indices: npt.NDArray[np.int32]
    flip: npt.NDArray[np.bool_]
    n_samples: np.int32
    variants: VariantInfo = None
    nonunique_indices: Optional[npt.NDArray[np.int32]] = None
    
    @property
    def sample_indices(self):
        return np.arange(self.A.shape[0]-1, self.A.shape[0] - self.n_samples - 1, -1, dtype=np.int32)

    @staticmethod
    def from_genotypes(
        genotypes: csc_matrix,
        flip: npt.NDArray[np.bool_],
        variant_info: pl.DataFrame = None,
        find_recombinations: bool = True,
        verbosity: int = 0,
    ):
        """
        Infers a linear ARG from a genotype matrix.
        :param genotypes: CSC matrix of 0-1 valued, phased genotypes; rows = samples, cols = variants
        ref and alt alleles flipped
        :param variant_info: polars dataframe containing required variant information, or none
        :param find_recombinations: whether to condense the graph by inferring recombination nodes
        :param make_triangular: whether to re-order rows and columns such that the adjacency matrix is triangular
        :return: linear ARG instance
        """
        A, flip, variants_idx, samples_idx, variant_info = linear_arg_from_genotypes(
            genotypes, variant_info, find_recombinations, verbosity
        )
        A_filt, variants_idx_reindexed, samples_idx_reindexed  = remove_degree_zero_nodes(A, variants_idx, samples_idx)
        A_tri, variants_idx_tri = make_triangular(A_filt, variants_idx_reindexed, samples_idx_reindexed)
        linarg = LinearARG(A_tri, variants_idx_tri, flip, len(samples_idx), VariantInfo(variant_info))
        linarg.calculate_nonunique_indices()
        return linarg
        

    # @staticmethod
    # def from_plink(prefix: str) -> "LinearARG":
    #     import bed_reader as br

    #     with br.open_bed(f"{prefix}.bed") as bed:
    #         genotypes = bed.read_sparse(dtype="int8")

    #     larg = LinearARG.from_genotypes(genotypes)
    #     v_info = VariantInfo.from_open_bed(bed, larg.variant_indices)
    #     larg.variants = v_info
    #     return larg

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
        snps_only: bool = False
    ) -> Union[tuple, "LinearARG"]:
        genotypes, flip, v_info = read_vcf(path, phased, region, flip_minor_alleles,
                                    maf_filter=maf_filter, remove_indels=snps_only)
        if include_samples:
            genotypes = genotypes[include_samples, :]
        result = LinearARG.from_genotypes(genotypes, flip, v_info, verbosity=verbosity)
        return result, genotypes if return_genotypes else result

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
        mean = aslinearoperator(np.ones((self.shape[0], 1), dtype=np.float32)) @ aslinearoperator(self.allele_frequencies)
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
        v = np.zeros((other.shape[1], self.num_nonunique_indices), dtype=other.dtype, order='F')

        if any(self.flip):
            temp = (other.T * (-1) ** self.flip.reshape(1,-1)).astype(other.dtype)
        else:
            temp = other.T

        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        add_at(v, variant_nonunique_indices, temp)
        spsolve_forward_triangular_matmat(self.A, v, self.nonunique_indices)
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        return v[:, sample_nonunique_indices].T + np.sum(other[self.flip], axis=0)

    def _rmatmat(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.ndim == 1:
            other = other.reshape(1, -1)
        if other.shape[0] != self.shape[0]:
            raise ValueError(
                f"Incorrect dimensions for matrix multiplication. " f"Inputs had size {other.shape} and {self.shape}."
            )

        self.calculate_nonunique_indices()
        v = np.zeros((other.shape[1], self.num_nonunique_indices), dtype=other.dtype, order='F')   
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]   
        v[:, sample_nonunique_indices] = other.T
        spsolve_backward_triangular_matmat(self.A, v, self.nonunique_indices)
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
                f"Incorrect dimensions for matrix multiplication. " f"Inputs had size {other.shape} and {self.shape}."
            )
        v = np.zeros((other.shape[0], self.A.shape[1]), dtype=other.dtype)
        v[:, self.sample_indices] = other
        
        x = spsolve_triangular(eye(self.A.shape[1]) - self.A.T, v.T, lower=False)
        
        x = x[self.variant_indices]
        if np.any(self.flip):
            x[self.flip] = np.sum(other, axis=0) - x[self.flip]  # TODO what if other is a matrix?
        return x

    def _matvec(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.shape != (self.shape[1],) and other.shape != (self.shape[1], 1):
            raise ValueError(
                f"Incorrect dimensions for matrix-vector multiplication. Inputs had size {self.shape} and {other.shape}."
            )

        v = np.zeros(self.A.shape[0], dtype=np.float64)
        temp = other.ravel().astype(np.float64) * ((-1) ** self.flip.ravel())
        np.add.at(v, self.variant_indices, temp)  # handles duplicate variant indices
        spsolve_forward_triangular(self.A, v)
        result = np.asarray(v[self.sample_indices]) + np.sum(other[self.flip])
        return result if other.ndim == 1 else result.reshape(-1, 1)

    def _rmatvec(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
        if other.shape != (self.shape[0],) and other.shape != (self.shape[0], 1):
            raise ValueError(
                f"Incorrect dimensions for vector-matrix multiplication. Inputs had size {other.shape} and {self.shape}."
            )
        v = np.zeros(self.A.shape[0], dtype=np.float64)
        v[self.sample_indices] = other.ravel().astype(np.float64)
        spsolve_backward_triangular(self.A, v)
        v = v[self.variant_indices]
        if np.any(self.flip):
            v[self.flip] = np.sum(other) - v[self.flip]
        return v if other.ndim == 1 else v.reshape(-1, 1)

    def __getitem__(self, key: tuple[slice, slice]) -> "LinearARG":
        # TODO make this work with syntax like linarg[:100,] (works with linarg[:100,:])
        # rows, cols = key
        # return LinearARG(self.A, self.sample_indices[rows], self.variants[cols])
        pass

    def copy(self) -> "LinearARG":
        # return LinearARG(self.A.copy(), self.sample_indices.copy(), self.variants.copy())
        pass

    def write(self, prefix: Union[str, PathLike], compression_option: str = "gzip"):
        """Writes LinearARG to disk.
        :param prefix: The base path and prefix used for output files.
        :return: None
        """

        # write out DAG info
        with h5py.File(prefix + ".h5", "w") as f:
            f.attrs['n'] = self.A.shape[0]
            f.attrs['n_samples'] = self.n_samples
            f.create_dataset('indptr', data=self.A.indptr, compression=compression_option, shuffle=True)
            f.create_dataset('indices', data=self.A.indices, compression=compression_option, shuffle=True)
            f.create_dataset('data', data=self.A.data, compression=compression_option, shuffle=True)
            f.create_dataset('variant_indices', data=self.variant_indices, compression=compression_option, shuffle=True)
            f.create_dataset('flip', data=self.flip, compression=compression_option, shuffle=True)
            if self.nonunique_indices is not None:
                f.create_dataset('nonunique_indices', data=self.nonunique_indices, compression=compression_option, shuffle=True)
            if self.variants is not None:
                for field in self.variants.req_fields:
                    if field == 'POS':
                        f.create_dataset(field, data=np.array(self.variants.table[field]).astype(int), compression=compression_option, shuffle=True)
                    else:
                        f.create_dataset(field, data=np.array(self.variants.table[field]).astype('S'), compression=compression_option, shuffle=True)
        return

    @staticmethod
    def read(
        h5_fname: Union[str, PathLike],
        load_metadata = False,
    ) -> "LinearARG":
        """Reads LinearARG data from provided PLINK2 formatted files.

        :param matrix_fname: Filename for the .h5 file.
        :return: A LinearARG object.
        """
        if load_metadata:
            v_info = VariantInfo.read(h5_fname)
        else:
            v_info = None
            
        with h5py.File(h5_fname, 'r') as f:
            A = csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=(f.attrs['n'], f.attrs['n']))
            variant_indices = f['variant_indices'][:]
            flip = f['flip'][:]
            n_samples = f.attrs['n_samples']
            if 'nonunique_indices' in f:
                nonunique_indices = f['nonunique_indices'][:]
            else:
                nonunique_indices = None
                
        return LinearARG(A, variant_indices, flip, n_samples, v_info, nonunique_indices)

    def calculate_nonunique_indices(self) -> None:
        """Calculates and stores non-unique indices to facilitate memory-efficient matmat and rmatmat operations."""
        if self.nonunique_indices is None:
            self.nonunique_indices = get_nonunique_indices_csc(
                self.A.indices,
                self.A.indptr,
                self.sample_indices,
                self.variant_indices,
            )
            self.nonunique_indices = np.asarray(self.nonunique_indices)
            print(f"Non-unique indices: {self.num_nonunique_indices} vs. {self.A.shape[0]}")
            
    @cached_property
    def num_nonunique_indices(self) -> Optional[int]:
        if self.nonunique_indices is None:
            return None
        return np.max(self.nonunique_indices) + 1


def remove_degree_zero_nodes(A: csc_matrix, variant_indices: npt.NDArray[np.uint], sample_indices: npt.NDArray[np.uint]) -> tuple:
    """
    Removes degree-zero recombination nodes from the graph, while ensuring all nodes
    in variant_indices and sample_indices are retained. Returns the
    filtered graph and reindexed variant/sample indices.
    """
    node_degree = (A.getnnz(axis=0) + A.getnnz(axis=1))
    nonzero_indices = set(np.where(node_degree > 0)[0])
    required_indices = set(variant_indices).union(sample_indices)
    indices_to_keep = np.array(sorted(nonzero_indices.union(required_indices)), dtype=int)
    
    index_map = -np.ones(A.shape[0], dtype=int)
    index_map[indices_to_keep] = np.arange(len(indices_to_keep))
    
    A_filt = A[indices_to_keep, :][:, indices_to_keep]
    variant_indices_reindexed = index_map[variant_indices]
    sample_indices_reindexed = index_map[sample_indices]
    
    return A_filt, variant_indices_reindexed, sample_indices_reindexed
            

def make_triangular(A: csc_matrix, variant_indices: npt.NDArray[np.uint], sample_indices: npt.NDArray[np.uint]) -> tuple:
    """
    Triangularizes A by putting nodes in topological order (parents before children) such that sample/leaf nodes are in reverse order
    starting from the final row/column of the returned csc_matrix. Additionally, variant_indices are reindexed with respect to this
    new node ordering.
    """
    A_csr = csr_matrix(A)
    order = np.asarray(topological_sort(A_csr, nodes_to_ignore=sample_indices))[:-len(sample_indices)]
    order = np.append(order, sample_indices[::-1])
    inv_order = np.argsort(order).astype(np.int32)

    A_triangular = A[order, :][:, order]
    variant_indices_reordered = inv_order[variant_indices]
    
    return A_triangular, variant_indices_reordered