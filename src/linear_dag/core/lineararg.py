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
    req_cols: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT"]

    def __post_init__(self):
        for req_col in self.req_cols:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in variant table")

    @cached_property
    def is_flipped(self):
        return self.table[self.flip_field].to_numpy().astype(bool)

    @cached_property
    def indices(self):
        return self.table[self.idx_field].to_numpy().astype(np.int32)

    def copy(self):
        return VariantInfo(self.table.clone())

    def __getitem__(self, key):
        return VariantInfo(self.table[key])

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "VariantInfo":
        if path is None:
            raise ValueError("path argument cannot be None")

        open_f = gzip.open if str(path).endswith(".gz") else open
        header_map = None
        var_table = defaultdict(list)
        with open_f(path, "rt") as var_file:
            for line in var_file:
                if line.startswith("##"):
                    continue
                elif line[0] == "#":
                    names = line[1:].strip().split()
                    header_map = {key: idx for idx, key in enumerate(names)}
                    for req_name in cls.req_fields:
                        if req_name not in header_map:
                            # we check again later based on dataframe, but better to error out early when parsing
                            raise ValueError(f"Required column {req_name} not found in header table")
                    continue

                # parse row; this can easily break...
                row = line.strip().split()
                for field in header_map:
                    value = row[header_map[field]]
                    if field == 'POS': # cast POS as int to save memory
                        value = int(value)
                    var_table[field].append(value)

        var_table = pl.DataFrame(var_table)

        # return class instance
        return cls(var_table)

    def write(self, path: Union[str, PathLike]):
        open_f = gzip.open if str(path).endswith(".gz") else open
        with open_f(path, "wt") as pvar_file:
            pvar_file.write(f"##fileformat=PVARv1.0{linesep}")
            pvar_file.write(f'##INFO=<ID=IDX,Number=1,Type=Integer,Description="Variant Index">{linesep}')
            pvar_file.write(f'##INFO=<ID=FLIP,Number=0,Type=Flag,Description="Flip Information">{linesep}')
            pvar_file.write("\t".join([f"#{self.req_fields[0]}"] + self.req_fields[1:]) + linesep)
            pvar_file.flush()
            pvar_file.write(self.table.write_csv(include_header=False, separator="\t"))
        return

    # @classmethod
    # def from_open_bed(
    #     cls, bed: br.open_bed, indices: Optional[npt.ArrayLike] = None, is_flipped: Optional[npt.ArrayLike] = None
    # ):
    #     # doesn't really follow conventions for a class name...
    #     # req_cols: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", idx_field, flip_field]
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
    A: csc_matrix
    sample_indices: npt.NDArray[np.int32]
    variants: VariantInfo
    nonunique_indices: Optional[npt.NDArray[np.int32]] = None

    @property
    def variant_indices(self) -> npt.NDArray[np.int32]:
        return self.variants.indices

    @property
    def flip(self):
        return self.variants.is_flipped
    A: csr_matrix
    flip: npt.NDArray[np.bool_]
    variant_indices: npt.NDArray[np.uint]
    sample_indices: npt.NDArray[np.uint]
    variants: VariantInfo = None

    @staticmethod
    def from_genotypes(
        genotypes: csc_matrix,
        flip: npt.NDArray[np.bool_],
        variant_info: pl.DataFrame = None,
        find_recombinations: bool = True,
        make_triangular: bool = True,
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
        linear_arg_adjacency_matrix, flip, variants_idx, samples_idx, variant_info = linear_arg_from_genotypes(
            genotypes, variant_info, find_recombinations, verbosity
        )
        result = LinearARG(linear_arg_adjacency_matrix, flip, variants_idx, samples_idx, VariantInfo(variant_info))
        if make_triangular:
            result = result.make_triangular()

        return result

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
            temp = other.T * (-1) ** self.flip.reshape(1,-1)
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
            v[:, self.flip] = np.sum(other, axis=0) - v[:, self.flip]
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
        rows, cols = key
        return LinearARG(self.A, self.sample_indices[rows], self.variants[cols])

    def copy(self) -> "LinearARG":
        return LinearARG(self.A.copy(), self.sample_indices.copy(), self.variants.copy())

    def write(self, prefix: Union[str, PathLike], compression_option: str = "gzip"):
        """Writes LinearARG triplet to disk.
        :param prefix: The base path and prefix used for output files.
        :return: None
        """
        # write out sample info
        # temporary fix
        iids = None
        with gzip.open(f"{prefix}.psam.gz", "wt") as f_samples:
            f_samples.write("#IID IDX\n")
            if iids is None:
                iids = [f"sample_{idx}" for idx in range(self.shape[0])]
            for i, iid in enumerate(iids):
                f_samples.write(f"{iid} {self.sample_indices[i]}\n")
        # self.samples.write(prefix + ".psam")

        # write out variant info
        self.variants.write(prefix + ".pvar.gz")

        # write out DAG info
        with h5py.File(prefix + ".h5", "w") as f:
            f.attrs['n'] = self.A.shape[0]
            f.create_dataset('indptr', data=self.A.indptr, compression=compression_option, shuffle=True)
            f.create_dataset('indices', data=self.A.indices, compression=compression_option, shuffle=True)
            f.create_dataset('data', data=self.A.data, compression=compression_option, shuffle=True)
            f.create_dataset('variant_indices', data=self.variant_indices, compression=compression_option, shuffle=True)
            f.create_dataset('flip', data=self.flip, compression=compression_option, shuffle=True)
        return

    @staticmethod
    def read(
        matrix_fname: Union[str, PathLike],
        variant_fname: Union[str, PathLike] = None,
        samples_fname: Union[str, PathLike] = None,
        load_metadata = False,
    ) -> "LinearARG":
        """Reads LinearARG data from provided PLINK2 formatted files.

        :param matrix_fname: Filename for the .h5 file containing the adjacency matrix, variant_indices, and flip.
        :param variant_fname: Filename for the .pvar file containing variant data.
        :param samples_fname: Filename for the .psam file containing sample IDs.

        :return: A tuple containing the LinearARG object, list of variant IDs, and list of IIDs.
        """
        if not variant_fname:
            variant_fname = matrix_fname[:-3] + ".pvar.gz"
        if not samples_fname:
            samples_fname = matrix_fname[:-3] + ".psam.gz"

        # Load sample info
        # temporary fix
        sample_info = pl.read_csv(samples_fname, separator=" ")
        sample_indices = np.array(sample_info["IDX"], dtype=np.int32)
        # s_info = SampleInfo.read(samples_fname)

        # Load variant info
        if load_metadata:
            v_info = VariantInfo.read(variant_fname)
        else:
            v_info = None

        # Load the adjacency matrix
        with h5py.File(matrix_fname, 'r') as f:
            A = csr_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=(f.attrs['n'], f.attrs['n']))
            variant_indices = f['variant_indices'][:]
            flip = f['flip'][:]

        A = csc_matrix(A)
        
        return LinearARG(A, sample_indices, v_info)
        # return LinearARG(A, s_info.indices, v_info)
        # Construct the final object and return!
        return LinearARG(A, flip, variant_indices, sample_indices, v_info)

    def make_triangular(self) -> "LinearARG":
        order = np.asarray(topological_sort(self.A))
        inv_order = np.argsort(order).astype(np.int32)

        A = self.A[order, :][:, order]
        s_idx = inv_order[self.sample_indices]
        v_idx = inv_order[self.variant_indices]
        

        # this results in an out of bounds error since the variant indices are greater than the number of rows in v_info
        # v_info = self.variants[v_idx]
        # return LinearARG(A, s_idx, v_info)

        return LinearARG(A, s_idx, VariantInfo(v_info))

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
            
        return LinearARG(A, self.flip, v_idx, s_idx, self.variants)
