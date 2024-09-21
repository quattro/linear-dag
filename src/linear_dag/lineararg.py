# lineararg.py
import gzip

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from os import linesep, PathLike
from typing import ClassVar, DefaultDict, Optional, Union

import bed_reader as br
import cyvcf2 as cv
import numpy as np
import numpy.typing as npt
import polars as pl

from scipy.sparse import csc_matrix, csr_matrix, eye, load_npz, save_npz
from scipy.sparse.linalg import spsolve_triangular

from .brick_graph import BrickGraph
from .linear_arg_inference import linear_arg_from_genotypes
from .one_summed_cy import linearize_brick_graph
from .recombination import Recombination
from .solve import topological_sort



@dataclass
class VariantInfo:
    """Metadata about variants represented in the linear dag.

    **Attributes**
    """

    table: pl.DataFrame

    flip_field: ClassVar[str] = "FLIP"
    idx_field: ClassVar[str] = "IDX"
    req_fields: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", "INFO"]
    req_cols: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", idx_field, flip_field]

    def __post_init__(self):
        for req_col in self.req_cols:
            if req_col not in self.table.columns:
                raise ValueError(f"Required column {req_col} not found in variant table")

    @cached_property
    def is_flipped(self):
        return self.table[self.flip_field].to_numpy().astype(bool)

    @cached_property
    def indices(self):
        return self.table[self.idx_field].to_numpy().astype(int)

    def copy(self):
        return VariantInfo(self.table.clone())

    def __getitem__(self, key):
        return VariantInfo(self.table[key])

    @classmethod
    def read(cls, path: Union[str, PathLike]) -> "VariantInfo":
        if path is None:
            raise ValueError("path argument cannot be None")

        def _parse_info(info_str):
            idx = -1
            flip = False
            for info in info_str.split(";"):
                s_info = info.split("=")
                if len(s_info) == 2 and s_info[0] == "IDX":
                    idx = int(s_info[1])
                elif len(s_info) == 1 and s_info[0] == "FLIP":
                    flip = True

            return idx, flip

        open_f = gzip.open if str(path).endswith(".gz") else open
        header_map = None
        var_table = defaultdict(list)
        with open_f(path, "r") as var_file:
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
                for field in cls.req_fields:
                    # skip INFO for now...
                    if field == "INFO":
                        continue
                    value = row[header_map[field]]
                    var_table[field].append(value)

                # parse info to pull index and flip info if they exist
                idx, flip = _parse_info(row[header_map["INFO"]])
                var_table[cls.idx_field].append(idx)
                var_table[cls.flip_field].append(flip)

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

            # flush to make sure this exists before writing the table out
            pvar_file.flush()

            # we need to map IDX and FLIP columns back to INFO
            # this was giving me "AttributeError: 'Expr' object has no attribute 'apply'"
            # sub_table = self.table.with_columns(
            #     (
            #         pl.col(self.idx_field).apply(lambda idx: f"IDX={idx}")
            #         + pl.col(self.flip_field).apply(lambda flip: f";{self.flip_field}" if flip else "")
            #     ).alias("INFO")
            # ).drop([self.idx_field, self.flip_field])
            # sub_table.write_csv(pvar_file, has_header=False, separator="\t")
            
            info_col = pl.Series([f'IDX={idx};FLIP={flip}' for idx, flip in zip(self.table['IDX'], self.table['FLIP'])])
            sub_table = self.table.with_columns(info_col.alias('INFO')).drop([self.idx_field, self.flip_field]) 
            sub_table.write_csv(pvar_file, include_header=False, separator="\t")

        return

    @classmethod
    def from_open_bed(
        cls, bed: br.open_bed, indices: Optional[npt.ArrayLike] = None, is_flipped: Optional[npt.ArrayLike] = None
    ):
        # doesn't really follow conventions for a class name...
        # req_cols: ClassVar[list[str]] = ["CHROM", "POS", "ID", "REF", "ALT", idx_field, flip_field]
        df = dict()
        df["CHROM"] = bed.chromosome
        df["POS"] = bed.bp_position
        df["ID"] = bed.sid
        df["REF"] = bed.allele_1
        df["ALT"] = bed.allele_2
        if indices is None:
            df[cls.idx_col] = -1 * np.ones(len(bed.bp_position), dtype=int)
        else:
            if len(indices) != len(bed.bp_position):
                raise ValueError("Length of indices does not match number of variants")
            df[cls.idx_col] = indices
        if is_flipped is None:
            df[cls.flip_field] = np.zeros(len(bed.iid)).astype(bool)
        else:
            if len(is_flipped) != len(bed.bp_position):
                raise ValueError("Length of is_flipped flags does not match number of variants")
            df[cls.flip_field] = is_flipped

        return cls(pl.DataFrame(df))

    @staticmethod
    def _update_dict_from_vcf(
        var: cv.Variant, is_flipped: bool, data: DefaultDict[str, list]
    ) -> DefaultDict[str, list]:
        data["CHROM"].append(var.CHROM)
        data["ID"].append(var.ID)
        data["POS"].append(var.POS)
        data["REF"].append(var.REF)
        data["ALT"].append(",".join(var.ALT))
        data["FLIP"].append(is_flipped)

        return data


@dataclass
class LinearARG:
    A: csr_matrix
    sample_indices: npt.NDArray[np.uint]
    variants: VariantInfo

    @property
    def variant_indices(self):
        return self.variants.indices

    @property
    def flip(self):
        return self.variants.is_flipped

    @staticmethod
    def from_genotypes(
        genotypes: csc_matrix,
        variant_info: pl.DataFrame = None,
        find_recombinations: bool = True,
        make_triangular: bool = True,
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
        linear_arg_adjacency_matrix, samples_idx, variant_info = linear_arg_from_genotypes(genotypes, variant_info, find_recombinations)
        result = LinearARG(linear_arg_adjacency_matrix, samples_idx, VariantInfo(variant_info))
        if make_triangular:
            result = result.make_triangular()

        return result

    @staticmethod
    def from_plink(prefix: str) -> "LinearARG":
        import bed_reader as br

        with br.open_bed(f"{prefix}.bed") as bed:
            genotypes = bed.read_sparse(dtype="int8")

        larg = LinearARG.from_genotypes(genotypes)
        v_info = VariantInfo.from_open_bed(bed, larg.variant_indices)
        larg.variants = v_info
        return larg

    #  TODO move file IO out of LinearARG
    @staticmethod
    def from_vcf(path: Union[str, PathLike],
                 phased: bool = True,
                 region: Optional[str] = None,
                 flip_minor_alleles: bool = False,
                 return_genotypes: bool = False) -> "LinearARG":
        vcf = cv.VCF(path, gts012=True, strict_gt=True)
        data = []
        idxs = []
        ptrs = [0]

        ploidy = 1 if phased else 2

        # push most of the branching up here to define functions for fewer branch conditions during loop
        if phased:
            read_gt = lambda var: np.ravel(np.asarray(var.genotype.array())[:, :2])  # noqa: E731
        else:
            read_gt = lambda var: var.gt_types  # noqa: E731

        def final_read(var, flip_minor_alleles):
            gts = read_gt(var)
            if not flip_minor_alleles:
                return gts, False
            af = np.mean(gts) / ploidy
            if af > 0.5:
                return ploidy - gts, True
            else:
                return gts, False

        var_table = defaultdict(list)
        # TODO: handle missing data
        for var in vcf(region):
            gts, is_flipped = final_read(var, flip_minor_alleles)

            (idx,) = np.where(gts != 0)
            data.append(gts[idx])
            idxs.append(idx)
            ptrs.append(ptrs[-1] + len(idx))
            var_table = VariantInfo._update_dict_from_vcf(var, is_flipped, var_table)

        v_info = pl.DataFrame(var_table)

        data = np.concatenate(data)
        idxs = np.concatenate(idxs)
        ptrs = np.array(ptrs)
        genotypes = csc_matrix((data, idxs, ptrs))
        result = LinearARG.from_genotypes(genotypes, v_info)
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

    def __str__(self):
        return f"A: shape {self.A.shape}, nonzeros {self.A.nnz}"

    def __matmul__(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
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

    def __rmatmul__(self, other: npt.ArrayLike) -> npt.NDArray[np.number]:
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
        return LinearARG(self.A, self.samples[rows], self.variants[cols])

    def copy(self) -> "LinearARG":
        return LinearARG(self.A.copy(), self.samples.copy(), self.variants.copy())

    def write(self, prefix: Union[str, PathLike]):
        """Writes LinearARG triplet to disk.

        :param prefix: The base path and prefix used for output files.

        :return: None
        """
        # write out sample info
        # temporary fix
        iids = None
        with open(f"{prefix}.psam", "w") as f_samples:
                f_samples.write("#IID IDX\n")
                if iids is None:
                    iids = [f"sample_{idx}" for idx in range(self.shape[0])]
                for i, iid in enumerate(iids):
                    f_samples.write(f"{iid} {self.sample_indices[i]}\n")
        # self.samples.write(prefix + ".psam")

        # write out variant info
        self.variants.write(prefix + ".pvar")

        # write out DAG info
        save_npz(f"{prefix}.npz", self.A)

        return

    @staticmethod
    def read(
        matrix_fname: Union[str, PathLike],
        variant_fname: Union[str, PathLike],
        samples_fname: Union[str, PathLike],
    ) -> "LinearARG":
        """Reads LinearARG data from provided PLINK2 formatted files.

        :param matrix_fname: Filename for the .npz file containing the adjacency matrix.
        :param variant_fname: Filename for the .pvar file containing variant data.
        :param samples_fname: Filename for the .psam file containing sample IDs.

        :return: A tuple containing the LinearARG object, list of variant IDs, and list of IIDs.
        """

        # Load sample info
        # temporary fix
        sample_info = pl.read_csv(samples_fname, separator=' ')
        sample_indices = np.array(sample_info['IDX'])
        # s_info = SampleInfo.read(samples_fname)

        # Load variant info
        v_info = VariantInfo.read(variant_fname)

        # Load the adjacency matrix
        A = load_npz(matrix_fname)

        # Construct the final object and return!
        return LinearARG(A, sample_indices, v_info)
        # return LinearARG(A, s_info.indices, v_info)


    def make_triangular(self) -> "LinearARG":
        order = np.asarray(topological_sort(self.A))
        inv_order = np.argsort(order)

        A = self.A[order, :][:, order]
        s_idx = inv_order[self.sample_indices]

        v_idx = inv_order[self.variant_indices]
        v_info = self.variants.table.clone()
        v_idx = pl.Series(v_idx)
        v_info = v_info.with_columns(v_idx.alias('IDX'))
        
        # this results in an out of bounds error since the variant indices are greater than the number of rows in v_info
        # v_info = self.variants[v_idx]
        # return LinearARG(A, s_idx, v_info)

        return LinearARG(A, s_idx, VariantInfo(v_info))
