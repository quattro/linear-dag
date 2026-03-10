# lineararg.py
import logging
import os

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

from linear_dag.core.operators import get_pairing_matrix
from linear_dag.core.solve import get_carriers, get_nonunique_indices_csc
from linear_dag.genotype import read_vcf

from .digraph import DiGraph
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
    """Sparse linear-operator representation of a linear ARG.

    This [`linear_dag.core.lineararg.LinearARG`][] class wraps the graph
    adjacency matrix and variant/sample indexing needed to expose genotype
    operations through the
    `scipy.sparse.linalg.LinearOperator` interface.

    !!! Example
        ```python
        linarg = LinearARG.read("example.h5", block="1:1000-2000")
        x = np.ones((linarg.shape[1], 1), dtype=np.float32)
        y = linarg @ x
        ```

    """

    A: csc_matrix  # samples must be in descending order starting from the final row/col
    variant_indices: npt.NDArray[np.int32]
    flip: npt.NDArray[np.bool_]
    n_samples: np.int32
    n_individuals: Optional[np.int32] = None
    variants: Optional[pl.LazyFrame] = None
    iids: Optional[pl.Series] = None
    nonunique_indices: Optional[npt.NDArray[np.int32]] = None
    sex: Optional[npt.NDArray[np.int32]] = None  # determines how individual_indices are handled
    # allele_counts: Optional[npt.NDArray[np.int32]] = None

    @cached_property
    def allele_counts(self) -> npt.NDArray[np.int32]:
        """Compute and cache allele counts (sum of allele dosages across all samples)."""
        return np.ones(self.shape[0], dtype=np.int32) @ self

    def set_allele_counts(self, counts: npt.NDArray[np.int32]) -> None:
        """Pre-set allele counts (e.g., when loading from disk)."""
        object.__setattr__(self, "allele_counts", counts)

    @cached_property
    def allele_frequencies(self):
        """Return per-variant allele frequencies across all samples.

        **Returns:**

        - NumPy array of allele frequencies with length `self.shape[1]`.
        """
        if self.allele_counts is None:  # if not precomputed
            return (np.ones(self.shape[0], dtype=np.int32) @ self) / self.shape[0]
        else:
            return self.allele_counts / self.shape[0]

    @property
    def individual_indices(self):
        """Return graph-node indices corresponding to individual nodes.

        **Returns:**

        - NumPy array of node indices for individual nodes.

        **Raises:**

        - `ValueError`: if this [`linear_dag.core.lineararg.LinearARG`][] has no
          individual nodes.
        """
        if self.n_individuals is None:
            raise ValueError("The linear ARG does not have individual nodes. Try running add_individual_nodes first.")
        return np.arange(self.A.shape[0] - self.n_individuals, self.A.shape[0], dtype=np.int32)

    @property
    def sample_indices(self):
        """Return graph-node indices corresponding to sample haplotype nodes.

        **Returns:**

        - NumPy array of sample-node indices ordered from first to last sample row.
        """
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
        logger: Optional[logging.Logger] = None,
    ):
        """Infer a [`linear_dag.core.lineararg.LinearARG`][] from phased genotypes.

        **Arguments:**

        - `genotypes`: CSC matrix of phased haploid genotypes with shape `(n_samples, n_variants)`.
        - `flip`: Boolean per-variant mask indicating allele flips.
        - `variant_info`: Optional Polars dataframe with variant metadata.
        - `iids`: Optional identifiers corresponding to genotype rows.
        - `find_recombinations`: Whether to infer recombination nodes before linearization.
        - `sex`: Optional per-individual sex array for later individual-node expansion.
        - `logger`: Optional logger for timing/progress output.

        **Returns:**

        - Inferred [`linear_dag.core.lineararg.LinearARG`][] instance.

        **Raises:**

        - `TypeError`: Propagated from inference if `genotypes` is not CSC.
        """
        import time

        A, flip, variants_idx, samples_idx, variant_info = linear_arg_from_genotypes(
            genotypes, flip, variant_info, find_recombinations, logger=logger
        )

        if logger is not None:
            logger.info("Removing degree-zero nodes")
        t0 = time.time()
        A_filt, variants_idx_reindexed, samples_idx_reindexed = remove_degree_zero_nodes(A, variants_idx, samples_idx)
        t1 = time.time()
        if logger is not None:
            logger.info(f"  Time: {t1 - t0:.3f}s")

        if logger is not None:
            logger.info("Making triangular")
        t0 = time.time()
        A_tri, variants_idx_tri = make_triangular(A_filt, variants_idx_reindexed, samples_idx_reindexed)
        t1 = time.time()
        if logger is not None:
            logger.info(f"  Time: {t1 - t0:.3f}s")

        if logger is not None:
            logger.info("Creating LinearARG object")
        t0 = time.time()
        linarg = LinearARG(
            A_tri,
            variants_idx_tri,
            flip,
            len(samples_idx),
            None,
            variant_info.lazy(),
            iids=pl.Series(iids).cast(pl.String),
        )
        linarg.calculate_nonunique_indices()
        t1 = time.time()
        if logger is not None:
            logger.info(f"  Time: {t1 - t0:.3f}s")
        return linarg

    @staticmethod
    def from_vcf(
        path: Union[str, PathLike],
        phased: bool = True,
        region: Optional[str] = None,
        include_samples: Optional[list] = None,
        flip_minor_alleles: bool = False,
        return_genotypes: bool = False,
        logger: Optional[logging.Logger] = None,
        maf_filter: float = None,
        snps_only: bool = False,
        remove_multiallelics: bool = False,
    ) -> Union[tuple, "LinearARG"]:
        """Read a VCF and infer a [`linear_dag.core.lineararg.LinearARG`][].

        **Arguments:**

        - `path`: Path to VCF/BCF input.
        - `phased`: Whether the input genotypes are phased.
        - `region`: Optional region string passed to VCF reader.
        - `include_samples`: Optional row indices to keep after loading.
        - `flip_minor_alleles`: Whether to flip minor alleles during VCF parsing.
        - `return_genotypes`: Whether to return `(linarg, genotypes)` instead of only `linarg`.
        - `logger`: Optional logger for timing/progress output.
        - `maf_filter`: Optional MAF filter applied during VCF parsing.
        - `snps_only`: Whether to remove indels.
        - `remove_multiallelics`: Whether to remove multi-allelic sites.

        **Returns:**

        - [`linear_dag.core.lineararg.LinearARG`][] or a tuple of
          ([`linear_dag.core.lineararg.LinearARG`][], `csc_matrix`) when
          `return_genotypes` is `True`.

        **Raises:**

        - `ValueError`: If no valid variants are found in the VCF after filtering.
        """
        import time

        if logger is not None:
            logger.info("Reading VCF")
        t0 = time.time()
        genotypes, flip, v_info, iids = read_vcf(
            path,
            phased,
            region,
            flip_minor_alleles,
            maf_filter=maf_filter,
            remove_indels=snps_only,
            remove_multiallelics=remove_multiallelics,
        )
        t1 = time.time()
        if logger is not None:
            logger.info(f"  Time: {t1 - t0:.3f}s")

        if genotypes is None:
            raise ValueError("No valid variants found in VCF")
        if logger is not None:
            logger.info(f"Number of variants: {genotypes.shape[1]}")

        if phased:
            iids = [id_ for id_ in iids for _ in range(2)]

        if include_samples:
            genotypes = genotypes[include_samples, :]
            iids = [iids[i] for i in include_samples]

        result = LinearARG.from_genotypes(genotypes, flip, v_info, iids=iids, logger=logger)

        return (result, genotypes) if return_genotypes else result

    @property
    def shape(self):
        """Return linear-operator shape `(n_samples, n_variants)`.

        **Returns:**

        - Two-tuple `(n_samples, n_variants)`.
        """
        n = len(self.sample_indices)
        m = len(self.variant_indices)
        return n, m

    @property
    def nnz(self):
        """Return number of non-zero entries in the adjacency matrix.

        **Returns:**

        - Integer number of non-zero entries.
        """
        return self.A.nnz

    @property
    def ndim(self):
        """Return dimensionality of the linear operator.

        **Returns:**

        - Integer `2`.
        """
        return 2

    @property
    def dtype(self):
        """Return element dtype of the underlying sparse adjacency matrix.

        **Returns:**

        - NumPy dtype object.
        """
        return self.A.dtype

    @property
    def mean_centered(self):
        """Return a mean-centered genotype linear operator.

        **Returns:**

        - `LinearOperator` with per-variant mean removed.
        """
        mean = aslinearoperator(np.ones((self.shape[0], 1), dtype=np.float32)) @ aslinearoperator(
            self.allele_frequencies
        )
        return self - mean

    @property
    def normalized(self):
        """Return a normalized genotype operator with variance-stabilized columns.

        **Returns:**

        - `LinearOperator` with mean-zero, variance-scaled columns.
        """
        pq = self.allele_frequencies * (1 - self.allele_frequencies)
        pq[pq == 0] = 1
        return self.mean_centered @ aslinearoperator(diags(pq**-0.5))

    def number_of_heterozygotes(self, indiv_to_include: np.ndarray | None = None):
        """Count heterozygous carriers for each variant.

        !!! info
            This method requires individual nodes. Call `add_individual_nodes()`
            first when working from haplotype-only graphs.

        **Arguments:**

        - `indiv_to_include`: optional boolean mask over individuals.

        **Returns:**

        - Integer array of heterozygote counts per variant.

        **Raises:**

        - `ValueError`: if individual nodes are absent or mask dtype is not boolean.
        """
        if self.n_individuals is None:
            raise ValueError("The linear ARG does not have individual nodes. Try running add_individual_nodes first.")
        if indiv_to_include is None:
            indiv_to_include = np.ones(self.n_individuals, dtype=np.bool_)
        if indiv_to_include.dtype != np.bool_:
            raise ValueError("indiv_to_include must be a boolean array")

        # num_het = 2 * num_carriers - num_alleles
        vi = np.zeros(self.A.shape[0], dtype=np.float64)
        vi[self.individual_indices[indiv_to_include]] = 2
        vi[self.sample_indices[np.repeat(indiv_to_include, 2)]] = -1
        spsolve_backward_triangular(self.A, vi)
        return vi[self.variant_indices].astype(np.int32)

    def number_of_carriers(self, indiv_to_include: np.ndarray | None = None):
        """Count carriers for each variant in selected individuals.

        !!! info

            For flipped variants, this method reports carrier counts after applying
            the flip convention used by this
            [`linear_dag.core.lineararg.LinearARG`][].

        **Arguments:**

        - `indiv_to_include`: optional boolean mask over individuals.

        **Returns:**

        - Integer array of carrier counts per variant.

        **Raises:**

        - `ValueError`: if individual nodes are absent or mask dtype is not boolean.
        """
        if self.n_individuals is None:
            raise ValueError("The linear ARG does not have individual nodes. Try running add_individual_nodes first.")
        if indiv_to_include is None:
            indiv_to_include = np.ones(self.n_individuals, dtype=np.bool_)
        if indiv_to_include.dtype != np.bool_:
            raise ValueError("indiv_to_include must be a boolean array")
        # n1 + n2
        vi = np.zeros(self.A.shape[0], dtype=np.float64)
        vi[self.individual_indices[indiv_to_include]] = 1
        spsolve_backward_triangular(self.A, vi)
        alt_carriers = vi[self.variant_indices]

        if not np.any(self.flip):
            return alt_carriers.astype(np.int32)

        # n1 + 2*n2
        vh = np.zeros(self.A.shape[0], dtype=np.float64)
        vh[self.sample_indices[np.repeat(indiv_to_include, 2)]] = 1
        spsolve_backward_triangular(self.A, vh)
        hap_counts = vh[self.variant_indices]

        # handle flipped variants
        hom_alt = hap_counts - alt_carriers
        ref_carriers = np.sum(indiv_to_include) - hom_alt
        num_carriers = alt_carriers.copy()
        num_carriers[self.flip] = ref_carriers[self.flip]

        return num_carriers.astype(np.int32)

    def get_carriers_subset(self, variant_indices: npt.NDArray[np.int_], unphased: bool = False) -> csc_matrix:
        """
        Get carriers for a subset of variants specified by variant_indices.

        **Arguments:**

        - `variant_indices`: Indices into the variant dimension (`0` to `shape[1]-1`).
        - `unphased`: Whether to collapse paired haplotypes into diploid carriers.

        **Returns:**

        - CSC matrix of shape `(n_samples, len(variant_indices))` indicating carriers.
        """
        variant_node_indices = self.variant_indices[variant_indices]
        if self.n_individuals is None:
            A = self.A
        else:
            n = self.A.shape[0] - self.n_individuals
            A = self.A[:n, :][:, :n]
        carriers: csc_matrix = get_carriers(A, variant_node_indices, self.n_samples)
        if unphased:
            carriers = get_pairing_matrix(self.shape[0]) @ carriers
        return carriers

    def remove_samples(self, iids_to_remove: npt.NDArray[np.int_]):
        """Create a new [`linear_dag.LinearARG`][] with selected sample IDs removed.

        **Arguments:**

        - `iids_to_remove`: iterable of sample IDs to remove.

        **Returns:**

        - New [`linear_dag.LinearARG`][] with updated adjacency and metadata.
        """
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
        """Create a deep copy of this [`linear_dag.core.lineararg.LinearARG`][].

        **Returns:**

        - New [`linear_dag.core.lineararg.LinearARG`][] with copied sparse data and metadata.
        """
        copied = LinearARG(
            A=self.A.copy(),
            variant_indices=self.variant_indices.copy(),
            flip=self.flip.copy(),
            n_samples=np.int32(self.n_samples),
            n_individuals=None if self.n_individuals is None else np.int32(self.n_individuals),
            variants=self.variants.clone() if self.variants is not None else None,
            iids=self.iids.clone() if self.iids is not None else None,
            nonunique_indices=None if self.nonunique_indices is None else self.nonunique_indices.copy(),
            sex=None if self.sex is None else self.sex.copy(),
        )
        # Preserve cached allele counts when already materialized.
        if "allele_counts" in self.__dict__:
            copied.set_allele_counts(self.allele_counts.copy())
        return copied

    def write(
        self,
        h5_fname: Union[str, PathLike],
        block_info: Optional[dict] = None,
        compression_option: str = "gzip",
        save_allele_counts: bool = True,
    ):
        """Write [`linear_dag.core.lineararg.LinearARG`][] data to disk.

        **Arguments:**

        - `h5_fname`: Base path/prefix used for output files.
        - `block_info`: Optional dictionary with keys `chrom`, `start`, and `end`.
        - `compression_option`: HDF5 compression option.
        - `save_allele_counts`: Whether to persist precomputed allele counts.

        **Returns:**

        - `None`.

        **Raises:**

        - `FileExistsError`: If writing a non-block file and the output already exists.
        """

        fname = h5_fname if str(h5_fname).endswith(".h5") else str(h5_fname) + ".h5"
        mode = "a" if block_info else "w"
        if (not block_info) and os.path.exists(fname):
            raise FileExistsError(
                f"The file '{fname}' already exists."
                "To append a new linear ARG to an existing file, specify `block_info`."
            )
        with h5py.File(fname, mode) as f:
            if block_info:
                block_name = f"{block_info['chrom']}:{block_info['start']}-{block_info['end']}"
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
                str_iids = np.array(self.iids, dtype=object)
                f.create_dataset(
                    "iids",
                    data=str_iids,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression=compression_option,
                    shuffle=True,
                )
            if self.n_individuals is not None:
                destination.attrs["n_individuals"] = self.n_individuals
            if self.variants is not None:
                variant_info = self.variants.collect()
                for field in ["CHROM", "POS", "ID", "REF", "ALT"]:
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
                            data=np.array(variant_info[field], dtype=object),
                            dtype=h5py.string_dtype(encoding="utf-8"),
                            compression=compression_option,
                            shuffle=True,
                        )

            if save_allele_counts:
                allele_counts = np.ones(self.shape[0], dtype=np.int32) @ self
                destination.create_dataset(
                    "allele_counts",
                    data=allele_counts.astype(int),
                    compression=compression_option,
                    shuffle=True,
                )

        return

    def write_blosc(
        self,
        h5_fname: Union[str, PathLike],
        block_info: Optional[dict] = None,
        save_threshold: bool = False,
        codec: str = "zstd",
        level: int = 5,
    ):
        """Write [`linear_dag.core.lineararg.LinearARG`][] data with Blosc compression.

        This method uses Blosc compression (default: Zstd level 5 with bitshuffle)
        which provides 2-3x faster read performance compared to gzip while maintaining
        good compression ratios. Files written with this method can be read normally
        using [`linear_dag.core.lineararg.LinearARG.read`][].

        !!! info
            Reading Blosc-compressed files requires `hdf5plugin` to be installed
            in the runtime environment.

        **Arguments:**

        - `h5_fname`: Base path/prefix used for output files.
        - `block_info`: Optional dictionary with keys `chrom`, `start`, and `end`.
        - `save_threshold`: Whether to save threshold summary attributes.
        - `codec`: Blosc codec (`'zstd'`, `'lz4'`, `'lz4hc'`, `'zlib'`).
        - `level`: Blosc compression level (`0-9`).

        **Returns:**

        - `None`.

        **Raises:**

        - `ImportError`: If `hdf5plugin` is unavailable.
        - `FileExistsError`: If writing a non-block file and the output already exists.
        """
        try:
            import hdf5plugin
        except ImportError:
            raise ImportError("hdf5plugin is required for Blosc compression. ")

        fname = h5_fname if str(h5_fname).endswith(".h5") else str(h5_fname) + ".h5"
        mode = "a" if block_info else "w"
        if (not block_info) and os.path.exists(fname):
            raise FileExistsError(
                f"The file '{fname}' already exists."
                "To append a new linear ARG to an existing file, specify `block_info`."
            )

        # Helper function to get appropriate Blosc filter for a dataset
        def get_blosc_filter(data):
            """Return appropriate Blosc filter based on data type."""
            # For string dtypes, use regular shuffle instead of bitshuffle
            if data.dtype.kind in ["U", "S"]:
                return hdf5plugin.Blosc(cname=codec, clevel=level, shuffle=hdf5plugin.Blosc.SHUFFLE)
            else:
                # For numeric dtypes, use bitshuffle for better compression
                return hdf5plugin.Blosc(cname=codec, clevel=level, shuffle=hdf5plugin.Blosc.BITSHUFFLE)

        with h5py.File(fname, mode) as f:
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

            # Write main datasets with Blosc compression
            destination.create_dataset("indptr", data=self.A.indptr, compression=get_blosc_filter(self.A.indptr))
            destination.create_dataset("indices", data=self.A.indices, compression=get_blosc_filter(self.A.indices))
            destination.create_dataset("data", data=self.A.data, compression=get_blosc_filter(self.A.data))
            destination.create_dataset(
                "variant_indices", data=self.variant_indices, compression=get_blosc_filter(self.variant_indices)
            )
            destination.create_dataset("flip", data=self.flip, compression=get_blosc_filter(self.flip))

            if self.nonunique_indices is not None:
                destination.create_dataset(
                    "nonunique_indices",
                    data=self.nonunique_indices,
                    compression=get_blosc_filter(self.nonunique_indices),
                )
            if self.iids is not None and "iids" not in f.keys():
                str_iids = np.array(self.iids, dtype="S")
                f.create_dataset("iids", data=str_iids, compression=get_blosc_filter(str_iids))
            if self.n_individuals is not None:
                destination.attrs["n_individuals"] = self.n_individuals
            if self.variants is not None:
                variant_info = self.variants.collect()
                for field in ["CHROM", "POS", "ID", "REF", "ALT"]:
                    if field == "POS":
                        pos_data = np.array(variant_info[field]).astype(int)
                        destination.create_dataset(
                            field,
                            data=pos_data,
                            compression=get_blosc_filter(pos_data),
                        )
                    else:
                        str_data = np.array(variant_info[field], dtype=object)
                        destination.create_dataset(
                            field,
                            data=str_data,
                            dtype=h5py.string_dtype(encoding="utf-8"),
                            compression=get_blosc_filter(str_data),
                        )

            if save_threshold:
                N = self.A.shape[0]
                af = self.allele_frequencies
                maf = np.minimum(af, 1 - af)
                order = int(np.ceil(np.log10(N)))
                thresholds = 10.0 ** -np.arange(1, order + 1)
                destination.attrs["threshold_values"] = thresholds
                destination.attrs["threshold_n_variants"] = (maf[:, None] > thresholds).sum(axis=0)

        return

    @staticmethod
    def read_variant_info(
        h5_fname: Union[str, PathLike],
        block: Optional[str] = None,
    ) -> pl.LazyFrame:
        """Reads variant info from provided HDF5 file.

        **Arguments:**

        - `h5_fname`: Base path/prefix of the HDF5 file.
        - `block`: Optional block/group name.

        **Returns:**

        - LazyFrame containing `CHROM`, `POS`, `ID`, `REF`, and `ALT`.
        """
        fname = h5_fname if str(h5_fname).endswith(".h5") else str(h5_fname) + ".h5"
        with h5py.File(fname, "r") as file:
            f = file[block] if block else file
            v_dict = {field: f[field][:].astype(str) for field in ["CHROM", "POS", "ID", "REF", "ALT"]}
            v_info = pl.LazyFrame(
                v_dict,
                schema=[
                    ("CHROM", pl.String),
                    ("POS", pl.Int32),
                    ("ID", pl.String),
                    ("REF", pl.String),
                    ("ALT", pl.String),
                ],
            )
        return v_info

    @staticmethod
    def read(
        h5_fname: Union[str, PathLike],
        block: Optional[str] = None,
        load_metadata: bool = False,
    ) -> "LinearARG":
        """Read [`linear_dag.core.lineararg.LinearARG`][] data from HDF5 files.

        !!! info

            If `hdf5plugin` is unavailable, Blosc-compressed files may fail to
            load even though gzip/lzf-backed files can still be read.

        **Arguments:**

        - `h5_fname`: Base path/prefix of the HDF5 file.
        - `block`: Optional block/group name.
        - `load_metadata`: Whether to load variant metadata into `variants`.

        **Returns:**

        - A [`linear_dag.core.lineararg.LinearARG`][] object.

        **Raises:**

        - `ValueError`: If required datasets (e.g. `iids`) are missing.
        """
        import importlib.util as iu

        # Try to import hdf5plugin to enable Blosc decompression
        # This is needed for files written with write_blosc()
        # if None Blosc files will fail to read, but gzip/lzf files will work fine
        if iu.find_spec("hdf5plugin") is None:
            import warnings

            warnings.warn("hdf5plugin is required for blosc compression; this may impact reading")
        else:
            import hdf5plugin  # noqa: F401

        fname = h5_fname if str(h5_fname).endswith(".h5") else str(h5_fname) + ".h5"
        with h5py.File(fname, "r") as file:
            iids = pl.Series(file["iids"][:].astype(str))
            f = file[block] if block else file
            A = csc_matrix((f["data"][:], f["indices"][:], f["indptr"][:]), shape=(f.attrs["n"], f.attrs["n"]))
            variant_indices = f["variant_indices"][:]
            flip = f["flip"][:]
            n_samples = f.attrs["n_samples"]
            n_individuals = f.attrs.get("n_individuals", None)
            nonunique_indices = f["nonunique_indices"][:] if "nonunique_indices" in f else None

            if load_metadata:
                v_info = LinearARG.read_variant_info(h5_fname, block)
            else:
                v_info = None

            if "allele_counts" in f:
                allele_counts = f["allele_counts"][:]
            else:
                allele_counts = None

        linarg = LinearARG(
            A,
            variant_indices,
            flip,
            n_samples,
            n_individuals=n_individuals,
            variants=v_info,
            iids=iids,
            nonunique_indices=nonunique_indices,
        )
        if allele_counts is not None:
            linarg.set_allele_counts(allele_counts)
        return linarg

    def filter_variants_by_maf(self, maf_threshold: float) -> None:
        """Filter variants to those with minor allele frequency above a threshold.

        **Arguments:**

        - `maf_threshold`: MAF cutoff (for example `0.01` for 1%).

        **Returns:**

        - `None`. The instance is mutated in-place.
        """
        af = self.allele_frequencies
        maf = np.minimum(af, 1 - af)
        mask = maf > maf_threshold
        self.variant_indices = self.variant_indices[mask]
        self.flip = self.flip[mask]

    def filter_variants_by_mask(self, mask: npt.NDArray[np.bool_]) -> None:
        """Filter variants using a boolean keep-mask.

        **Arguments:**

        - `mask`: Boolean array with length equal to `len(self.variant_indices)`.

        **Returns:**

        - `None`. The instance is mutated in-place.

        **Raises:**

        - `ValueError`: If `mask` length does not match number of variants.
        """
        if len(mask) != len(self.variant_indices):
            raise ValueError(f"Mask length ({len(mask)}) must match number of variants ({len(self.variant_indices)})")
        self.variant_indices = self.variant_indices[mask]
        self.flip = self.flip[mask]

    def filter_variants_by_bed(
        self,
        bed_regions: "pl.DataFrame",
        maf_threshold: float = 0.0,
    ) -> None:
        """Filter variants to BED-overlapping variants above a MAF threshold.

        **Arguments:**

        - `bed_regions`: DataFrame with columns `chrom`, `chromStart`, `chromEnd`.
        - `maf_threshold`: MAF threshold applied to BED-overlapping variants.

        **Returns:**

        - `None`. The instance is mutated in-place.

        **Raises:**

        - `ValueError`: If variant metadata is not available on this
          [`linear_dag.core.lineararg.LinearARG`][].
        """
        if self.variants is None:
            raise ValueError("LinearARG must have variant metadata to filter by BED regions")

        variant_info = self.variants.collect()
        chrom = variant_info["CHROM"].to_numpy().astype(str)
        pos = variant_info["POS"].to_numpy()

        in_bed = variants_in_bed_regions(chrom, pos, bed_regions)

        af = self.allele_frequencies
        maf = np.minimum(af, 1 - af)
        maf_ok = maf > maf_threshold

        mask = in_bed & maf_ok
        self.filter_variants_by_mask(mask)

    def calculate_nonunique_indices(self) -> None:
        """Compute and cache non-unique index compression metadata.

        **Returns:**

        - `None`. Updates `self.nonunique_indices` in-place when absent.
        """
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

    @cached_property
    def num_nonunique_indices(self) -> Optional[int]:
        """Return number of compressed non-unique indices if available.

        **Returns:**

        - Integer count or `None` when non-unique indices are not initialized.
        """
        if self.nonunique_indices is None:
            return None
        return np.max(self.nonunique_indices) + 1

    def add_individual_nodes(self, sex: npt.NDArray[np.uint] = None) -> "LinearARG":
        """Create a new [`linear_dag.core.lineararg.LinearARG`][] with individual nodes.

        **Arguments:**

        - `sex`: Optional per-individual sex coding used to choose haploid vs diploid attachment.

        **Returns:**

        - New [`linear_dag.core.lineararg.LinearARG`][] instance that includes individual nodes.
        """
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
            sex=self.sex if sex is None else sex.copy(),
        )
        linarg.calculate_nonunique_indices()

        return linarg


def list_blocks(h5_fname: Union[str, PathLike]) -> pl.DataFrame:
    """List HDF5 block groups and their metadata attributes.

    **Arguments:**

    - `h5_fname`: Path to HDF5 file (`.h5` suffix is optional).

    **Returns:**

    - Polars DataFrame with one row per block, sorted by chromosome/start.
      Returns `None` when no block groups are present.
    """
    if not str(h5_fname).endswith(".h5"):
        h5_fname = str(h5_fname) + ".h5"
    block_data = []

    def parse_block_name(block_name):
        if len(block_name.split("_")) == 3:  # chr1_start_end
            chrom, start, _ = block_name.split("_")
        else:  # chr1:start-end
            chrom = block_name.split(":")[0]
            start = block_name.split(":")[1].split("-")[0]

        chrom = str(chrom)
        if chrom.startswith("chr"):
            chrom = chrom[3:]

        try:
            chrom_key = (0, int(chrom))
        except ValueError:
            # Put non-numeric chromosomes (e.g. X/Y/MT) after numeric chromosomes,
            # and sort lexicographically within that group.
            chrom_key = (1, chrom)

        try:
            start_key = int(float(start))
        except ValueError:
            start_key = float("inf")

        return (*chrom_key, start_key)

    with h5py.File(h5_fname, "r") as f:
        block_names = [b for b in list(f.keys()) if isinstance(f[b], h5py.Group) and b != "iids"]
        block_names = sorted(block_names, key=parse_block_name)

        if not block_names:
            return None
        else:
            for block_name in block_names:
                group = f[block_name]
                attrs = group.attrs
                block_info = {key: value for key, value in attrs.items()}
                block_info["block_name"] = block_name
                block_data.append(block_info)

    return pl.DataFrame(block_data)


def list_iids(h5_fname: Union[str, PathLike]) -> pl.Series:
    """Load IID labels from the root of a linear-ARG HDF5 file.

    **Arguments:**

    - `h5_fname`: Path to HDF5 file (`.h5` suffix is optional).

    **Returns:**

    - Polars Series named `iids`.

    **Raises:**

    - `ValueError`: If the `iids` dataset is missing.
    """
    if not str(h5_fname).endswith(".h5"):
        h5_fname = str(h5_fname) + ".h5"
    with h5py.File(h5_fname, "r") as h5f:
        if "iids" not in h5f.keys():
            raise ValueError("iids not found in HDF5 file")
        iids_data = h5f["iids"][:]
        iids = pl.Series("iids", iids_data.astype(str))
    return iids


def variants_in_bed_regions(
    chrom: npt.NDArray,
    pos: npt.NDArray,
    bed_regions: pl.DataFrame,
) -> npt.NDArray[np.bool_]:
    """Return a mask for variants that fall inside any BED interval.

    BED coordinates are interpreted as 0-based half-open ranges `[start, end)`.

    **Arguments:**

    - `chrom`: Chromosome label per variant.
    - `pos`: Position per variant.
    - `bed_regions`: DataFrame with columns `chrom`, `chromStart`, `chromEnd`.

    **Returns:**

    - Boolean mask with `True` for variants inside at least one BED interval.
    """
    mask = np.zeros(len(pos), dtype=bool)

    for row in bed_regions.iter_rows(named=True):
        bed_chrom = row["chrom"]
        bed_start = row["chromStart"]
        bed_end = row["chromEnd"]

        # Match chromosome and position range [start, end)
        chrom_match = chrom == bed_chrom
        pos_in_range = (pos >= bed_start) & (pos < bed_end)
        mask |= chrom_match & pos_in_range

    return mask


def compute_variant_filter_mask(
    hdf5_file: str,
    block_name: str,
    maf_threshold: float = 0.0,
    bed_regions: Optional[pl.DataFrame] = None,
    bed_maf_threshold: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """Compute a boolean mask for variant filtering based on MAF and BED regions.

    Filtering logic:
    - If bed_regions is None: include variants with MAF > maf_threshold
    - If bed_regions is provided:
      - Variants inside BED regions: include if MAF > bed_maf_threshold
      - Variants outside BED regions: include if MAF > maf_threshold

    **Arguments:**

    - `hdf5_file`: Path to HDF5 file.
    - `block_name`: Block/group name to process.
    - `maf_threshold`: MAF threshold outside BED regions.
    - `bed_regions`: Optional DataFrame with BED regions.
    - `bed_maf_threshold`: MAF threshold inside BED regions.

    **Returns:**

    - Boolean inclusion mask with one entry per variant.

    **Raises:**

    - `KeyError`: If the requested block or required datasets are missing.
    """
    with h5py.File(hdf5_file, "r") as f:
        g = f[block_name]
        af = g["allele_counts"][:] / g.attrs["n_samples"]
        maf = np.minimum(af, 1 - af)

        if bed_regions is not None:
            chrom = g["CHROM"][:].astype(str)
            pos = g["POS"][:]
            in_bed = variants_in_bed_regions(chrom, pos, bed_regions)

            # Dual threshold: bed_maf_threshold inside BED, maf_threshold outside
            mask = (in_bed & (maf > bed_maf_threshold)) | (~in_bed & (maf > maf_threshold))
        else:
            mask = maf > maf_threshold

    return mask


def compute_filtered_variant_count(
    hdf5_file: str,
    block_name: str,
    maf_threshold: float = 0.0,
    bed_regions: Optional[pl.DataFrame] = None,
    bed_maf_threshold: float = 0.0,
) -> int:
    """Compute the number of variants that pass the filter criteria.

    **Arguments:**

    - `hdf5_file`: Path to HDF5 file.
    - `block_name`: Block/group name to process.
    - `maf_threshold`: MAF threshold outside BED regions.
    - `bed_regions`: Optional DataFrame with BED regions.
    - `bed_maf_threshold`: MAF threshold inside BED regions.

    **Returns:**

    - Number of variants that pass the active filter logic.
    """
    mask = compute_variant_filter_mask(hdf5_file, block_name, maf_threshold, bed_regions, bed_maf_threshold)
    return int(np.sum(mask))


def load_variant_info(
    h5_fname: str,
    block_names: Union[list[str], None] = None,
    columns: str = "id_only",
    maf_threshold: Optional[float] = None,
):
    """Load and optionally MAF-filter variant metadata across selected blocks.

    **Arguments:**

    - `h5_fname`: Path to HDF5 file (`.h5` suffix is optional).
    - `block_names`: Optional block names to include. Defaults to all blocks.
    - `columns`: One of `"id_only"`, `"no_id"`, or `"all"`.
    - `maf_threshold`: Optional MAF cutoff used per block before concatenation.

    **Returns:**

    - Polars `LazyFrame` with requested columns.

    **Raises:**

    - `KeyError`: If selected blocks or required datasets are missing.
    """
    if not str(h5_fname).endswith(".h5"):
        h5_fname = str(h5_fname) + ".h5"

    blocks_df = list_blocks(h5_fname)
    blocks = block_names or blocks_df.get_column("block_name").to_list()
    if block_names is not None:
        blocks_df = blocks_df.filter(pl.col("block_name").is_in(block_names))

    # Initialize lists to collect data
    chrom_list, pos_list, ref_list, alt_list, id_list = [], [], [], [], []

    with h5py.File(h5_fname, "r") as f:
        for block in blocks:
            g = f[block]

            # Calculate MAF mask
            if maf_threshold is not None:
                af = g["allele_counts"][:] / g.attrs["n_samples"]
                maf = np.minimum(af, 1 - af)
                mask = maf > maf_threshold
            else:
                mask = np.ones(g["allele_counts"].shape[0], dtype=bool)

            # Only process if any variants pass the filter
            if np.any(mask):
                if columns in ("all", "no_id"):
                    chrom_list.extend(g["CHROM"][:][mask])
                    pos_list.extend(g["POS"][:][mask])
                    ref_list.extend(g["REF"][:][mask])
                    alt_list.extend(g["ALT"][:][mask])
                if columns in ("all", "id_only"):
                    id_list.extend(g["ID"][:][mask])

    # Build LazyFrame from collected data
    data_dict = {}
    schema_list = []

    if columns in ("all", "no_id"):
        data_dict.update(
            {
                "CHROM": chrom_list,
                "POS": pos_list,
                "REF": ref_list,
                "ALT": alt_list,
            }
        )
        schema_list.extend(
            [
                ("CHROM", pl.Binary),
                ("POS", pl.Int32),
                ("REF", pl.Binary),
                ("ALT", pl.Binary),
            ]
        )

    if columns in ("all", "id_only"):
        data_dict["ID"] = id_list
        schema_list.append(("ID", pl.Binary))

    return pl.LazyFrame(data_dict, schema=schema_list)


def add_individuals_to_graph(
    A: csc_matrix,
    samples_idx: npt.NDArray[np.uint],
    sex: npt.NDArray[np.uint] = None,
) -> tuple:
    """Add individual nodes that connect to their constituent sample haplotypes.

    Assumes each individual corresponds to adjacent haplotypes in `samples_idx`.
    If `sex` is provided, entries with value `1` are treated as haploid.

    **Arguments:**

    - `A`: Existing adjacency matrix.
    - `samples_idx`: Sample-node indices in haplotype order.
    - `sex`: Optional per-individual sex coding array.

    **Returns:**

    - Tuple `(A_updated, individual_indices)`.
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
    """Remove degree-zero internal nodes while keeping variant/sample nodes.

    **Arguments:**

    - `A`: Input adjacency matrix.
    - `variant_indices`: Variant-node indices that must be retained.
    - `sample_indices`: Sample-node indices that must be retained.

    **Returns:**

    - Tuple `(A_filtered, variant_indices_reindexed, sample_indices_reindexed)`.

    **Raises:**

    - `AssertionError`: If provided indices are out of bounds.
    """
    node_degree = A.getnnz(axis=0) + A.getnnz(axis=1)
    nonzero_indices = set(np.where(node_degree > 0)[0])
    required_indices = set(variant_indices).union(sample_indices)
    assert all(idx < A.shape[0] for idx in nonzero_indices)
    assert all(idx < A.shape[0] for idx in variant_indices)
    assert all(idx < A.shape[0] for idx in sample_indices)
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
    """Topologically reorder graph nodes and reindex variant nodes accordingly.

    Sample/leaf nodes are placed at the tail in reverse sample order.

    **Arguments:**

    - `A`: Input adjacency matrix.
    - `variant_indices`: Variant-node indices for `A`.
    - `sample_indices`: Sample-node indices for `A`.

    **Returns:**

    - Tuple `(A_triangular, variant_indices_reordered)`.
    """
    A_csr = csr_matrix(A)
    order = np.asarray(topological_sort(A_csr, nodes_to_ignore=set(sample_indices)))[: -len(sample_indices)]
    order = np.append(order, sample_indices[::-1])
    inv_order = np.argsort(order).astype(np.int32)

    A_triangular = A[order, :][:, order]
    variant_indices_reordered = inv_order[variant_indices]

    return A_triangular, variant_indices_reordered
