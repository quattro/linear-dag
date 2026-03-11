# linear_arg_inference.py
import logging
import time
import warnings

import numpy as np
import polars as pl

from scipy.sparse import csc_matrix, csr_matrix, eye

from .brick_graph import BrickGraph
from .one_summed_cy import linearize_brick_graph
from .recombination import Recombination
from .solve import spinv_make_triangular


def linear_arg_from_genotypes(
    genotypes,
    flip,
    variant_info,
    find_recombinations,
    logger: logging.Logger | None = None,
):
    """Infer a linearized ARG adjacency matrix from sparse genotypes.

    Let $G$ denote the phased genotype matrix with haplotypes on rows and
    variants on columns. This routine constructs a brick graph from $G$,
    optionally identifies recombination nodes, and returns a linearized sparse
    adjacency matrix ready for
    [`linear_dag.core.lineararg.LinearARG`][] construction.

    !!! info

        The returned adjacency matrix is not yet cleaned for disconnected
        internal nodes or topologically reordered sample/variant indices; those
        later steps are handled by
        [`linear_dag.core.lineararg.LinearARG.from_genotypes`][].

    **Arguments:**

    - `genotypes`: Sparse CSC genotype matrix (`samples x variants`).
    - `flip`: Boolean flip vector aligned to input variants.
    - `variant_info`: Optional variant metadata DataFrame.
    - `find_recombinations`: Whether to explicitly infer recombination events.
    - `logger`: Optional logger for timing/progress output.

    **Returns:**

    - Tuple `(adjacency, flip, variant_indices, sample_indices, variant_info)` for
      constructing a [`linear_dag.core.lineararg.LinearARG`][].

    **Raises:**

    - `TypeError`: If `genotypes` is not a CSC sparse matrix.
    """

    if type(genotypes) is not csc_matrix:
        raise TypeError

    def _emit_progress(message: str) -> None:
        if logger is not None:
            logger.info(message)

    _emit_progress("Inferring brick graph")
    t0 = time.time()
    brick_graph, samples_idx, variants_idx = BrickGraph.from_genotypes(genotypes)
    t1 = time.time()
    _emit_progress(f"  Time: {t1 - t0:.3f}s")

    _emit_progress("Finding recombinations")
    t0 = time.time()
    recom = Recombination.from_graph(brick_graph)
    if find_recombinations:
        recom.find_recombinations()
    t1 = time.time()
    _emit_progress(f"  Time: {t1 - t0:.3f}s")

    _emit_progress("Linearizing brick graph")
    t0 = time.time()
    linear_arg_adjacency_matrix = csc_matrix(linearize_brick_graph(recom))
    t1 = time.time()
    _emit_progress(f"  Time: {t1 - t0:.3f}s")

    num_variants = len(variants_idx)
    if variant_info is None:
        data = {
            "CHROM": np.zeros(num_variants),
            "POS": np.arange(num_variants),
            "REF": np.zeros(num_variants),
            "ALT": np.ones(num_variants),
            "FLIP": np.zeros(num_variants),
            "ID": np.arange(num_variants),
        }
        variant_info = pl.DataFrame(data)

    return linear_arg_adjacency_matrix, flip, variants_idx, samples_idx, variant_info


def infer_brick_graph_using_containment(genotypes: csc_matrix, ploidy) -> csr_matrix:
    """Build a brick-graph containment closure from genotype carrier relationships.

    For each allele count level $n \\in \\{1, \\dots, \\text{ploidy}\\}$, the
    routine builds a carrier-incidence matrix and marks variant pairs whose
    carrier sets are nested. The final graph is the intersection of those
    containment relations across allele-count levels.

    **Arguments:**

    - `genotypes`: Sparse CSC genotype matrix.
    - `ploidy`: Maximum allele count per sample/haplotype.

    **Returns:**

    - CSR adjacency/closure matrix representing containment across variants.
    """

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
            brick_graph_closure = temp
        else:
            brick_graph_closure = brick_graph_closure.multiply(temp)

    return brick_graph_closure


def pad_trailing_zeros(A: csr_matrix, num_cols: int) -> csr_matrix:
    """Append zero columns to the right side of a CSR matrix.

    **Arguments:**

    - `A`: Input CSR matrix.
    - `num_cols`: Number of zero columns to append.

    **Returns:**

    - CSR matrix with `num_cols` trailing all-zero columns.
    """

    return csr_matrix((A.data, A.indices, A.indptr), shape=(A.shape[0], num_cols + A.shape[1]))


def pad_leading_zeros(A: csr_matrix, num_cols: int) -> csr_matrix:
    """Append zero columns to the left side of a CSR matrix.

    **Arguments:**

    - `A`: Input CSR matrix.
    - `num_cols`: Number of zero columns to prepend.

    **Returns:**

    - CSR matrix with `num_cols` leading all-zero columns.
    """

    return csr_matrix((A.data, A.indices + num_cols, A.indptr), shape=(A.shape[0], num_cols + A.shape[1]))


def path_sum(graph: csr_matrix) -> csr_matrix:
    """Compute the path-sum operator $(I - A)^{-1}$ for a sparse DAG adjacency.

    Let $A$ denote `graph`. For a triangular DAG adjacency, the returned matrix
    corresponds to the path-count or reachability-sum operator obtained from
    $(I - A)^{-1}$.

    **Arguments:**

    - `graph`: CSR adjacency matrix.

    **Returns:**

    - CSR matrix containing inferred path counts.
    """

    IminusA = csr_matrix(eye(graph.shape[0]) - graph)
    IminusA.eliminate_zeros()
    assert np.all(IminusA.diagonal() == 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore overflow error
        number_of_paths = spinv_make_triangular(IminusA)
    return csr_matrix(number_of_paths)


def setdiag(matrix: csr_matrix, value: int) -> None:
    """Set the diagonal of a CSR matrix despite SciPy `setdiag` edge cases.

    !!! info

        This helper exists because `scipy.sparse.csr_matrix.setdiag` can leave
        some diagonal entries unchanged for the sparsity patterns encountered in
        this module.

    **Arguments:**

    - `matrix`: CSR matrix to modify in place.
    - `value`: Diagonal value to assign.

    **Returns:**

    - `None`.
    """
    matrix.setdiag(value)
    if np.all(matrix.diagonal() == value):  # sometimes not, unclear why
        return

    for i in range(1, len(matrix.indptr)):
        for j in range(matrix.indptr[i - 1], matrix.indptr[i]):
            if matrix.indices[j] == i - 1:
                matrix.data[j] = value
    matrix.eliminate_zeros()
    assert np.all(matrix.diagonal() == value)
