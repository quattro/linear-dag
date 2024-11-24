# linear_arg_inference.py
import warnings

import numpy as np
import polars as pl

from scipy.sparse import csc_matrix, csr_matrix, eye

from .brick_graph import BrickGraph
from .one_summed_cy import linearize_brick_graph
from .recombination import Recombination
from .solve import spinv_make_triangular


def linear_arg_from_genotypes(genotypes, variant_info, find_recombinations, verbosity):
    if type(genotypes) is not csc_matrix:
        raise TypeError

    if verbosity > 0:
        print("Inferring brick graph")
    brick_graph, samples_idx, variants_idx = BrickGraph.from_genotypes(genotypes)

    if verbosity > 0:
        print("Finding recombinations")
    recom = Recombination.from_graph(brick_graph)
    if find_recombinations:
        recom.find_recombinations()

    if verbosity > 0:
        print("Linearizing brick graph")
    linear_arg_adjacency_matrix = linearize_brick_graph(recom)

    num_variants = len(variants_idx)
    if variant_info is None:
        data = {
            "CHROM": np.zeros(num_variants),
            "POS": np.arange(num_variants),
            "REF": np.zeros(num_variants),
            "ALT": np.ones(num_variants),
            "FLIP": np.zeros(num_variants),
            "ID": np.arange(num_variants),
            "INFO": np.zeros(num_variants),
        }
        variant_info = pl.DataFrame(data)
    variant_info = variant_info.with_columns(pl.lit(np.asarray(variants_idx)).alias("IDX"))

    return linear_arg_adjacency_matrix, samples_idx, variant_info


def infer_brick_graph_using_containment(genotypes: csc_matrix, ploidy) -> csr_matrix:
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
    return csr_matrix((A.data, A.indices, A.indptr), shape=(A.shape[0], num_cols + A.shape[1]))


def pad_leading_zeros(A: csr_matrix, num_cols: int) -> csr_matrix:
    return csr_matrix((A.data, A.indices + num_cols, A.indptr), shape=(A.shape[0], num_cols + A.shape[1]))


def path_sum(graph: csr_matrix) -> csr_matrix:
    IminusA = csr_matrix(eye(graph.shape[0]) - graph)
    IminusA.eliminate_zeros()
    assert np.all(IminusA.diagonal() == 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # Ignore overflow error
        number_of_paths = spinv_make_triangular(IminusA)
    return csr_matrix(number_of_paths)


def setdiag(matrix: csr_matrix, value: int) -> None:
    """
    Workaround for bug in scipy.sparse.csr_matrix.setdiag()
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
