# linear_arg_inference.py
import warnings

import numpy as np

from scipy.sparse import block_diag, csc_matrix, csr_matrix, eye, hstack, triu, vstack

from .solve import spinv_make_triangular, spinv_triangular, topological_sort


def infer_brick_graph_using_containment(genotypes: csc_matrix, ploidy) -> csr_matrix:
    more_than_one_carrier = np.diff(genotypes.indptr) > 1
    for n in range(1, ploidy + 1):
        X_carrier = genotypes[:, more_than_one_carrier] >= n
        X_carrier = X_carrier.astype(np.int32)
        R_carrier = X_carrier.transpose().dot(X_carrier)

        temp = R_carrier.copy()
        for i in range(R_carrier.shape[0]):
            row_data = R_carrier.data[R_carrier.indptr[i]: R_carrier.indptr[i + 1]]
            temp.data[temp.indptr[i]: temp.indptr[i + 1]] = row_data >= R_carrier[i, i]

        if n == 1:
            brick_graph_closure = temp
        else:
            brick_graph_closure = brick_graph_closure.multiply(temp)

    return brick_graph_closure


def remove_undirected_edges(adjacency_matrix):
    ties = adjacency_matrix.multiply(adjacency_matrix.transpose())
    result = adjacency_matrix - triu(ties, k=1)
    result.eliminate_zeros()
    return result


def linearize_brick_graph(brick_graph_closure: csr_matrix) -> csr_matrix:
    brick_graph_closure = remove_undirected_edges(brick_graph_closure)
    row_counts = np.diff(brick_graph_closure.indptr)
    triangular_order = np.argsort(-row_counts)
    original_order = np.argsort(triangular_order)
    brick_graph_closure = brick_graph_closure[triangular_order, :][:, triangular_order].tocsc()
    brick_graph_closure.sort_indices()
    A = spinv_triangular(brick_graph_closure)
    A = eye(A.indptr.shape[0] - 1) - A
    A = A[original_order, :][:, original_order]
    return csr_matrix(A)


def add_singleton_variants(genotypes: csc_matrix, initial_linear_arg: csr_matrix) -> csr_matrix:
    one_or_zero_carriers = np.where(np.diff(genotypes.indptr) <= 1)[0]
    more_than_one_carrier = np.where(np.diff(genotypes.indptr) > 1)[0]
    variant_ordering = np.argsort(np.concatenate((more_than_one_carrier, one_or_zero_carriers)))
    A = csr_matrix(block_diag((initial_linear_arg, eye(len(one_or_zero_carriers)))))
    A = A[variant_ordering, :][:, variant_ordering]
    return A


def add_samples_to_linear_arg(genotypes: csc_matrix, initial_linear_arg: csr_matrix) -> csr_matrix:
    n, m = genotypes.shape
    assert initial_linear_arg.shape == (m, m), "linear arg should have one node per variant in the genotype matrix"
    A_sample = genotypes - genotypes @ initial_linear_arg
    zeros_matrix = csr_matrix((m + n, n))
    vertical_stack = vstack([A_sample, initial_linear_arg])
    A = hstack([zeros_matrix, vertical_stack])
    A = A.astype(np.int32)

    return csr_matrix(A)


def add_samples_to_brick_graph_closure(genotypes: csc_matrix, brick_graph_closure: csr_matrix) -> csr_matrix:
    n, m = genotypes.shape
    result = vertcat(csr_matrix(genotypes), brick_graph_closure)
    pad_leading_zeros(result, n)
    return result


def vertcat(A: csr_matrix, B: csr_matrix) -> csr_matrix:
    indptrs = np.concatenate((A.indptr, A.indptr[-1] + B.indptr[1:]))
    indices = np.concatenate((A.indices, B.indices))
    data = np.concatenate((A.data, B.data))
    return csr_matrix((data, indices, indptrs))


def pad_leading_zeros(A: csr_matrix, num_cols: int) -> None:
    A = csr_matrix((A.data, A.indices + num_cols, A.indptrs))


def add_samples_to_initial_brick_graph(genotypes: csc_matrix, brick_graph: csr_matrix) -> csr_matrix:
    n, m = genotypes.shape
    assert brick_graph.shape == (m, m), "brick_graph should have one node per variant in the genotype matrix"
    assert np.all(genotypes.data == 1), "this doesn't work with unphased data"
    assert np.all(brick_graph.diagonal() == 0), "requires unit diagonal"

    indices = []
    indptrs = [0]

    for row in range(n):
        row_data_indices = range(genotypes.indptr[row], genotypes.indptr[row + 1])
        ancestors = set(genotypes.indices[row_data_indices])
        parents = []
        for ancestor in ancestors:
            ancestor_indices = range(genotypes.indptr[ancestor], genotypes.indptr[ancestor + 1])
            ancestor_parents = set(genotypes.indices[ancestor_indices])
            if ancestor_parents.isdisjoint(ancestors):
                parents.append(ancestor)

        indices += sorted(parents)
        indptrs.append(indptrs[-1] + len(parents))

    result = csc_matrix((np.ones_like(indices), indices, indptrs))
    return add_samples_to_brick_graph_closure(result, brick_graph)


def closure_transitive_reduction(transitive_graph: csr_matrix) -> csr_matrix:
    if np.any(transitive_graph.data != 1):
        raise ValueError("Edge weights of the transitive graph should be one")
    number_of_paths = path_sum(transitive_graph)
    number_of_paths.data[number_of_paths.data != 1] = 0
    number_of_paths.eliminate_zeros()
    return number_of_paths


def transitive_closure(graph: csr_matrix) -> csr_matrix:
    number_of_paths = path_sum(graph)
    number_of_paths.data = np.ones_like(number_of_paths.data)
    return number_of_paths


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
