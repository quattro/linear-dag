# linear_arg_inference.py
import numpy as np

from scipy.sparse import block_diag, csc_matrix, csr_matrix, eye, hstack, triu, vstack

from .solve import spinv_triangular


def infer_brick_graph_using_containment(genotypes, ploidy):
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


def remove_undirected_edges(adjacency_matrix):
    ties = adjacency_matrix.multiply(adjacency_matrix.transpose())
    result = adjacency_matrix - triu(ties, k=1)
    result.eliminate_zeros()
    return result


def linearize_brick_graph(brick_graph_closure: csc_matrix) -> csc_matrix:
    row_counts = np.diff(brick_graph_closure.indptr)
    triangular_order = np.argsort(-row_counts)
    original_order = np.argsort(triangular_order)
    brick_graph_closure = brick_graph_closure[triangular_order, :][:, triangular_order].tocsc()
    brick_graph_closure.sort_indices()
    indptr, indices, data = spinv_triangular(
        brick_graph_closure.indptr.astype(np.intc),
        brick_graph_closure.indices.astype(np.intc),
        brick_graph_closure.data.astype(np.intc),
    )

    A = eye(indptr.shape[0] - 1) - csc_matrix((data, indices, indptr))
    A = A[original_order, :][:, original_order]
    return A


def add_singleton_variants(genotypes: csc_matrix, initial_linear_arg: csc_matrix) -> csc_matrix:
    one_or_zero_carriers = np.where(np.diff(genotypes.indptr) <= 1)[0]
    more_than_one_carrier = np.where(np.diff(genotypes.indptr) > 1)[0]
    variant_ordering = np.argsort(np.concatenate((more_than_one_carrier, one_or_zero_carriers)))
    A = csc_matrix(block_diag((initial_linear_arg, eye(len(one_or_zero_carriers)))))
    A = A[variant_ordering, :][:, variant_ordering]
    return A


def add_samples_to_linear_arg(genotypes: csc_matrix, initial_linear_arg: csr_matrix) -> csr_matrix:
    n, m = genotypes.shape
    assert initial_linear_arg.shape == (m, m), "linear arg should have one node per variant in the genotype matrix"
    A_sample = genotypes - genotypes @ initial_linear_arg

    # Concatenate
    zeros_matrix = csr_matrix((m + n, n))
    vertical_stack = vstack([A_sample, initial_linear_arg])
    A = hstack([zeros_matrix, vertical_stack])
    A = A.astype(np.int32)

    return csr_matrix(A)
