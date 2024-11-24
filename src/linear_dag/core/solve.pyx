import numpy as np
cimport numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from .data_structures cimport Stack


def spinv_make_triangular(sparse_matrix: "csr_matrix") -> "csc_matrix":
    """
    Inverse of a sparse matrix with unit diagonal that can be made triangular through a permutation of rows and columns.
    :param sparse_matrix: CSR matrix with unit diagonal s.t. a permutation of rows and columns makes it triangular
    :return: inverse as a CSC matrix
    """
    if np.any(sparse_matrix.diagonal() != 1):
        raise ValueError("This function requires an input matrix having unit diagonal")

    order = topological_sort(sparse_matrix)
    inverse_order = np.argsort(order)
    result = sparse_matrix[order, :][:, order]
    result.sort_indices()
    result = spinv_triangular(result)
    return result[inverse_order, :][:, inverse_order]


def spinv_triangular(A: "csr_matrix") -> "csc_matrix":
    """
    Inverse of a lower-triangular sparse CSR matrix with unit diagonal and sorted indices.
    """
    cdef int[:] indptr = A.indptr.astype(np.intc)
    cdef int[:] indices = A.indices.astype(np.intc)
    cdef int[:] data = A.data.astype(np.intc)
    cdef int n = len(indptr)
    cdef int nz = len(data)

    x_indptr = np.zeros(n, dtype=np.intc)
    x_indices = np.zeros(nz, dtype=np.intc)
    x_data = np.zeros(nz, dtype=np.intc)
    dense_vector = np.zeros(n, dtype=np.intc)
    last_touched = -np.ones(n, dtype=np.intc)

    cdef int next_x_indptr = 0
    cdef int i, j, k, l, m, nj

    for i in range(n - 1):
        x_indptr[i] = next_x_indptr

        # Data range of row i
        indptr_start = indptr[i]
        indptr_stop = indptr[i + 1]
        nj = indptr_stop - indptr_start

        assert(data[indptr_stop - 1] == 1, "Input should be a lower-triangular CSR matrix with unit diagonal")
        assert(indices[indptr_stop - 1] == i, "Input should be a lower-triangular CSR matrix with unit diagonal")

        # Column indices of off-diagonal elements
        col_idx = indices[indptr_start : indptr_stop - 1]

        # Compute the dot product, storing in a full matrix
        for k in range(nj - 1):
            j = col_idx[k]
            x_col_indices = x_indices[x_indptr[j]:x_indptr[j+1]]
            for m in range(len(x_col_indices)):
                l = x_col_indices[m]
                if last_touched[l] < i:
                    last_touched[l] = i
                    dense_vector[l] = 0
                dense_vector[l] += data[k + indptr_start] * x_data[m + x_indptr[j]]

        next_x_indptr = x_indptr[i]

        # Convert the dense vector to sparse
        for k in range(nj - 1):
            j = col_idx[k]
            row_idx = x_indices[x_indptr[j]:x_indptr[j + 1]]
            for l in row_idx:
                if dense_vector[l] == 0:
                    continue

                # Check if more space is needed
                if next_x_indptr >= nz:
                    nz *= 2
                    x_indices.resize(nz, refcheck=False)
                    x_data.resize(nz, refcheck=False)

                x_indices[next_x_indptr] = l
                x_data[next_x_indptr] = -dense_vector[l]
                dense_vector[l] = 0
                next_x_indptr += 1

        # Diagonal element
        if next_x_indptr >= nz:
            nz *= 2
            x_indices.resize(nz, refcheck=False)
            x_data.resize(nz, refcheck=False)
        x_indices[next_x_indptr] = i
        x_data[next_x_indptr] = 1
        next_x_indptr += 1

    x_indptr[i+1] = next_x_indptr
    return csc_matrix((x_data[:next_x_indptr], x_indices[:next_x_indptr], x_indptr))

def topological_sort(A: "csr_matrix") -> np.ndarray:
    """
    The topological sort of a directed graph with adjacency matrix equal to A transpose. If A[i,j] != 0,
    then j will come before i in the ordering. The diagonal of A is ignored.
    """

    cdef int num_nodes = A.shape[0]
    if A.shape[1] != num_nodes:
        raise ValueError

    A_csc: csc_matrix = csc_matrix(A)
    cdef int[:] csc_indptr = A_csc.indptr
    cdef int[:] csc_indices = A_csc.indices

    cdef np.ndarray num_unvisited_nonself_parents = np.diff(A.indptr.astype(np.intc))
    num_unvisited_nonself_parents -= A.diagonal().astype(np.intc)

    # nodes_to_visit initialized with nodes having in-degree 0
    cdef Stack nodes_to_visit = Stack()
    cdef int i
    for i in np.where(num_unvisited_nonself_parents == 0)[0]:
        nodes_to_visit.push(i)

    cdef np.ndarray result = np.empty(num_nodes, dtype=np.intc)
    cdef int node, child
    i = 0
    while nodes_to_visit.length > 0:
        node = nodes_to_visit.pop()
        result[i] = node

        # Add children to nodes_to_visit once all of their other parents have been visited
        for child in csc_indices[csc_indptr[node] : csc_indptr[node+1]]:
            num_unvisited_nonself_parents[child] -= 1
            if child == node:
                assert num_unvisited_nonself_parents[child] == -1
            if num_unvisited_nonself_parents[child] == 0:
                nodes_to_visit.push(child)
        i += 1

    if i != num_nodes:
        raise ValueError("Adjacency matrix is not acyclic")

    return result
