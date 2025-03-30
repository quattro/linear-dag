# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from .data_structures cimport Stack
cimport scipy.linalg.cython_blas as blas

def spsolve_forward_triangular(A: "csr_matrix", b: np.ndarray) -> None:
    """Solves the system (I-A)x = b in place, where A is a lower triangular zero-diagonal CSR matrix.
    
    Parameters
    ----------
    A : csr_matrix
        Lower triangular zero-diagonal CSR matrix
    b : ndarray
        Right-hand side vector, must be float64 dtype
    """
    _spsolve_forward_triangular_float64(A.indptr, A.indices, A.data, b)

cdef void _spsolve_forward_triangular_float64(int[:] indptr, int[:] indices, int[:] data, double[:] b_view) nogil:
    cdef int node_idx, edge_idx = 0
    cdef int num_nodes = len(indptr) - 1
    
    for node_idx in range(num_nodes):
        while edge_idx < indptr[node_idx + 1]:
            b_view[node_idx] += b_view[indices[edge_idx]] * data[edge_idx]
            edge_idx += 1

def spsolve_backward_triangular(A: "csr_matrix", b: np.ndarray) -> None:
    """Solves the system (I-A)'x = b in place, where A is a lower triangular zero-diagonal matrix."""
    
    cdef int num_nodes = len(A.indptr) - 1
    cdef int node_idx
    cdef int edge_idx = A.indptr[num_nodes]
    
    cdef int[:] indptr = A.indptr
    cdef int[:] indices = A.indices
    cdef int[:] data = A.data
    cdef double[:] b_view = b

    with nogil:
        for node_idx in range(num_nodes - 1, -1, -1):
            while edge_idx > indptr[node_idx]:
                edge_idx -= 1
                b_view[indices[edge_idx]] += b_view[node_idx] * data[edge_idx]


def spsolve_forward_triangular_matmat(A: "csr_matrix", b: np.ndarray) -> None:
    """Solves (I-A)x' = b' in place, where A is a lower-triangular, zero-diagonal CSR matrix.
    Assumes that b is Fortran-contiguous."""
    
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    if b.dtype == np.float64:
        _spsolve_forward_triangular_matmat_float64(A.indptr, A.indices, A.data, b)
    elif b.dtype == np.float32:
        _spsolve_forward_triangular_matmat_float32(A.indptr, A.indices, A.data, b)
    else:
        # Fall back to float64
        b_copy = b.astype(np.float64)
        _spsolve_forward_triangular_matmat_float64(A.indptr, A.indices, A.data, b_copy)
        b[:] = b_copy

cdef void _spsolve_forward_triangular_matmat_float64(int[:] indptr, int[:] indices, int[:] data, double[:, :] b_view) nogil:
    cdef int node_idx, edge_idx = 0
    cdef int num_nodes = len(indptr) - 1
    cdef int vector_length = b_view.shape[0]
    cdef int* vector_length_ptr = &vector_length
    cdef int inc = 1
    cdef int* inc_ptr = &inc
    cdef double alpha
    cdef double* alpha_ptr = &alpha

    # Get a pointer to the underlying data of b_view
    cdef double* b_ptr = &b_view[0, 0]
    cdef double* source_ptr
    cdef double* destination_ptr = b_ptr 

    for node_idx in range(num_nodes):
        while edge_idx < indptr[node_idx + 1]:
            alpha = <double> data[edge_idx]
            source_ptr = b_ptr + indices[edge_idx] * vector_length

            # Call the BLAS axpy routine for destination += alpha * source
            blas.daxpy(vector_length_ptr, alpha_ptr, source_ptr, inc_ptr, destination_ptr, inc_ptr)
            
            edge_idx += 1
        
        destination_ptr += vector_length

cdef void _spsolve_forward_triangular_matmat_float32(int[:] indptr, int[:] indices, int[:] data, float[:, :] b_view) nogil:
    cdef int node_idx, edge_idx = 0
    cdef int num_nodes = len(indptr) - 1
    cdef int vector_length = b_view.shape[0]
    cdef int* vector_length_ptr = &vector_length
    cdef int inc = 1
    cdef int* inc_ptr = &inc
    cdef float alpha
    cdef float* alpha_ptr = &alpha

    # Get a pointer to the underlying data of b_view
    cdef float* b_ptr = &b_view[0, 0]
    cdef float* source_ptr
    cdef float* destination_ptr = b_ptr 

    for node_idx in range(num_nodes):
        while edge_idx < indptr[node_idx + 1]:
            alpha = <float> data[edge_idx]
            source_ptr = b_ptr + indices[edge_idx] * vector_length

            # Call the BLAS axpy routine for destination += alpha * source
            blas.saxpy(vector_length_ptr, alpha_ptr, source_ptr, inc_ptr, destination_ptr, inc_ptr)
            
            edge_idx += 1
        
        destination_ptr += vector_length


def spsolve_backward_triangular_matmat(A: "csr_matrix", b: np.ndarray) -> None:
    """Solves (I-A)'x' = b' in place, where A is a lower-triangular, zero-diagonal CSR matrix.
    Assumes that b is Fortran-contiguous."""
    
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    if b.dtype == np.float64:
        _spsolve_backward_triangular_matmat_float64(A.indptr, A.indices, A.data, b)
    elif b.dtype == np.float32:
        _spsolve_backward_triangular_matmat_float32(A.indptr, A.indices, A.data, b)
    else:
        # Fall back to float64
        b_copy = b.astype(np.float64)
        _spsolve_backward_triangular_matmat_float64(A.indptr, A.indices, A.data, b_copy)
        b[:] = b_copy

cdef void _spsolve_backward_triangular_matmat_float64(int[:] indptr, int[:] indices, int[:] data, double[:, :] b_view) nogil:
    cdef int num_nodes = len(indptr) - 1
    cdef int node_idx
    cdef int edge_idx = indptr[num_nodes]
    cdef int vector_length = b_view.shape[0]
    cdef int* vector_length_ptr = &vector_length
    cdef int inc = 1
    cdef int* inc_ptr = &inc
    cdef double alpha
    cdef double* alpha_ptr = &alpha

    # Pointers into b_view
    cdef double* b_ptr = &b_view[0, 0]
    cdef double* source_ptr = b_ptr + vector_length * (num_nodes - 1)
    cdef double* destination_ptr 

    for node_idx in range(num_nodes - 1, -1, -1):
        while edge_idx > indptr[node_idx]:
            edge_idx -= 1
            alpha = <double> data[edge_idx]
            destination_ptr = b_ptr + indices[edge_idx] * vector_length

            # Call the BLAS axpy routine for destination += alpha * source
            blas.daxpy(vector_length_ptr, alpha_ptr, source_ptr, inc_ptr, destination_ptr, inc_ptr)
        
        source_ptr -= vector_length

cdef void _spsolve_backward_triangular_matmat_float32(int[:] indptr, int[:] indices, int[:] data, float[:, :] b_view) nogil:
    cdef int num_nodes = len(indptr) - 1
    cdef int node_idx
    cdef int edge_idx = indptr[num_nodes]
    cdef int vector_length = b_view.shape[0]
    cdef int* vector_length_ptr = &vector_length
    cdef int inc = 1
    cdef int* inc_ptr = &inc
    cdef float alpha
    cdef float* alpha_ptr = &alpha

    # Pointers into b_view
    cdef float* b_ptr = &b_view[0, 0]
    cdef float* source_ptr = b_ptr + vector_length * (num_nodes - 1)
    cdef float* destination_ptr 

    for node_idx in range(num_nodes - 1, -1, -1):
        while edge_idx > indptr[node_idx]:
            edge_idx -= 1
            alpha = <float> data[edge_idx]
            destination_ptr = b_ptr + indices[edge_idx] * vector_length

            # Call the BLAS axpy routine for destination += alpha * source
            blas.saxpy(vector_length_ptr, alpha_ptr, source_ptr, inc_ptr, destination_ptr, inc_ptr)
        
        source_ptr -= vector_length

def add_at(destination: np.ndarray, indices: np.ndarray, source: np.ndarray):
    """
    Adds source values to destination at specified indices.
    Equivalent to np.add.at(destination, (np.arange(destination.shape[0]).reshape(-1,1), indices.reshape(1,-1)), source)
    but much faster.
    
    Parameters:
    -----------
    destination : np.ndarray, shape (n_cols, n_rows)
        The target array to add values to, must be Fortran-contiguous
    indices : np.ndarray, shape (n_indices,)
        The indices to add values at
    source : np.ndarray, shape (n_cols, n_indices)
        The source values to add, must be Fortran-contiguous
    """
    if destination.dtype == np.float64:
        _add_at_float64(destination, indices, source)
    elif destination.dtype == np.float32:
        _add_at_float32(destination, indices, source)
    else:
        # Fall back to float64
        dest_copy = destination.astype(np.float64)
        source_copy = source.astype(np.float64)
        _add_at_float64(dest_copy, indices, source_copy)
        destination[:] = dest_copy

cdef void _add_at_float64(double[:, :] dest_view, long[:] indices_view, double[:, :] source_view) nogil:
    cdef int n_cols = dest_view.shape[0]
    cdef int n_rows = dest_view.shape[1]
    cdef int n_indices = indices_view.shape[0]
    
    cdef int col_idx, idx_pos, dest_idx
    
    for idx_pos in range(n_indices):
        for col_idx in range(n_cols):
            dest_idx = indices_view[idx_pos]
            dest_view[col_idx, dest_idx] += source_view[col_idx, idx_pos]

cdef void _add_at_float32(float[:, :] dest_view, long[:] indices_view, float[:, :] source_view) nogil:
    cdef int n_cols = dest_view.shape[0]
    cdef int n_rows = dest_view.shape[1]
    cdef int n_indices = indices_view.shape[0]
    
    cdef int col_idx, idx_pos, dest_idx
    
    for idx_pos in range(n_indices):
        for col_idx in range(n_cols):
            dest_idx = indices_view[idx_pos]
            dest_view[col_idx, dest_idx] += source_view[col_idx, idx_pos]

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
