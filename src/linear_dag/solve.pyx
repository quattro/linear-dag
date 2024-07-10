import numpy as np
cimport numpy as np

# Inverse of a lower-triangular sparse CSR matrix, or upper-triangular CSC matrix, with unit diagonal
cpdef spinv_triangular(int[:] indptr, int[:] indices, int[:] data):

    cdef int n = len(indptr)
    cdef int nz = len(data)

    x_indptr = np.zeros(n, dtype=np.intc)
    x_indices = np.zeros(5 * nz, dtype=np.intc) # TODO how much space needed?
    x_data = np.zeros(5 * nz, dtype=np.intc)
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

        # Diagonal should be all ones
        assert(data[indptr_stop - 1] == 1)
        assert(indices[indptr_stop - 1] == i)

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

        # Convert the dense vector to sparse
        next_x_indptr = x_indptr[i]
        for k in range(nj - 1):
            j = col_idx[k]
            row_idx = x_indices[x_indptr[j]:x_indptr[j + 1]]
            for l in row_idx:
                if dense_vector[l] == 0:
                    continue
                x_indices[next_x_indptr] = l
                x_data[next_x_indptr] = -dense_vector[l]
                dense_vector[l] = 0
                next_x_indptr += 1

        # Diagonal element
        x_indices[next_x_indptr] = i
        x_data[next_x_indptr] = 1
        next_x_indptr += 1

    x_indptr[i+1] = next_x_indptr
    return x_indptr, x_indices[:next_x_indptr], x_data[:next_x_indptr]
