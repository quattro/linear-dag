import linear_dag as ld
import numpy as np

from scipy.sparse import csr_matrix


def main():
    # Create a small lower-triangular matrix with unit diagonal
    L = np.array([[1, 0, 0], [2, 1, 0], [3, 0, 1]])

    # Convert the matrix to CSR format
    L_csr = csr_matrix(L)
    L_csr.sort_indices()

    # Extract CSR components
    indptr = L_csr.indptr.astype(np.int32)
    indices = L_csr.indices.astype(np.int32)
    data = L_csr.data.astype(np.int32)

    # Call the spinv_triangular function
    x_indptr, x_indices, x_data = ld.spinv_triangular(indptr, indices, data)

    # Construct the CSR matrix from the returned components
    L_inv_csr = csr_matrix((x_data, x_indices, x_indptr), shape=L.shape)

    # Verify the result by multiplying the matrix with its inverse
    identity_matrix = L_csr.dot(L_inv_csr)

    # Check if the product is close to the identity matrix
    assert np.allclose(
        identity_matrix.toarray(), np.eye(L.shape[0])
    ), "The product L * L_inv is not the identity matrix"

    print("OK")


if __name__ == "__main__":
    main()
