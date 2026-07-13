from dataclasses import dataclass

import numpy as np

from scipy.sparse import csc_matrix


@dataclass(frozen=True)
class OneSparseMatrix:
    """CSC topology whose edge weights are implicitly one except for sparse exceptions.

    Non-unit edges retain an index and a value. Edge indices refer to positions
    in the CSC `indices` array and must be sorted.

    !!! Example

        ```python
        compressed = OneSparseMatrix.from_csc(adjacency)
        assert np.array_equal(compressed.to_csc().toarray(), adjacency.toarray())
        ```
    """

    indptr: np.ndarray
    indices: np.ndarray
    nonunit_edge_indices: np.ndarray
    nonunit_values: np.ndarray
    shape: tuple[int, int]

    __array_priority__ = 10.1

    def __post_init__(self) -> None:
        arrays = (
            self.indptr,
            self.indices,
            self.nonunit_edge_indices,
            self.nonunit_values,
        )
        if any(array.dtype != np.int32 for array in arrays):
            raise TypeError("OneSparseMatrix arrays must all have dtype int32")
        if self.indptr.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("CSC indptr and indices must be one-dimensional")
        if self.indptr.size != self.shape[1] + 1:
            raise ValueError("indptr length must equal the number of columns plus one")
        if self.indptr[-1] != self.indices.size:
            raise ValueError("the final indptr entry must equal the number of edges")
        if self.nonunit_edge_indices.size != self.nonunit_values.size:
            raise ValueError("non-unit edge indices and values must have equal length")
        if self.nonunit_edge_indices.size:
            if self.nonunit_edge_indices[0] < 0 or self.nonunit_edge_indices[-1] >= self.indices.size:
                raise ValueError("non-unit edge indices must index the CSC edge array")
            if np.any(self.nonunit_edge_indices[1:] <= self.nonunit_edge_indices[:-1]):
                raise ValueError("non-unit edge indices must be strictly increasing")

    @classmethod
    def from_csc(cls, matrix: csc_matrix) -> "OneSparseMatrix":
        """Compress a CSC matrix whose most common edge weight is one.

        **Arguments:**

        - `matrix`: Sparse matrix with integer-valued edge weights.

        **Returns:**

        - A matrix retaining CSC topology and only non-unit edge weights.
        """
        matrix = matrix.tocsc(copy=False)
        data = np.asarray(matrix.data)
        nonunit = np.flatnonzero(data != 1).astype(np.int32)
        return cls(
            indptr=np.asarray(matrix.indptr, dtype=np.int32).copy(),
            indices=np.asarray(matrix.indices, dtype=np.int32).copy(),
            nonunit_edge_indices=nonunit,
            nonunit_values=np.asarray(data[nonunit], dtype=np.int32),
            shape=matrix.shape,
        )

    @property
    def nnz(self) -> int:
        """Return the number of stored edges.

        **Returns:**

        - Number of entries in the CSC topology.
        """
        return self.indices.size

    @property
    def dtype(self) -> np.dtype:
        """Return the logical edge-weight dtype.

        **Returns:**

        - NumPy `int32` dtype.
        """
        return np.dtype(np.int32)

    @property
    def data(self) -> np.ndarray:
        """Materialize the logical CSC edge-weight array.

        !!! info

            The returned array is independent. Mutating it does not change the
            compressed matrix; use [`linear_dag.core.one_sparse.OneSparseMatrix.to_csc`][]
            when a mutable SciPy sparse matrix is required.

        **Returns:**

        - Dense `int32` edge weights aligned to `indices`.
        """
        data = np.ones(self.nnz, dtype=np.int32)
        data[self.nonunit_edge_indices] = self.nonunit_values
        return data

    @property
    def nbytes(self) -> int:
        """Return bytes used by topology and compressed edge weights.

        **Returns:**

        - Sum of the four backing arrays' allocated bytes.
        """
        return sum(
            array.nbytes
            for array in (
                self.indptr,
                self.indices,
                self.nonunit_edge_indices,
                self.nonunit_values,
            )
        )

    @property
    def edge_weight_nbytes(self) -> int:
        """Return bytes used only by non-unit edge metadata.

        **Returns:**

        - Bytes occupied by non-unit edge indices and values.
        """
        return self.nonunit_edge_indices.nbytes + self.nonunit_values.nbytes

    def to_csc(self) -> csc_matrix:
        """Materialize the logical matrix as an ordinary SciPy CSC matrix.

        **Returns:**

        - Equivalent CSC matrix with a dense `int32` values array.
        """
        return csc_matrix((self.data, self.indices.copy(), self.indptr.copy()), shape=self.shape)

    def tocsc(self, copy: bool = True) -> csc_matrix:
        """Materialize the logical matrix as a SciPy CSC matrix.

        This compatibility adapter mirrors the common SciPy sparse-matrix
        method name. Materialization necessarily allocates the full edge-weight
        array; callers should therefore keep the result local to operations
        that require SciPy's sparse-matrix API.

        **Arguments:**

        - `copy`: Accepted for compatibility. The returned matrix is always
          independent because its dense edge-weight array must be constructed.

        **Returns:**

        - Equivalent SciPy `csc_matrix`.
        """
        return self.to_csc()

    def tocsr(self, copy: bool = True):
        """Materialize the logical matrix as a SciPy CSR matrix.

        **Arguments:**

        - `copy`: Accepted for compatibility.

        **Returns:**

        - Equivalent SciPy `csr_matrix`.
        """
        return self.to_csc().tocsr(copy=copy)

    def tocoo(self, copy: bool = True):
        """Materialize the logical matrix as a SciPy COO matrix.

        **Arguments:**

        - `copy`: Accepted for compatibility.

        **Returns:**

        - Equivalent SciPy `coo_matrix`.
        """
        return self.to_csc().tocoo(copy=copy)

    def copy(self) -> "OneSparseMatrix":
        """Return a deep copy of this matrix.

        **Returns:**

        - Independent [`linear_dag.core.one_sparse.OneSparseMatrix`][].
        """
        return OneSparseMatrix(
            self.indptr.copy(),
            self.indices.copy(),
            self.nonunit_edge_indices.copy(),
            self.nonunit_values.copy(),
            self.shape,
        )

    def __getitem__(self, key):
        return self.to_csc()[key]

    @property
    def T(self):
        return self.to_csc().T

    def __rsub__(self, other):
        return other - self.to_csc()

    def __matmul__(self, other):
        return self.to_csc() @ other

    def __rmatmul__(self, other):
        return other @ self.to_csc()
