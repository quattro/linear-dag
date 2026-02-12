import numpy as np
import pytest

from linear_dag.structure import pca, svd
from scipy.sparse.linalg import aslinearoperator


class _DummyLinearARG:
    def __init__(self, normalized_matrix: np.ndarray):
        self.normalized = aslinearoperator(np.asarray(normalized_matrix, dtype=np.float64))


def _build_dummy_linarg() -> tuple[_DummyLinearARG, np.ndarray]:
    matrix = np.array(
        [
            [2.0, 0.0, 1.0, 3.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 2.0, 1.0, 0.0],
            [3.0, 1.0, 1.0, 2.0],
            [1.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    return _DummyLinearARG(matrix), matrix


def test_svd_returns_sorted_singular_values():
    linarg, matrix = _build_dummy_linarg()
    left_vecs, singular_values, right_vecs = svd(linarg, k=3)

    expected = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)[:3]

    assert left_vecs.shape == (matrix.shape[0], 3)
    assert right_vecs.shape == (3, matrix.shape[1])
    assert np.all(np.diff(singular_values) <= 0)
    assert np.allclose(singular_values, expected, atol=1e-6)


def test_pca_returns_sorted_real_eigenpairs():
    linarg, matrix = _build_dummy_linarg()
    eigenvectors, eigenvalues = pca(linarg, k=3)

    gram = matrix @ matrix.T
    expected = np.linalg.eigvalsh(gram)[::-1][:3]

    assert eigenvectors.shape == (matrix.shape[0], 3)
    assert np.isrealobj(eigenvalues)
    assert np.all(np.diff(eigenvalues) <= 0)
    assert np.allclose(eigenvalues, expected, atol=1e-6)

    for i, eigenvalue in enumerate(eigenvalues):
        residual = gram @ eigenvectors[:, i] - eigenvalue * eigenvectors[:, i]
        assert np.linalg.norm(residual) < 1e-5


@pytest.mark.parametrize(
    ("routine", "k", "error_type", "message"),
    [
        ("svd", 0, ValueError, "svd: k must be >= 1"),
        ("svd", 4, ValueError, "svd: k must be < 4"),
        ("svd", 1.5, TypeError, "svd: k must be an integer"),
        ("pca", 0, ValueError, "pca: k must be >= 1"),
        ("pca", 5, ValueError, "pca: k must be < 5"),
        ("pca", 2.5, TypeError, "pca: k must be an integer"),
    ],
)
def test_structure_rank_validation(routine, k, error_type, message):
    linarg, _ = _build_dummy_linarg()
    fn = svd if routine == "svd" else pca
    with pytest.raises(error_type, match=message):
        fn(linarg, k=k)
