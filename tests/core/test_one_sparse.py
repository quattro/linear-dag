import h5py
import numpy as np

from linear_dag.core.lineararg import LinearARG
from linear_dag.core.one_sparse import OneSparseMatrix


def test_one_sparse_roundtrip_preserves_csc(linarg_h5_path, first_block_name):
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    compressed = OneSparseMatrix.from_csc(linarg.A)

    restored = compressed.to_csc()

    assert np.array_equal(restored.indptr, linarg.A.indptr)
    assert np.array_equal(restored.indices, linarg.A.indices)
    assert np.array_equal(restored.data, linarg.A.data)
    assert compressed.edge_weight_nbytes < linarg.A.data.nbytes


def test_compressed_lineararg_products_match_default(linarg_h5_path, first_block_name):
    default = LinearARG.read(linarg_h5_path, block=first_block_name)
    compressed = LinearARG.read(linarg_h5_path, block=first_block_name, compress_edge_weights=True)
    rng = np.random.default_rng(42)

    variant_vector = rng.standard_normal(default.shape[1])
    sample_vector = rng.standard_normal(default.shape[0])
    variant_matrix = rng.standard_normal((default.shape[1], 7)).astype(np.float32)
    sample_matrix = rng.standard_normal((default.shape[0], 7)).astype(np.float32)

    assert np.allclose(compressed._matvec(variant_vector), default._matvec(variant_vector))
    assert np.allclose(compressed._rmatvec(sample_vector), default._rmatvec(sample_vector))
    assert np.allclose(compressed._matmat(variant_matrix), default._matmat(variant_matrix))
    assert np.allclose(compressed._rmatmat(sample_matrix), default._rmatmat(sample_matrix))


def test_compressed_edge_weight_hdf5_roundtrip(tmp_path, linarg_h5_path, first_block_name):
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    path = tmp_path / "compressed.h5"

    linarg.write(path, save_allele_counts=False, compress_edge_weights=True)

    with h5py.File(path, "r") as file:
        assert "data" not in file
        assert file.attrs["edge_weight_encoding"] == "one_sparse_v1"
        assert "nonunit_edge_indices" in file
        assert "nonunit_values" in file

    default_reload = LinearARG.read(path)
    compressed_reload = LinearARG.read(path, compress_edge_weights=True)
    assert np.array_equal(default_reload.A.data, linarg.A.data)
    assert np.array_equal(compressed_reload.A.to_csc().data, linarg.A.data)
