import h5py
import numpy as np

from linear_dag import LinearARG, OneSparseMatrix
from linear_dag.association.blup import blup


def test_one_sparse_roundtrip_preserves_csc(linarg_h5_path, first_block_name):
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    assert isinstance(linarg.A, OneSparseMatrix)
    original = linarg.A.to_csc()
    compressed = OneSparseMatrix.from_csc(original)

    restored = compressed.to_csc()

    assert np.array_equal(restored.indptr, original.indptr)
    assert np.array_equal(restored.indices, original.indices)
    assert np.array_equal(restored.data, original.data)
    assert compressed.edge_weight_nbytes < original.data.nbytes


def test_compressed_lineararg_products_match_default(linarg_h5_path, first_block_name):
    compressed = LinearARG.read(linarg_h5_path, block=first_block_name, compress_edge_weights=True)
    default = compressed.copy()
    default.A = compressed.A.to_csc()
    rng = np.random.default_rng(42)

    variant_vector = rng.standard_normal(default.shape[1])
    sample_vector = rng.standard_normal(default.shape[0])
    variant_matrix = rng.standard_normal((default.shape[1], 7)).astype(np.float32)
    sample_matrix = rng.standard_normal((default.shape[0], 7)).astype(np.float32)

    assert np.allclose(compressed._matvec(variant_vector), default._matvec(variant_vector))
    assert np.allclose(compressed._rmatvec(sample_vector), default._rmatvec(sample_vector))
    assert np.allclose(compressed._matmat(variant_matrix), default._matmat(variant_matrix))
    assert np.allclose(compressed._rmatmat(sample_matrix), default._rmatmat(sample_matrix))


def test_default_hdf5_roundtrip_uses_compressed_schema(tmp_path, linarg_h5_path, first_block_name):
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    path = tmp_path / "compressed.h5"

    linarg.write(path, save_allele_counts=False)

    with h5py.File(path, "r") as file:
        assert "data" not in file
        assert file.attrs["edge_weight_encoding"] == "one_sparse_v1"
        assert "nonunit_edge_indices" in file
        assert "nonunit_values" in file

    reloaded = LinearARG.read(path)
    assert isinstance(reloaded.A, OneSparseMatrix)
    assert np.array_equal(reloaded.A.to_csc().data, linarg.A.to_csc().data)


def test_legacy_hdf5_write_remains_readable_as_compressed(tmp_path, linarg_h5_path, first_block_name):
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    path = tmp_path / "legacy.h5"

    linarg.write(path, save_allele_counts=False, compress_edge_weights=False)

    with h5py.File(path, "r") as file:
        assert "data" in file
        assert "edge_weight_encoding" not in file.attrs

    reloaded = LinearARG.read(path)
    assert isinstance(reloaded.A, OneSparseMatrix)
    assert np.array_equal(reloaded.A.to_csc().data, linarg.A.to_csc().data)


def test_scipy_boundary_operations_return_compressed_linearargs(linarg_h5_path, first_block_name):
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)

    removed = linarg.remove_samples([str(linarg.iids[0])])
    with_individuals = linarg.add_individual_nodes()

    assert isinstance(removed.A, OneSparseMatrix)
    assert isinstance(with_individuals.A, OneSparseMatrix)


def test_blup_matches_scipy_adjacency_adapter(linarg_h5_path, first_block_name):
    compressed = LinearARG.read(linarg_h5_path, block=first_block_name)
    scipy_backed = compressed.copy()
    scipy_backed.A = compressed.A.to_csc()
    phenotype = np.random.default_rng(123).standard_normal(compressed.shape[0])

    np.testing.assert_allclose(
        blup(compressed, 0.5, phenotype),
        blup(scipy_backed, 0.5, phenotype),
        rtol=1e-6,
        atol=1e-6,
    )
