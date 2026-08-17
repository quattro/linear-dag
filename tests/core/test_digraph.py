import h5py
import numpy as np

from linear_dag.core.brick_graph import BrickGraph
from linear_dag.core.digraph import DiGraph
from linear_dag.core.recombination import Recombination
from linear_dag.genotype import read_vcf, write_vcf_to_hdf5


def test_to_csc_arrays_sorts_parents_and_sums_duplicate_edges():
    graph = DiGraph(4, 5)
    for parent, child in [(2, 1), (0, 1), (2, 1), (1, 3)]:
        graph.create_edge(parent, child)

    indptr, indices, data, n = graph.to_csc_arrays()

    assert n == 4
    np.testing.assert_array_equal(indptr, np.array([0, 0, 2, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(indices, np.array([0, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(data, np.array([1.0, 2.0, 1.0]))
    np.testing.assert_array_equal(
        graph.to_csc().toarray(),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_recombination_reserves_exactly_two_replacement_edges():
    edges = []
    for parents, children in [((0, 1), range(4, 10)), ((2, 3), range(10, 16))]:
        edges.extend((parent, child) for parent in parents for child in children)
    graph = DiGraph(16, len(edges))
    for parent, child in edges:
        graph.create_edge(parent, child)

    recombination = Recombination.from_graph(graph)

    assert recombination.number_of_edges == len(edges)
    assert recombination.max_edges == len(edges) + 2
    assert max(map(len, recombination.get_clique_rows)) > 1

    recombination.find_recombinations()

    assert recombination.number_of_edges < len(edges)
    assert recombination.max_edges == len(edges) + 2


def test_batched_hdf5_forward_backward_matches_in_memory(test_data_dir, tmp_path):
    vcf_path = test_data_dir / "1kg_small.vcf"
    genotype_path = tmp_path / "genotypes.h5"
    in_memory_prefix = tmp_path / "in_memory"
    streamed_prefix = tmp_path / "streamed"
    genotypes, _, _, _ = read_vcf(vcf_path)
    write_vcf_to_hdf5(vcf_path, genotype_path, batch_nnz=19, batch_columns=7)

    expected_samples = BrickGraph.forward_backward(
        genotypes,
        add_samples=True,
        save_to_disk=True,
        out=str(in_memory_prefix),
    )
    actual_samples = BrickGraph.forward_backward_from_hdf5(
        str(genotype_path),
        add_samples=True,
        out=str(streamed_prefix),
        batch_nnz=19,
    )

    np.testing.assert_array_equal(actual_samples, expected_samples)
    for direction in ["forward", "backward"]:
        with h5py.File(f"{in_memory_prefix}_{direction}_graph.h5", "r") as expected:
            with h5py.File(f"{streamed_prefix}_{direction}_graph.h5", "r") as actual:
                assert actual.attrs["n"] == expected.attrs["n"]
                np.testing.assert_array_equal(actual["rows"][:], expected["rows"][:])
                np.testing.assert_array_equal(actual["cols"][:], expected["cols"][:])
