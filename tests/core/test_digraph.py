import gc
import weakref

import h5py
import numpy as np

from scipy.sparse import csc_matrix

from linear_dag.core.brick_graph import BrickGraph, read_graph_from_disk, reduction_union, reduction_union_packed
from linear_dag.core.digraph import DiGraph
from linear_dag.core.recombination import Recombination
from linear_dag.genotype import read_vcf, write_vcf_to_hdf5
from linear_dag.pipeline import reduction_union_recom, run_forward_backward


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


def test_packed_reduction_union_preserves_edge_order_and_releases_chunks():
    forward = DiGraph(8, 8)
    backward = DiGraph(8, 8)
    for parent, child in [(0, 2), (0, 3), (1, 3), (2, 4), (3, 5), (4, 6)]:
        forward.create_edge(parent, child)
    for parent, child in [(7, 5), (7, 4), (6, 3), (5, 2), (4, 1)]:
        backward.create_edge(parent, child)

    expected_union = reduction_union(forward, backward)
    packed_union = reduction_union_packed(forward, backward, chunk_size=3)
    packed_edge_list = [tuple(map(int, endpoint)) for chunk in packed_union.chunks for endpoint in chunk]

    assert packed_edge_list == expected_union.edge_list()
    assert packed_union.number_of_nodes == expected_union.max_nodes
    assert packed_union.number_of_edges == expected_union.number_of_edges
    assert packed_union.allocated_nbytes == 4 * 3 * 2 * np.dtype(np.int32).itemsize
    assert all(chunk.dtype == np.int32 for chunk in packed_union.chunks)

    chunk_references = [weakref.ref(chunk) for chunk in packed_union.chunks]
    expected_recombination = Recombination.from_graph(expected_union)
    actual_recombination = Recombination.from_packed_edges(packed_union)
    gc.collect()

    assert packed_union.released
    assert packed_union.chunks == ()
    assert all(reference() is None for reference in chunk_references)
    assert actual_recombination.edge_list() == expected_recombination.edge_list()
    np.testing.assert_array_equal(actual_recombination.get_cliques, expected_recombination.get_cliques)
    for actual, expected in zip(
        actual_recombination.get_clique_rows,
        expected_recombination.get_clique_rows,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    with np.testing.assert_raises_regex(RuntimeError, "already been consumed"):
        Recombination.from_packed_edges(packed_union)


def test_packed_reduction_union_handles_empty_and_single_edge_boundaries():
    empty_forward = DiGraph(3, 1)
    empty_backward = DiGraph(3, 1)
    empty_union = reduction_union_packed(empty_forward, empty_backward, chunk_size=1)

    assert empty_union.number_of_nodes == 3
    assert empty_union.number_of_edges == 0
    assert empty_union.allocated_nbytes == 0
    assert empty_union.chunks == ()

    empty_recombination = Recombination.from_packed_edges(empty_union)
    assert empty_recombination.number_of_nodes == 3
    assert empty_recombination.number_of_edges == 0

    single_forward = DiGraph(3, 1)
    single_backward = DiGraph(3, 1)
    single_forward.create_edge(0, 2)
    expected_union = reduction_union(single_forward, single_backward)
    packed_union = reduction_union_packed(single_forward, single_backward, chunk_size=1)

    assert [tuple(map(int, endpoint)) for chunk in packed_union.chunks for endpoint in chunk] == [(0, 2)]
    expected_recombination = Recombination.from_graph(expected_union)
    actual_recombination = Recombination.from_packed_edges(packed_union)
    assert actual_recombination.edge_list() == expected_recombination.edge_list()


def test_disk_brick_graph_keeps_minimal_native_edge_arena_and_stream_order(tmp_path):
    output = tmp_path / "streamed.h5"
    brick_graph = BrickGraph(2, 4, save_to_disk=True, out=str(output))
    streamed_edges = [(0, 2), (1, 3), (2, 4), (3, 5)]

    assert brick_graph._native_graph_stats == (0, 1)
    for parent, child in streamed_edges:
        brick_graph.add_edge(parent, child)
    assert brick_graph._native_graph_stats == (0, 1)
    del brick_graph
    gc.collect()

    with h5py.File(output, "r") as stored:
        assert stored.attrs["n"] == 6
        np.testing.assert_array_equal(stored["rows"][:], np.array([0, 1, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(stored["cols"][:], np.array([2, 3, 4, 5], dtype=np.int32))

    in_memory = BrickGraph(2, 4)
    assert in_memory._native_graph_stats == (0, 6)
    for parent, child in streamed_edges:
        in_memory.add_edge(parent, child)
    assert in_memory._native_graph_stats == (4, 6)


def test_disk_brick_graph_handles_zero_variant_boundary(tmp_path):
    genotypes = csc_matrix((2, 0), dtype=np.int32)
    output_prefix = tmp_path / "empty"

    expected_forward, expected_backward, expected_samples = BrickGraph.forward_backward(genotypes)
    actual_samples = BrickGraph.forward_backward(
        genotypes,
        save_to_disk=True,
        out=str(output_prefix),
    )

    np.testing.assert_array_equal(actual_samples, expected_samples)
    assert expected_forward.number_of_edges == 0
    assert expected_backward.number_of_edges == 0
    for direction in ("forward", "backward"):
        with h5py.File(f"{output_prefix}_{direction}_graph.h5", "r") as stored:
            assert stored.attrs["n"] == 2
            assert stored["rows"].shape == (0,)
            assert stored["cols"].shape == (0,)


def test_collect_cliques_matches_edge_labels_in_ascending_order():
    edges = []
    for parents, children in [((0, 1), range(6, 10)), ((2, 3), range(10, 13)), ((4, 5), range(13, 16))]:
        edges.extend((parent, child) for parent in parents for child in children)
    graph = DiGraph(16, len(edges))
    for parent, child in edges:
        graph.create_edge(parent, child)

    recombination = Recombination.from_graph(graph)
    cliques = recombination.get_cliques
    clique_rows = recombination.get_clique_rows

    for clique_index in range(max(cliques) + 1):
        np.testing.assert_array_equal(clique_rows[clique_index], np.flatnonzero(cliques == clique_index))
    assert sum(map(len, clique_rows)) == np.count_nonzero(cliques >= 0)

    heap, priorities = recombination.get_heap
    assert len(heap) == np.count_nonzero(priorities)


def test_sparse_heap_handles_exhausting_the_only_clique():
    edges = [(parent, child) for parent in (0, 1) for child in range(2, 8)]
    graph = DiGraph(8, len(edges))
    for parent, child in edges:
        graph.create_edge(parent, child)

    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()

    assert recombination.number_of_edges < len(edges)


def test_sparse_heap_handles_graph_without_cliques():
    graph = DiGraph(3, 2)
    graph.create_edge(0, 1)
    graph.create_edge(1, 2)

    recombination = Recombination.from_graph(graph)
    heap, _ = recombination.get_heap

    assert heap == []
    recombination.find_recombinations()
    assert recombination.number_of_edges == 2


def test_recombination_is_deterministic_for_equal_priority_cliques():
    edges = []
    for parents, children in [((0, 1), range(6, 10)), ((2, 3), range(10, 14)), ((4, 5), range(14, 18))]:
        edges.extend((parent, child) for parent in parents for child in children)

    outputs = []
    for _ in range(2):
        graph = DiGraph(18, len(edges))
        for parent, child in edges:
            graph.create_edge(parent, child)
        recombination = Recombination.from_graph(graph)
        recombination.find_recombinations()
        outputs.append(tuple(np.asarray(array).copy() for array in recombination.to_csc_arrays()[:3]))

    for first, second in zip(*outputs):
        np.testing.assert_array_equal(first, second)


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
    expected_forward, expected_backward, in_memory_samples = BrickGraph.forward_backward(genotypes)
    np.testing.assert_array_equal(actual_samples, in_memory_samples)
    for direction in ["forward", "backward"]:
        with h5py.File(f"{in_memory_prefix}_{direction}_graph.h5", "r") as expected:
            with h5py.File(f"{streamed_prefix}_{direction}_graph.h5", "r") as actual:
                assert actual.attrs["n"] == expected.attrs["n"]
                np.testing.assert_array_equal(actual["rows"][:], expected["rows"][:])
                np.testing.assert_array_equal(actual["cols"][:], expected["cols"][:])

        expected_graph = expected_forward if direction == "forward" else expected_backward
        actual_graph = read_graph_from_disk(f"{streamed_prefix}_{direction}_graph.h5")
        np.testing.assert_array_equal(actual_graph.to_csc().toarray(), expected_graph.to_csc().toarray())


def test_streamed_pipeline_singletons_use_carrier_sample_nodes(tmp_path):
    out = tmp_path / "run"
    partition = "0_chr1:1-10"
    genotype_dir = out / "genotype_matrices"
    genotype_dir.mkdir(parents=True)
    genotypes = csc_matrix(
        np.array(
            [
                [1, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.int8,
        )
    )
    with h5py.File(genotype_dir / f"{partition}.h5", "w") as handle:
        handle.create_dataset("shape", data=genotypes.shape)
        handle.create_dataset("indptr", data=genotypes.indptr)
        handle.create_dataset("indices", data=genotypes.indices)
        handle.create_dataset("data", data=genotypes.data)

    run_forward_backward(str(out), "", partition)
    reduction_union_recom(str(out), "", partition)

    with h5py.File(out / "brick_graph_partitions" / f"{partition}.h5", "r") as handle:
        variant_indices = handle["variant_indices"][:]
        sample_indices = handle["sample_indices"][:]
        np.testing.assert_array_equal(variant_indices[:2], np.repeat(sample_indices[0], 2))
        assert variant_indices[2] not in set(sample_indices)
