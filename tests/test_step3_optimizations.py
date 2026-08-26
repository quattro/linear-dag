import h5py
import numpy as np

from scipy.sparse import csc_matrix, csr_matrix

from linear_dag.core.brick_graph import (
    get_graph_statistics,
    merge_brick_graphs,
    merge_brick_graphs_into,
)
from linear_dag.core.digraph import DiGraph
from linear_dag.core.lineararg import remove_degree_zero_nodes
from linear_dag.core.one_summed_cy import linearize_brick_graph
from linear_dag.core.recombination import Recombination
from linear_dag.pipeline import _read_genotype_partition_stats


def _write_brick_graph(path, edges, sample_indices, variant_indices, num_nodes):
    rows, columns = zip(*edges)
    adjacency = csc_matrix(
        (np.ones(len(edges), dtype=np.int32), (rows, columns)),
        shape=(num_nodes, num_nodes),
    )
    with h5py.File(path, "w") as f:
        f.attrs["n"] = num_nodes
        f.create_dataset("indptr", data=adjacency.indptr)
        f.create_dataset("indices", data=adjacency.indices)
        f.create_dataset("data", data=adjacency.data)
        f.create_dataset("sample_indices", data=np.asarray(sample_indices, dtype=np.int64))
        f.create_dataset("variant_indices", data=np.asarray(variant_indices, dtype=np.int64))


def test_direct_recombination_merge_preserves_graph_order_and_linearization(tmp_path):
    partition_ids = ["part0", "part1"]
    edges = [(parent, child) for parent in (0, 1) for child in (2, 3, 4, 5)]
    for partition_id in partition_ids:
        _write_brick_graph(
            tmp_path / f"{partition_id}.h5",
            edges,
            sample_indices=[4, 5],
            variant_indices=[0, 1],
            num_nodes=6,
        )

    num_samples, num_nodes, num_edges = get_graph_statistics(str(tmp_path), partition_ids)
    merged, expected_variants, _, _ = merge_brick_graphs(str(tmp_path), partition_ids)
    expected = Recombination.from_graph(merged)

    reserved_edges = max(num_edges + 2, (5 * num_edges + 3) // 4)
    actual = Recombination(num_nodes + num_edges // 4 + 1, reserved_edges)
    actual._limit_initial_edge_capacity(num_edges + 2)
    actual_variants, actual_samples, actual_nodes, actual_edges = merge_brick_graphs_into(
        actual,
        str(tmp_path),
        partition_ids,
        num_samples,
        num_nodes,
    )

    assert actual_samples == num_samples
    assert actual_nodes == num_nodes
    assert actual_edges == num_edges
    assert actual.edge_list() == expected.edge_list()
    np.testing.assert_array_equal(actual_variants, expected_variants)

    actual.compute_cliques()
    actual.collect_cliques(num_edges + 2)
    expected.find_recombinations()
    actual.find_recombinations()
    assert actual.edge_list() == expected.edge_list()
    assert actual.number_of_nodes == expected.number_of_nodes

    expected_adjacency = csr_matrix(linearize_brick_graph(expected))
    assert actual.workspace_allocated
    actual.release_workspace()
    assert not actual.workspace_allocated
    actual._activate_initial_edge_reserve()
    actual_adjacency = csr_matrix(linearize_brick_graph(actual))

    assert actual.max_edges == reserved_edges
    np.testing.assert_array_equal(actual_adjacency.indptr, expected_adjacency.indptr)
    np.testing.assert_array_equal(actual_adjacency.indices, expected_adjacency.indices)
    np.testing.assert_array_equal(actual_adjacency.data, expected_adjacency.data)


def test_read_genotype_partition_stats_matches_sparse_computation(tmp_path):
    genotypes = csc_matrix(
        np.array(
            [
                [1, 0, 1, 0, 0],
                [0, 0, 1, 2, 0],
                [1, 0, 1, 0, 1],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.int16,
        )
    )
    filename = tmp_path / "genotypes.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("shape", data=genotypes.shape)
        f.create_dataset("indptr", data=genotypes.indptr)
        f.create_dataset("indices", data=genotypes.indices)
        f.create_dataset("data", data=genotypes.data)

    folded_nnz, allele_counts = _read_genotype_partition_stats(filename)

    column_nnz = np.diff(genotypes.indptr)
    expected_folded_nnz = np.minimum(column_nnz, genotypes.shape[0] - column_nnz).sum()
    expected_allele_counts = np.ones(genotypes.shape[0]) @ genotypes
    assert folded_nnz == expected_folded_nnz
    np.testing.assert_array_equal(allele_counts, expected_allele_counts)


def test_hidden_edge_reserve_must_be_activated_before_use():
    graph = DiGraph(4, 4)
    graph._limit_initial_edge_capacity(2)
    graph.create_edge(0, 1)
    graph.create_edge(1, 2)

    with np.testing.assert_raises_regex(RuntimeError, "reserve must be activated"):
        graph.create_edge(2, 3)

    graph._activate_initial_edge_reserve()
    graph.create_edge(2, 3)
    assert graph.number_of_edges == 3
    assert graph.max_edges == 4


def test_remove_degree_zero_nodes_keeps_required_nodes_in_sorted_order():
    adjacency = csc_matrix(
        (np.ones(1, dtype=np.int32), ([0], [1])),
        shape=(5, 5),
    )

    filtered, variant_indices, sample_indices = remove_degree_zero_nodes(
        adjacency,
        np.array([2]),
        np.array([4]),
    )

    expected = adjacency[np.array([0, 1, 2, 4]), :][:, np.array([0, 1, 2, 4])]
    np.testing.assert_array_equal(filtered.toarray(), expected.toarray())
    np.testing.assert_array_equal(variant_indices, np.array([2]))
    np.testing.assert_array_equal(sample_indices, np.array([3]))
