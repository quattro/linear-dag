# pattern: Functional Core

from collections import deque

import numpy as np
import pytest

from scipy.sparse import csr_matrix

from linear_dag.core.digraph import DiGraph
from linear_dag.core.linear_arg_inference import path_sum
from linear_dag.core.one_summed_cy import linearize_brick_graph
from linear_dag.core.recombination import Recombination


def _graph_from_parent_words(parent_words):
    """Build a graph while preserving each supplied ordered parent word."""
    node_count = 1 + max(
        max(child for child, _ in parent_words),
        max(parent for _, parents in parent_words for parent in parents),
    )
    edge_count = sum(len(parents) for _, parents in parent_words)
    graph = DiGraph(node_count, edge_count)
    for child, parents in parent_words:
        # DiGraph inserts at the head of an in-edge list.
        for parent in reversed(parents):
            graph.create_edge(parent, child)
    return graph, node_count


def _factor(parent_words):
    graph, original_node_count = _graph_from_parent_words(parent_words)
    original_edges = graph.edge_list()
    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()
    factor_nodes = [
        node for node in range(original_node_count, recombination.max_nodes) if recombination.has_node(node)
    ]
    return graph, recombination, original_edges, factor_nodes


def _reachability(edges, node_count, original_node_count):
    children = [[] for _ in range(node_count)]
    for parent, child in edges:
        children[parent].append(child)

    reachable = np.zeros((original_node_count, original_node_count), dtype=bool)
    for source in range(original_node_count):
        queue = deque([source])
        seen = {source}
        while queue:
            parent = queue.popleft()
            for child in children[parent]:
                if child in seen:
                    continue
                seen.add(child)
                queue.append(child)
                if child < original_node_count:
                    reachable[source, child] = True
    return reachable


def _expanded_parent_word(recombination, child, original_node_count):
    expanded = []
    for parent in recombination.parents(child):
        if parent < original_node_count:
            expanded.append(parent)
        else:
            expanded.extend(_expanded_parent_word(recombination, parent, original_node_count))
    return tuple(expanded)


def _assert_semantics_and_structure(
    graph,
    recombination,
    original_edges,
    original_node_count,
    parent_words,
):
    output_edges = recombination.edge_list()
    assert len(output_edges) <= len(original_edges)
    assert len(output_edges) == len(set(output_edges))
    assert len(np.asarray(recombination.reverse_topological_sort())) == recombination.number_of_nodes

    expected = _reachability(original_edges, original_node_count, original_node_count)
    actual = _reachability(
        output_edges,
        max(recombination.max_nodes, original_node_count),
        original_node_count,
    )
    np.testing.assert_array_equal(actual, expected)
    assert recombination._validate_boundary_index()

    for child in range(original_node_count):
        if not recombination.has_node(child):
            continue
        parents = list(recombination.parents(child))
        assert len(parents) == len(set(parents))
    for child, expected_word in parent_words:
        assert _expanded_parent_word(recombination, child, original_node_count) == expected_word


def test_factors_one_maximal_k_parent_block():
    parent_words = [(4, (0, 1, 2)), (5, (0, 1, 2)), (6, (0, 1, 2))]
    graph, recombination, original_edges, factor_nodes = _factor(parent_words)

    assert len(factor_nodes) == 1
    factor = factor_nodes[0]
    assert tuple(recombination.parents(factor)) == (0, 1, 2)
    for child, _ in parent_words:
        assert tuple(recombination.parents(child)) == (factor,)
    _assert_semantics_and_structure(graph, recombination, original_edges, 7, parent_words)


def test_factors_f2_k2_without_increasing_edges():
    parent_words = [(2, (0, 1)), (3, (0, 1))]
    graph, recombination, original_edges, factor_nodes = _factor(parent_words)

    assert len(factor_nodes) == 1
    assert recombination.number_of_edges == len(original_edges)
    _assert_semantics_and_structure(graph, recombination, original_edges, 4, parent_words)


def test_boundary_index_workspace_tracks_live_domain_not_edge_capacity():
    parent_words = [(8 + child, tuple(range(8))) for child in range(32)]
    graph, original_node_count = _graph_from_parent_words(parent_words)

    recombination = Recombination.from_graph(graph)
    stats = recombination.workspace_statistics

    assert stats["node_scan_capacity"] == original_node_count
    assert stats["bucket_capacity"] == original_node_count
    assert stats["class_count"] == 7
    assert stats["class_capacity"] < stats["boundary_capacity"]
    assert stats["boundary_record_bytes"] <= 24
    assert stats["class_record_bytes"] <= 16
    assert stats["class_map_entry_bytes"] <= 4


def test_factor_node_reserve_is_not_eagerly_materialized():
    parent_words = [(8 + child, tuple(range(8))) for child in range(32)]
    graph, original_node_count = _graph_from_parent_words(parent_words)

    recombination = Recombination.from_graph(graph)
    stats = recombination.workspace_statistics

    assert stats["reserved_node_capacity"] > original_node_count
    assert stats["materialized_node_capacity"] == original_node_count

    recombination.find_recombinations()
    stats = recombination.workspace_statistics
    assert stats["materialized_node_capacity"] == original_node_count + 1


def test_internal_bucket_updates_are_batched_by_class():
    parent_words = [(8 + child, tuple(range(8))) for child in range(32)]
    graph, _ = _graph_from_parent_words(parent_words)

    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()

    # Seven initial classes are inserted once and removed once. Per-occurrence
    # rebucketing would perform hundreds of additional list mutations here.
    assert recombination.workspace_statistics["bucket_mutation_count"] == 14


def test_external_bucket_updates_are_batched_by_class():
    parent_words = [(100 + child, (2 + child % 2, 0, 1, 4 + child % 2)) for child in range(32)]
    graph, _ = _graph_from_parent_words(parent_words)

    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()

    # The shared center is factored first, followed by its two exposed groups.
    # Each affected old/new class should leave or enter a bucket only once.
    stats = recombination.workspace_statistics
    assert stats["bucket_mutation_count"] == 18
    assert stats["batched_external_factor_count"] > 0


def test_unique_external_classes_skip_bucket_replay_without_changing_order():
    q = 4
    parent_words = []
    child = 2 * q + 1
    for left_parent in range(q):
        for right_parent in range(q + 1, 2 * q + 1):
            parent_words.append((child, (left_parent, q, right_parent)))
            child += 1
    for left_parent in range(q):
        for _ in range(q + left_parent):
            parent_words.append((child, (left_parent, q)))
            child += 1
    graph, original_node_count = _graph_from_parent_words(parent_words)

    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()

    factor_words = [tuple(recombination.parents(node)) for node in range(original_node_count, recombination.max_nodes)]
    assert factor_words == [(3, 4), (2, 4), (1, 4), (0, 4)]
    stats = recombination.workspace_statistics
    assert stats["direct_external_factor_count"] == q
    assert stats["batched_external_factor_count"] == 0


@pytest.mark.parametrize(
    "block",
    [
        (0, 1, 2),  # tied selection extends left
        (2, 1, 0),  # tied selection extends right
        (0, 3, 1, 2),  # tied selection extends on both sides
    ],
)
def test_extends_to_the_complete_maximal_block(block):
    children = range(max(block) + 2, max(block) + 5)
    parent_words = [(child, block) for child in children]
    graph, recombination, original_edges, factor_nodes = _factor(parent_words)

    assert len(factor_nodes) == 1
    assert tuple(recombination.parents(factor_nodes[0])) == block
    _assert_semantics_and_structure(
        graph,
        recombination,
        original_edges,
        max(child for child, _ in parent_words) + 1,
        parent_words,
    )


def test_handles_nested_overlapping_blocks_and_factor_parents():
    parent_words = [
        (8, (0, 1, 2)),
        (9, (0, 1, 2)),
        (10, (0, 1, 3)),
        (11, (0, 1, 3)),
        (12, (4, 1, 2)),
        (13, (4, 1, 2)),
    ]
    graph, recombination, original_edges, factor_nodes = _factor(parent_words)

    assert any(parent in factor_nodes for factor in factor_nodes for parent in recombination.parents(factor))
    _assert_semantics_and_structure(graph, recombination, original_edges, 14, parent_words)


def test_handles_tied_maximum_classes_and_parent_list_boundaries():
    parent_words = [
        (8, (0, 1, 4)),
        (9, (0, 1, 5)),
        (10, (6, 2, 3)),
        (11, (7, 2, 3)),
    ]
    graph, recombination, original_edges, factor_nodes = _factor(parent_words)

    assert len(factor_nodes) == 2
    _assert_semantics_and_structure(graph, recombination, original_edges, 12, parent_words)


def test_deterministic_random_small_dags_preserve_reachability_and_boundaries():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        node_count = int(rng.integers(4, 13))
        parent_words = []
        for child in range(1, node_count):
            candidates = np.arange(child)
            parents = candidates[rng.random(child) < 0.45]
            if len(parents):
                rng.shuffle(parents)
                parent_words.append((child, tuple(map(int, parents))))
        if not parent_words:
            continue

        graph, recombination, original_edges, _ = _factor(parent_words)
        _assert_semantics_and_structure(
            graph,
            recombination,
            original_edges,
            node_count,
            parent_words,
        )


def test_linearized_genotype_products_are_exactly_preserved():
    parent_words = [
        (7, (0, 1, 2, 5)),
        (8, (0, 1, 2, 5)),
        (9, (0, 1, 2, 6)),
        (10, (0, 1, 2, 6)),
    ]
    graph, original_node_count = _graph_from_parent_words(parent_words)
    before = linearize_brick_graph(graph.copy())

    recombination = Recombination.from_graph(graph)
    recombination.find_recombinations()
    after = linearize_brick_graph(recombination)

    before_paths = path_sum(csr_matrix(before))
    after_paths = path_sum(csr_matrix(after))
    rng = np.random.default_rng(20260826)
    mutations = rng.integers(-3, 4, size=(original_node_count, 5), dtype=np.int32)
    samples = np.array([7, 8, 9, 10])

    expected = before_paths[samples, :original_node_count] @ mutations
    actual = after_paths[samples, :original_node_count] @ mutations
    np.testing.assert_array_equal(actual, expected)
