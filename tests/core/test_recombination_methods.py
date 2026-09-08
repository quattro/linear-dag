"""Validate algorithm selection against genotype and graph semantics."""

import itertools

import numpy as np
import pytest

from scipy.sparse import csc_matrix

from linear_dag.core import linear_arg_inference as lai
from linear_dag.core.brick_graph import BrickGraph
from linear_dag.core.digraph import DiGraph
from linear_dag.core.lineararg import LinearARG, make_triangular, remove_degree_zero_nodes
from linear_dag.core.recombination import Recombination
from linear_dag.core.recombination_methods import _recombination_class
from linear_dag.core.recombination_msbf import Recombination as MSBFRecombination


@pytest.mark.parametrize("method", ["legacy", "msbf"])
def test_exhaustive_three_by_three_genotype_reconstruction(method):
    for bits in itertools.product((0, 1), repeat=9):
        matrix = np.array(bits, dtype=np.int32).reshape(3, 3)
        for flip in (np.zeros(3, dtype=bool), np.array([True, False, True])):
            linarg = LinearARG.from_genotypes(csc_matrix(matrix), flip, recombination_method=method)
            expected = np.where(flip, 1 - matrix, matrix)
            np.testing.assert_array_equal(linarg @ np.eye(3), expected)
            np.testing.assert_array_equal(linarg.T @ np.eye(3), expected.T)


def test_default_retains_legacy_factorization():
    assert _recombination_class("legacy") is Recombination
    assert _recombination_class("msbf") is MSBFRecombination
    matrix = csc_matrix(np.random.default_rng(83).integers(0, 2, size=(20, 30), dtype=np.int32))
    flip = np.zeros(30, dtype=bool)
    default = LinearARG.from_genotypes(matrix.copy(), flip)
    legacy = LinearARG.from_genotypes(matrix.copy(), flip, recombination_method="legacy")
    np.testing.assert_array_equal(default.A.to_csc().toarray(), legacy.A.to_csc().toarray())
    np.testing.assert_array_equal(default.variant_indices, legacy.variant_indices)


@pytest.mark.parametrize("method", ["legacy", "msbf"])
def test_disabled_recombination_does_not_construct_workspace(monkeypatch, method):
    class ForbiddenRecombination:
        @staticmethod
        def from_graph(graph):
            raise AssertionError("Recombination is disabled")

    monkeypatch.setattr(lai, "_recombination_class", lambda name: ForbiddenRecombination)
    matrix = csc_matrix([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int32)
    linarg = LinearARG.from_genotypes(
        matrix.copy(), np.zeros(3, dtype=bool), find_recombinations=False, recombination_method=method
    )
    np.testing.assert_array_equal(linarg @ np.eye(3), matrix.toarray())


@pytest.mark.parametrize("method", [None, "MSBF", "unknown", ""])
def test_invalid_method_fails_before_graph_construction(monkeypatch, method):
    class ForbiddenBrickGraph:
        @staticmethod
        def from_genotypes(genotypes):
            raise AssertionError("Validation should happen first")

    monkeypatch.setattr(lai, "BrickGraph", ForbiddenBrickGraph)
    with pytest.raises(ValueError, match="Unknown recombination method"):
        lai.linear_arg_from_genotypes(
            csc_matrix((3, 3)), np.zeros(3, dtype=bool), None, False, recombination_method=method
        )


@pytest.mark.parametrize("method", ["legacy", "msbf"])
def test_real_vcf_reconstruction_and_hdf5_round_trip(method, test_data_dir, tmp_path):
    linarg, genotypes = LinearARG.from_vcf(
        test_data_dir / "1kg_small.vcf", return_genotypes=True, recombination_method=method
    )
    columns = np.arange(genotypes.shape[1])
    np.testing.assert_array_equal(linarg.get_carriers_subset(columns).toarray(), genotypes.toarray())
    path = tmp_path / "recombined.h5"
    linarg.write(path)
    restored = LinearARG.read(path)
    np.testing.assert_array_equal(restored.get_carriers_subset(columns).toarray(), genotypes.toarray())


@pytest.mark.parametrize("method", ["legacy", "msbf"])
@pytest.mark.parametrize("edges", [[], [(0, 1)], [(0, 2), (1, 2)]])
def test_empty_and_no_shared_pairs(method, edges):
    graph = DiGraph(4, max(1, len(edges)))
    for node in range(4):
        graph.create_node(node)
    for parent, child in edges:
        graph.create_edge(parent, child)
    recom = _recombination_class(method).from_graph(graph)
    recom.find_recombinations()
    assert sorted(recom.edge_list()) == sorted(edges)
    assert recom.number_of_nodes == 4


@pytest.mark.parametrize("first,second", list(itertools.product(["legacy", "msbf"], repeat=2)))
def test_second_recombination_pass_preserves_genotypes(first, second):
    matrix = csc_matrix(np.random.default_rng(32).integers(0, 2, size=(30, 40), dtype=np.int32))
    graph, samples, variants = BrickGraph.from_genotypes(matrix)
    recom = _recombination_class(first).from_graph(graph)
    recom.find_recombinations()
    recom = _recombination_class(second).from_graph(recom)
    recom.find_recombinations()
    adjacency = csc_matrix(lai.linearize_brick_graph(recom))
    adjacency, variants, samples = remove_degree_zero_nodes(adjacency, variants, samples)
    adjacency, variants = make_triangular(adjacency, variants, samples)
    linarg = LinearARG(adjacency, variants, np.zeros(matrix.shape[1], dtype=bool), len(samples))
    np.testing.assert_array_equal(linarg @ np.eye(matrix.shape[1]), matrix.toarray())


@pytest.mark.parametrize("copies", [2, 3])
def test_msbf_rejects_parallel_edges(copies):
    graph = DiGraph(4, 4 * copies)
    for child in (2, 3):
        for parent in [0, 1] * copies:
            graph.create_edge(parent, child)
    with pytest.raises(ValueError, match="unique parents per child"):
        MSBFRecombination.from_graph(graph)


def test_legacy_clique_ids_can_outgrow_initial_edge_count():
    matrix = csc_matrix(np.random.default_rng(83).integers(0, 2, size=(20, 30), dtype=np.int32))
    graph, _, _ = BrickGraph.from_genotypes(matrix)
    recom = Recombination.from_graph(graph)
    initial_capacity = len(recom.get_clique_rows)
    recom.find_recombinations()
    assert len(recom.get_clique_rows) > initial_capacity
    linarg = LinearARG.from_genotypes(matrix, np.zeros(30, dtype=bool), recombination_method="legacy")
    np.testing.assert_array_equal(linarg @ np.eye(30), matrix.toarray())


def test_from_vcf_rejects_invalid_method_before_reading():
    with pytest.raises(ValueError, match="Unknown recombination method"):
        LinearARG.from_vcf("missing.vcf", recombination_method="invalid")
