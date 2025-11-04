# one_summed_cy.pyx
import numpy as np
from .data_structures cimport CountingArray, Stack
from .digraph cimport node, edge, DiGraph
from scipy.sparse import csr_matrix
cimport numpy as cnp

cdef int REALLOC_FACTOR = 2

def linearize_brick_graph(G: DiGraph) -> csr_matrix:
    # G = G.copy()
    cdef int[:] edge_weights = np.ones(REALLOC_FACTOR * G.maximum_number_of_edges, dtype=np.intc)
    cdef long[:] order = G.reverse_topological_sort()

    # Weighted in-degree of each node in the subgraph of descendants of the current node
    cdef CountingArray subgraph_indegree = CountingArray(G.maximum_number_of_nodes)

    cdef Stack nodes_to_visit = Stack(G.maximum_number_of_nodes)
    cdef Stack nodes_to_visit_again = Stack(G.maximum_number_of_nodes)
    cdef edge* current_edge
    cdef edge* new_edge
    cdef long starting_node_idx
    cdef long visited_node_idx
    cdef int weighted_in_degree

    for starting_node_idx in order:
        subgraph_indegree.clear()
        nodes_to_visit.push(starting_node_idx)

        # Compute weighted subgraph in-degree: sum of in-edge weights for each descendant of the starting node,
        # if the parent node of that edge is also a descendant
        while nodes_to_visit.length > 0:
            visited_node_idx = nodes_to_visit.pop()
            if visited_node_idx != starting_node_idx:
                nodes_to_visit_again.push(visited_node_idx)
            current_edge = G.nodes[visited_node_idx].first_out
            while current_edge is not NULL:
                if not subgraph_indegree.contains(current_edge.v.index):
                    nodes_to_visit.push(current_edge.v.index)

                subgraph_indegree.increment_element(current_edge.v.index, edge_weights[current_edge.index])
                current_edge = current_edge.next_out

        # Adjust edge weights between starting node and each descendant whose subgraph in-degree is not 1
        while nodes_to_visit_again.length > 0:
            visited_node_idx = nodes_to_visit_again.pop()
            weighted_in_degree = subgraph_indegree.get_element(visited_node_idx)
            if weighted_in_degree == 1:
                continue
            new_edge = G.add_edge(starting_node_idx, visited_node_idx)  # could be duplicate edges between u,v
            while new_edge.index >= len(edge_weights):
                old_len = len(edge_weights)
                new_size = len(edge_weights) * REALLOC_FACTOR
                edge_weights = np.resize(np.asarray(edge_weights), new_size)
                edge_weights[old_len:] = 1
            edge_weights[new_edge.index] = 1 - weighted_in_degree

    cdef int[:] data = np.zeros(G.maximum_number_of_edges, dtype=np.intc)
    cdef long[:] row_ind = np.empty(G.maximum_number_of_edges, dtype=np.int64)
    cdef long[:] col_ind = np.empty(G.maximum_number_of_edges, dtype=np.int64)
    cdef long i
    cdef int counter = 0
    cdef edge * e
    for i in range(G.maximum_number_of_edges):
        e = G.get_edge(i)
        if e is NULL or e.u is NULL:
            continue

        data[counter] = edge_weights[e.index]
        row_ind[counter] = e.v.index
        col_ind[counter] = e.u.index
        counter += 1

    return csr_matrix((data[:counter], (row_ind[:counter], col_ind[:counter])),
                      shape=(G.maximum_number_of_nodes, G.maximum_number_of_nodes))
