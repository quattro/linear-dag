from .data_structures cimport DiGraph, IntegerSet
from .data_structures cimport node, edge

cpdef DiGraph reduction_union(DiGraph forward_reduction, DiGraph backward_reduction):
    """
    Computes the transitive reduction of the union of the partial orderings defined by two DAGs, which are the
    transitive reductions of the intersections of some partial ordering with a total ordering and its negation.
    :param forward_reduction: the transitive reduction of intersect(partial ordering, total ordering)
    :param backward_reduction: the transitive reduction of intersect(partial ordering, total ordering reversed)
    :return: the transitive reduction of the partial ordering
    """
    cdef int num_nodes = forward_reduction.maximum_number_of_nodes
    cdef IntegerSet reachable_in_two_hops = IntegerSet(num_nodes)
    cdef int node_index
    cdef node* current_node
    cdef edge* out_edge
    cdef DiGraph result = DiGraph(num_nodes, forward_reduction.number_of_edges + backward_reduction.number_of_edges)

    for node_index in range(num_nodes):
        # Set of nodes that is reachable in two hops from this one
        reachable_in_two_hops.clear()
        if forward_reduction.is_node[node_index]:
            search_two_hops(reachable_in_two_hops, forward_reduction, backward_reduction, node_index)
        if backward_reduction.is_node[node_index]:
            search_two_hops(reachable_in_two_hops, backward_reduction, forward_reduction, node_index)

        # Add neighbors that aren't reachable in two hops
        if forward_reduction.is_node[node_index]:
            current_node = forward_reduction.nodes[node_index]
            add_nonredundant_neighbors(result, current_node, reachable_in_two_hops)
        if backward_reduction.is_node[node_index]:
            current_node = backward_reduction.nodes[node_index]
            add_nonredundant_neighbors(result, current_node, reachable_in_two_hops)

    return result


cdef void search_two_hops(IntegerSet result, DiGraph first_graph, DiGraph second_graph, int starting_node_index):
    cdef node* starting_node = first_graph.nodes[starting_node_index]
    cdef edge* first_hop
    cdef edge* second_hop
    first_hop = starting_node.first_out
    while first_hop is not NULL:
        second_hop = second_graph.nodes[first_hop.v.index].first_out
        while second_hop is not NULL:
            result.add(second_hop.v.index)
            second_hop = second_hop.next_out
        first_hop = first_hop.next_out


cdef void add_nonredundant_neighbors(DiGraph result, node* starting_node, IntegerSet neighbors_to_exclude):
    cdef edge* out_edge = starting_node.first_out
    while out_edge is not NULL:
        if not neighbors_to_exclude.contains(out_edge.v.index):
            result.add_edge(starting_node.index, out_edge.v.index)
        out_edge = out_edge.next_out

