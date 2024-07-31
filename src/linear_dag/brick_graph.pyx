# brick_graph.pyx
import numpy as np
from .data_structures cimport node, edge, list_node
from .data_structures cimport DiGraph, LinkedListArray, CountingArray, Stack, IntegerList, IntegerSet
cimport numpy as cnp
from scipy.sparse import csr_matrix, csc_matrix
cdef int MAXINT = 2147483647

cdef class BrickGraph:
    """
    Implements the brick graph algorithm.
    """
    cdef DiGraph graph
    cdef DiGraph tree
    cdef int[:] clade_size
    cdef int root
    cdef LinkedListArray subsequence
    cdef CountingArray times_visited
    cdef CountingArray times_revisited
    cdef int num_samples
    cdef int num_variants
    cdef int direction

    def __cinit__(self, int num_samples, int num_variants):
        cnp.import_array()
        self.num_samples = num_samples
        self.num_variants = num_variants
        self.graph = DiGraph(num_variants + num_samples, num_variants + num_samples)
        self.graph.initialize_all_nodes()
        self.initialize_tree()
        tree_num_nodes = self.tree.maximum_number_of_nodes
        self.times_visited = CountingArray(tree_num_nodes)
        self.times_revisited = CountingArray(tree_num_nodes)
        self.direction = 0

    cpdef void initialize_tree(self):
        self.tree = DiGraph(self.num_samples * 2, self.num_samples * 2 - 1)
        self.root = self.num_samples
        self.clade_size = np.zeros(self.num_samples * 2, dtype=np.intc)
        cdef list edges = [(self.root, i) for i in range(self.num_samples)]
        self.tree.add_edges_from(edges)
        for i in range(self.num_samples):
            self.clade_size[i] = 1
        self.clade_size[self.root] = self.num_samples
        self.subsequence = LinkedListArray(self.tree.maximum_number_of_nodes)

    @staticmethod
    def from_genotypes(genotypes: csc_matrix) -> DiGraph:
        num_samples, num_variants = genotypes.shape

        # Forward pass
        forward_graph = BrickGraph(num_samples, num_variants)
        forward_graph.direction = 1
        cdef int i
        for i in range(num_variants):
            carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i + 1]]
            forward_graph.intersect_clades(carriers, i)

        # Add samples
        for i in range(num_samples):
            forward_graph.add_edges_from_subsequence(i, i + num_variants)

        # Backward pass
        backward_graph = BrickGraph(num_samples, num_variants)
        backward_graph.direction = -1
        for i in reversed(range(num_variants)):
            carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i+1]]
            backward_graph.intersect_clades(carriers, i)

        # Transitive reduction of their union
        prune_back_edges(forward_graph.graph, backward_graph.graph)
        return reduction_union(forward_graph.graph, backward_graph.graph)


    def to_csr(self) -> csr_matrix:
        edge_list = self.graph.edge_list()
        if len(edge_list) > 0:
            parents, children = zip(*edge_list)
        else:
            parents, children = [], []
        result = csr_matrix((np.ones(len(edge_list)), (children, parents)),
                          shape=(self.num_variants, self.num_variants))
        result.setdiag(1)
        return result

    def edge_list(self) -> list:
        return self.graph.edge_list()

    cpdef void intersect_clades(self, int[:] new_clade, int clade_index):
        """
        Adds a new clade to a rooted tree and splits existing clades if they intersect with the new clade. Returns the
        lowest common ancestor from the previous tree of nodes in the new clade.
        :param new_clade: integer-valued numpy array
        :param clade_index: integer identifier for the new clade
        :return: lowest common ancestor of nodes in new_clade
        """
        cdef int new_clade_size = len(new_clade)
        if new_clade_size == 0:
            return

        # Compute bottom-up ordering of nodes; update self.visits to tell number of leaves descended from each node
        cdef node * lowest_common_ancestor = self.partial_traversal(new_clade)
        self.times_revisited.clear()
        assert lowest_common_ancestor is not NULL

        # Create edges to the new clade from the variants whose intersection forms the LCA
        self.add_edges_from_subsequence(lowest_common_ancestor.index, clade_index)

        cdef IntegerList traversal = IntegerList(2 * len(new_clade))
        cdef int i
        for i in new_clade:
            traversal.push(i)
            self.times_revisited.set_element(i, 1)

        cdef edge* out_edge
        cdef edge* visited_edge
        cdef edge* unvisited_edge
        cdef node* sibling_node
        cdef node* new_root
        cdef node* parent_of_v
        cdef node* v = NULL
        cdef int node_idx
        cdef bint v_is_root
        cdef Stack visited_children, unvisited_children
        cdef int num_children_visited, num_children_unvisited, visits

        while traversal.length > 0:
            node_idx = traversal.pop()
            v = self.tree.nodes[node_idx]

            # Push a node when all its visited children have been found
            if v.first_in != NULL:
                i = v.first_in.u.index
                visits = self.times_revisited.increment_element(i, self.times_revisited.get_element(node_idx))
                if visits == self.times_visited.get_element(i):
                    traversal.push(i)

            # No unvisited children: means intersect(v, new_clade) == v
            if self.times_visited[node_idx] == self.clade_size[node_idx]:
                # self.subsequence.extend(node_idx, clade_index)
                continue

            visited_children, unvisited_children = self.get_visited_children(v)  # TODO avoid unvisited children?
            num_children_unvisited, num_children_visited = unvisited_children.length, visited_children.length
            assert num_children_unvisited > 0

            # If v is the LCA, then its clade is not partitioned, but rather a subclade is produced
            if node_idx == lowest_common_ancestor.index:
                assert num_children_visited > 1
                child_node = self.tree.add_node(-1)
                assert child_node.index < 2 * self.num_samples
                self.clade_size[child_node.index] = new_clade_size
                # self.subsequence.copy_list(node_idx, child_node.index)
                # self.subsequence.extend(child_node.index, clade_index)

                self.tree.add_edge(node_idx, child_node.index)
                while visited_children.length > 0:
                    child = visited_children.pop()
                    visited_edge = self.tree.nodes[child].first_in
                    self.tree.set_edge_parent(visited_edge, child_node)
                lowest_common_ancestor = child_node  # LCA in new tree
                break

            assert v.first_in is not NULL
            parent_of_v = v.first_in.u

            # Exactly one visited and one unvisited child: delete v, as there are existing nodes
            # for both intersect(v, new_clade) and intersect(v, new_clade_complement)
            if num_children_visited == 1 and num_children_unvisited == 1:
                self.subsequence.copy_list(node_idx, visited_children.pop())
                self.subsequence.copy_list(node_idx, unvisited_children.pop())
                self.subsequence.clear_list(node_idx)
                self.clade_size[node_idx] = 0
                self.tree.collapse_node_with_indegree_one(v)
                continue

            # Exactly one child w is visited: there is an existing node for intersect(v, new_clade);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade_complement)
            if num_children_visited == 1:
                i = visited_children.pop()
                visited_edge = self.tree.nodes[i].first_in
                self.tree.set_edge_parent(visited_edge, parent_of_v)
                self.times_visited.set_element(node_idx, 0)
                self.subsequence.copy_list(node_idx, i)
                self.clade_size[node_idx] -= self.clade_size[i]
                continue

            # Exactly one child is w is unvisited: there is an existing node for intersect(v, new_clade_complement);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade)
            if num_children_unvisited == 1:
                i = unvisited_children.pop()
                unvisited_edge = self.tree.nodes[i].first_in
                self.tree.set_edge_parent(unvisited_edge, parent_of_v)
                # self.subsequence.extend(node_idx, clade_index)  # v is now a subset of the new clade
                self.subsequence.copy_list(node_idx, i)
                self.clade_size[node_idx] -= self.clade_size[i]
                continue

            # Multiple visited and unvisited children: create new_node for intersect(v, new_clade_complement)
            # and replace v with intersect(v, new_clade)
            sibling_node = self.tree.add_node(-1)
            assert sibling_node.index < self.num_samples * 2
            self.times_visited.set_element(sibling_node.index, 0)
            self.subsequence.copy_list(node_idx, sibling_node.index)
            # self.subsequence.extend(node_idx, clade_index)
            # self.clade_variants.copy_list(node_idx, sibling_node.index)
            self.tree.add_edge(parent_of_v.index, sibling_node.index)
            while unvisited_children.length > 0:
                child = unvisited_children.pop()
                unvisited_edge = self.tree.nodes[child].first_in
                self.tree.set_edge_parent(unvisited_edge, sibling_node)
                self.clade_size[node_idx] -= self.clade_size[child]
                self.clade_size[sibling_node.index] += self.clade_size[child]

        self.subsequence.clear_list(lowest_common_ancestor.index)
        self.subsequence.extend(lowest_common_ancestor.index, clade_index)


    cdef node* partial_traversal(self, int[:] leaves):
        """
        Returns lowest common ancestor of 
        :param leaves: indices of leaf nodes
        :return: queue of nodes such pop() returns all children in the queue before returning a parent
        """
        self.times_visited.clear()
        cdef int num_leaves = len(leaves)
        if num_leaves == 0:
            return <node*> NULL

        # Bottom-up traversal from every leaf node to the root
        cdef int i
        cdef node * v
        cdef int num_visits
        for i in leaves:
            v = self.tree.nodes[i]
            while True:
                num_visits = self.times_visited.get_element(v.index) + 1
                self.times_visited.set_element(v.index, num_visits)
                if num_visits == num_leaves: # reached LCA
                    assert i == leaves[num_leaves-1]
                    break
                if v.first_in is NULL: # reached root
                    break
                v = v.first_in.u

        cdef node * lowest_common_ancestor = v
        return lowest_common_ancestor

    cdef void add_edges_from_subsequence(self, int add_from, int add_to):
        cdef int tree_node = add_from
        cdef list_node * place_in_list
        cdef int last_variant_found = -MAXINT * self.direction
        cdef int variant_idx
        while True:
            place_in_list = self.subsequence.head[tree_node]
            while place_in_list != NULL:
                variant_idx = place_in_list.value
                if variant_idx * self.direction > last_variant_found * self.direction:
                    self.graph.add_edge(variant_idx, add_to)
                    last_variant_found = variant_idx
                place_in_list = place_in_list.next
            if self.tree.nodes[tree_node].first_in is NULL:
                break
            tree_node = self.tree.nodes[tree_node].first_in.u.index

    # cdef walk_to_root(self, node* v)

    cdef tuple[Stack, Stack] get_visited_children(self, node* v):
        cdef Stack visited_children = Stack()
        cdef Stack unvisited_children = Stack()
        cdef int child
        cdef edge* e = v.first_out
        while e is not NULL:
            child = e.v.index
            e = e.next_out
            if self.times_visited.get_element(child) > 0:
                visited_children.push(child)
            else:
                unvisited_children.push(child)
        return visited_children, unvisited_children

    cdef void create_new_root(self):
        cdef node* new_root = self.tree.add_node(-1)
        assert new_root.index < self.num_samples * 2
        self.tree.add_edge(new_root.index, self.root)
        self.subsequence.copy_list(self.root, new_root.index)
        self.root = new_root.index
        self.clade_size[self.root] = self.num_samples

    def get_tree(self) -> DiGraph:
        return self.tree

    def get_graph(self) -> DiGraph:
        return self.graph

    def get_subsequence(self, node: int) -> cnp.ndarray:
        return self.subsequence.extract(node)


cpdef void prune_back_edges(DiGraph forward_graph, DiGraph backward_graph):
    """
    Prunes edges (v,u) from backward_graph if (u,v) is an edge of forward_graph; modifies backward_graph in-place.
    """
    cdef int num_nodes = max(forward_graph.maximum_number_of_nodes, backward_graph.maximum_number_of_nodes)
    cdef IntegerSet neighbors = IntegerSet(num_nodes)
    cdef int node_index
    cdef node * current_node
    cdef node * backward_node
    cdef edge * current_edge
    cdef edge * next_edge
    for node_index in range(num_nodes):
        neighbors.clear()
        current_edge = forward_graph.nodes[node_index].first_in
        while current_edge is not NULL:
            neighbors.add(current_edge.u.index)
            current_edge = current_edge.next_in

        current_edge = backward_graph.nodes[node_index].first_out
        while current_edge is not NULL:
            next_edge = current_edge.next_out
            if neighbors.contains(current_edge.v.index):
                backward_graph.remove_edge(current_edge)
            current_edge = next_edge

cpdef DiGraph reduction_union(DiGraph forward_reduction, DiGraph backward_reduction):
    """
    Computes the transitive reduction of the union of the partial orderings defined by two DAGs, which are the
    transitive reductions of the intersections of some partial ordering with a total ordering and its negation.
    :param forward_reduction: the transitive reduction of intersect(partial ordering, total ordering)
    :param backward_reduction: the transitive reduction of intersect(partial ordering, total ordering reversed)
    :param prune_loops: whether to prune edges (v,u) in backward_reduction if (u,v) is in forward_reduction 
    :return: the transitive reduction of the partial ordering
    """
    cdef int num_nodes = max(forward_reduction.maximum_number_of_nodes, backward_reduction.maximum_number_of_nodes)
    cdef IntegerSet reachable_in_two_hops = IntegerSet(num_nodes)
    cdef int node_index
    cdef node * current_node
    cdef edge * out_edge
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

# Subroutines of reduction_union
cdef void search_two_hops(IntegerSet result, DiGraph first_graph, DiGraph second_graph, int starting_node_index):
    cdef node * starting_node = first_graph.nodes[starting_node_index]
    cdef edge * first_hop
    cdef edge * second_hop
    first_hop = starting_node.first_out
    while first_hop is not NULL:
        second_hop = second_graph.nodes[first_hop.v.index].first_out
        while second_hop is not NULL:
            result.add(second_hop.v.index)
            second_hop = second_hop.next_out
        first_hop = first_hop.next_out

cdef void add_nonredundant_neighbors(DiGraph result, node * starting_node, IntegerSet neighbors_to_exclude):
    cdef edge * out_edge = starting_node.first_out
    while out_edge is not NULL:
        if not neighbors_to_exclude.contains(out_edge.v.index):
            result.add_edge(starting_node.index, out_edge.v.index)
        out_edge = out_edge.next_out

