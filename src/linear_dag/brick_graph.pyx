# brick_graph.pyx
from libc.stdlib cimport free, malloc
from .data_structures cimport node, edge, queue_node, list_node
from .data_structures cimport DiGraph, Queue, LinkedListArray, CountingArray
cimport numpy as cnp

cdef class BrickGraph:
    cdef DiGraph graph
    cdef DiGraph tree
    cdef LinkedListArray subsequence
    cdef CountingArray visits
    cdef int num_samples
    cdef int num_variants

    def __cinit__(self, int num_samples, int num_variants, int max_num_edges):
        self.num_samples = num_samples
        self.num_variants = num_variants
        self.graph = DiGraph(num_variants, max_num_edges)
        self.graph.initialize_all_nodes()
        self.initialize_tree()
        tree_num_nodes = self.tree.maximum_number_of_nodes
        self.visits = CountingArray(tree_num_nodes)

    def __init__(self, int num_samples, int num_variants, int max_num_edges):
        pass

    cpdef void initialize_tree(self):
        self.tree = DiGraph(self.num_samples * 2, self.num_samples * 2 - 1)
        cdef int root = self.num_samples
        cdef list edges = [(root, i) for i in range(self.num_samples)]
        self.tree.add_edges_from(edges)
        self.subsequence = LinkedListArray(self.tree.maximum_number_of_nodes)

    @staticmethod
    def from_csc(int[:] indptr, int[:] indices, int num_samples):
        num_variants = len(indptr) - 1
        total_allele_count = len(indices)
        result = BrickGraph(num_samples, num_variants, total_allele_count)

        # Forwards pass
        cdef int i
        for i in range(num_variants):
            carriers = indices[indptr[i]:indptr[i+1]]
            result.intersect_clades(carriers, i)

        # Backwards pass
        result.initialize_tree()
        for i in reversed(range(num_variants)):
            carriers = indices[indptr[i]:indptr[i+1]]
            result.intersect_clades(carriers, i)

        return result

    def to_csr(self) -> "csr_matrix":
        return self.graph.to_csr()

    def get_tree(self) -> DiGraph:
        return self.tree

    def get_graph(self) -> DiGraph:
        return self.graph

    def get_subsequence(self, node: int) -> np.ndarray:
        return self.subsequence.extract(node)

    cdef Queue partial_traversal(self, int[:] leaves):
        """
        Returns the ancestors of a set of leaf nodes, up to their lowest common ancestor, in bottom-up topological ordering.
        :param leaves: indices of leaf nodes
        :return: queue of nodes such pop() returns all children in the queue before returning a parent
        """
        self.visits.clear()
        cdef Queue result = Queue()
        cdef int num_leaves = len(leaves)
        if num_leaves == 0:
            return result

        cdef Queue active_nodes = Queue()
        cdef int i
        for i in leaves:
            active_nodes.push(self.tree.nodes[i])

        # Bottom-up traversal tracking how many visited leaves are descended from each node
        cdef node * v
        cdef int num_visits
        cdef bint v_is_lca = False
        while True:
            v = active_nodes.pop()
            assert v != NULL
            num_visits = self.visits[v.index] + 1
            self.visits[v.index] = num_visits
            v_is_lca = num_visits == num_leaves
            if v_is_lca:
                break

            if v.first_in != NULL:
                active_nodes.push(v.first_in.u)
        active_nodes.clear()
        cdef node * lowest_common_ancestor = v

        # Top-down traversal putting children ahead of parents
        cdef edge* out_edge
        cdef queue_node* place_in_queue = result.push_to_front(lowest_common_ancestor)
        while place_in_queue != NULL:
            v = place_in_queue.value
            out_edge = v.first_out
            while out_edge != NULL:
                if out_edge.v.index in self.visits:
                    result.push_to_front(out_edge.v)
                out_edge = out_edge.next_out
            place_in_queue = place_in_queue.prev
        return result

    cpdef int intersect_clades(self, int[:] new_clade, int clade_index):
        """
        Adds a new clade to a rooted tree and splits existing clades if they intersect with the new clade. Returns the
        lowest common ancestor from the previous tree of nodes in the new clade.
        :param new_clade: integer-valued numpy array
        :param clade_index: integer identifier for the new clade
        :return: lowest common ancestor of nodes in new_clade
        """

        # Compute bottom-up ordering of nodes; update self.visits to tell number of leaves descended from each node
        cdef Queue traversal = self.partial_traversal(new_clade)

        cdef int initial_num_nodes = len(new_clade)
        if initial_num_nodes == 0:
            return -1
        elif initial_num_nodes == 1:
            return new_clade[0]

        # Create edges between new clade identifier and the subsequence associated with the lca
        cdef node * lowest_common_ancestor = traversal.tail.value
        assert lowest_common_ancestor is not NULL
        cdef list_node* place_in_list = self.subsequence.head[lowest_common_ancestor.index]
        while place_in_list != NULL:
            self.graph.add_edge(place_in_list.value, clade_index)
            place_in_list = place_in_list.next

        cdef edge* out_edge
        cdef edge* next_edge
        cdef edge* some_visited_edge
        cdef edge* some_unvisited_edge
        cdef node* new_node
        cdef node* new_root
        cdef node* parent_of_v
        cdef node* v = NULL
        cdef bint v_is_root
        cdef int num_children_visited, num_children_unvisited

        # Partial traversal
        while v != lowest_common_ancestor:
            v = traversal.pop()

            # Count visited and unvisited children of v
            num_children_unvisited = 0
            num_children_visited = 0
            out_edge = v.first_out
            while out_edge is not NULL:
                if self.visits[out_edge.v.index] == 0:
                    num_children_unvisited += 1
                    some_unvisited_edge = out_edge
                else:
                    num_children_visited += 1
                    some_visited_edge = out_edge
                out_edge = out_edge.next_out

            # No unvisited children: nothing to do, unless v is the LCA
            if num_children_unvisited == 0:
                # v is the LCA: its subsequence is now (clade_index)
                if v == lowest_common_ancestor:
                    # add edges between this clade and those comprising v
                    # place_in_list = self.subsequence.head[v.index]
                    # while place_in_list != NULL:
                    #     self.graph.add_edge(place_in_list.value, clade_index)  # TODO replace v.index with correct thing
                    #     place_in_list = place_in_list.next

                    self.subsequence.clear_list(v.index)
                    self.subsequence.extend(v.index, clade_index)
                continue

            assert num_children_visited > 0, "Something went wrong with tree traversal"

            v_is_root = v.first_in == NULL
            if v_is_root:
                new_root = self.tree.add_node(-1)
                self.tree.add_edge(new_root.index, v.index)

            parent_of_v = v.first_in.u

            # Add edges to the brick graph between v and the variants of which parent_of_v is the intersection
            # # TODO can we create recombination nodes at this stage?
            # place_in_list = self.subsequence.head[parent_of_v.index]
            # while place_in_list != NULL:
            #     self.graph.add_edge(place_in_list.value, v.index)  # TODO replace v.index with correct thing
            #     place_in_list = place_in_list.next

            # Exactly one visited and one unvisited child: delete v
            if num_children_visited == 1 and num_children_unvisited == 1:
                self.subsequence.clear_list(v.index)
                self.tree.collapse_node_with_indegree_one(v)
                continue

            # Exactly one child w is visited: replace (v,w) with (parent(v), w)
            if num_children_visited == 1:
                self.tree.set_edge_parent(some_visited_edge, parent_of_v)
                self.visits[v.index] = 0
                continue

            # Exactly one child is w is unvisited: replace (v,w) with (parent(v), w)
            if num_children_unvisited == 1:
                self.tree.set_edge_parent(some_unvisited_edge, parent_of_v)
                if v == lowest_common_ancestor:
                    self.subsequence.clear_list(v.index)
                self.subsequence.extend(v.index, clade_index)  # v is now a subset of the new clade
                continue

            # Multiple visited and unvisited children: create new_node for the intersection between v and new_clade
            new_node = self.tree.add_node(-1)
            self.tree.add_edge(parent_of_v.index, new_node.index)
            self.replace_visited_out_edges(v, new_node)
            if v != lowest_common_ancestor:
                self.subsequence.copy_list(v.index, new_node.index)  # TODO inefficient?
            self.subsequence.extend(new_node.index, clade_index)
            self.visits[new_node.index] = self.visits[v.index]
            self.visits[v.index] = 0

        return lowest_common_ancestor.index

    cdef void replace_visited_out_edges(self, node* u, node* v):
        """
        Replace edges (u, w) for visited nodes w with (v, w)
        """
        cdef edge* out_edge = u.first_out
        while out_edge != NULL:
            next_edge = out_edge.next_out
            if self.visits[out_edge.v.index] > 0:
                self.tree.set_edge_parent(out_edge, v)
            out_edge = next_edge
