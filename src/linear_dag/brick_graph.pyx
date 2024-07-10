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

    def get_subsequence(self, node: int) -> cnp.ndarray:
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

    cdef void add_edges(self, int parent_tree_node_index, int child_clade_index):
        cdef list_node * place_in_list = self.subsequence.head[parent_tree_node_index]
        while place_in_list != NULL:
            self.graph.add_edge(place_in_list.value, child_clade_index)
            place_in_list = place_in_list.next

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

        # Create edges to the new clade from the variants whose intersection forms the LCA
        cdef node * lowest_common_ancestor = traversal.tail.value
        assert lowest_common_ancestor is not NULL
        self.add_edges(lowest_common_ancestor.index, clade_index)

        cdef edge* out_edge
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

            # No unvisited children: means intersect(v, new_clade) == v
            if num_children_unvisited == 0:
                self.subsequence.extend(v.index, clade_index)
                continue

            v_is_root = v.first_in == NULL
            if v_is_root:
                new_root = self.tree.add_node(-1)
                self.tree.add_edge(new_root.index, v.index)

            parent_of_v = v.first_in.u

            # Exactly one visited and one unvisited child: delete v, as there are existing nodes
            # for both intersect(v, new_clade) and intersect(v, new_clade_complement)
            if num_children_visited == 1 and num_children_unvisited == 1:
                self.tree.collapse_node_with_indegree_one(v)
                self.subsequence.clear_list(v.index)
                continue

            # Exactly one child w is visited: there is an existing node for intersect(v, new_clade);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade_complement)
            if num_children_visited == 1:
                self.tree.set_edge_parent(some_visited_edge, parent_of_v)
                self.visits[v.index] = 0
                continue

            # Exactly one child is w is unvisited: there is an existing node for intersect(v, new_clade_complement);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade)
            if num_children_unvisited == 1:
                self.tree.set_edge_parent(some_unvisited_edge, parent_of_v)
                self.subsequence.extend(v.index, clade_index)  # v is now a subset of the new clade
                continue

            # Multiple visited and unvisited children: create new_node for intersect(v, new_clade_complement)
            # and replace v with intersect(v, new_clade)
            new_node = self.create_sibling_node(v)
            self.replace_parent_of_unvisited_out_edges(v, new_node)
            self.subsequence.extend(v.index, clade_index)

        return lowest_common_ancestor.index

    cdef node* create_sibling_node(self, node* v):
        cdef node* parent_of_v = v.first_in.u
        new_node = self.tree.add_node(-1)
        self.tree.add_edge(parent_of_v.index, new_node.index)
        self.subsequence.copy_list(v.index, new_node.index)
        self.visits[new_node.index] = 0
        return new_node

    cdef void replace_parent_of_unvisited_out_edges(self, node* u, node* v):
        """
        Replace edges (u, w) with (v, w) for unvisited nodes w
        """
        cdef edge* out_edge = u.first_out
        while out_edge != NULL:
            next_edge = out_edge.next_out
            if self.visits[out_edge.v.index] == 0:
                self.tree.set_edge_parent(out_edge, v)
            out_edge = next_edge
