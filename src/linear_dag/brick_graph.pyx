# brick_graph.pyx
import numpy as np
from .data_structures cimport node, edge, queue_node, list_node
from .data_structures cimport DiGraph, Queue, LinkedListArray, CountingArray
cimport numpy as cnp
from scipy.sparse import csr_matrix, csc_matrix

cdef class BrickGraph:
    """
    Implements the brick graph algorithm.
    """
    cdef DiGraph graph
    cdef DiGraph tree
    cdef int root
    cdef LinkedListArray subsequence
    cdef CountingArray times_visited
    cdef int num_samples
    cdef int num_variants

    def __cinit__(self, int num_samples, int num_variants):
        cnp.import_array()
        self.num_samples = num_samples
        self.num_variants = num_variants
        self.graph = DiGraph(num_variants, num_variants)
        self.graph.initialize_all_nodes()
        self.initialize_tree()
        tree_num_nodes = self.tree.maximum_number_of_nodes
        self.times_visited = CountingArray(tree_num_nodes)

    cpdef void initialize_tree(self):
        self.tree = DiGraph(self.num_samples * 2, self.num_samples * 2 - 1)
        self.root = self.num_samples
        cdef list edges = [(self.root, i) for i in range(self.num_samples)]
        self.tree.add_edges_from(edges)
        self.subsequence = LinkedListArray(self.tree.maximum_number_of_nodes)

    @staticmethod
    def from_genotypes(genotypes: csc_matrix):
        num_samples, num_variants = genotypes.shape
        result = BrickGraph(num_samples, num_variants)

        # Forward pass
        cdef int i
        for i in range(num_variants):
            carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i + 1]]
            result.intersect_clades(carriers, i)

        # Backward pass
        result.initialize_tree()
        for i in reversed(range(num_variants)):
            carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i+1]]
            result.intersect_clades(carriers, i)

        return result

    def to_csr(self) -> csr_matrix:
        edge_list = self.graph.edge_list()
        parents, children = zip(*edge_list)
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

        # Compute bottom-up ordering of nodes; update self.visits to tell number of leaves descended from each node
        cdef Queue traversal = self.partial_traversal(new_clade)

        cdef int initial_num_nodes = len(new_clade)
        if initial_num_nodes <= 1:
            return

        # Create edges to the new clade from the variants whose intersection forms the LCA
        cdef node * lowest_common_ancestor = traversal.tail.value
        assert lowest_common_ancestor is not NULL
        self.add_edges_from_subsequence(lowest_common_ancestor.index, clade_index)

        cdef edge* out_edge
        cdef edge* visited_edge
        cdef edge* unvisited_edge
        cdef node* sibling_node
        cdef node* new_root
        cdef node* parent_of_v
        cdef node* v = NULL
        cdef bint v_is_root
        cdef int num_children_visited, num_children_unvisited

        # Partial traversal
        while traversal.length > 0:
            v = traversal.pop()
            visited_children, unvisited_children = self.get_visited_children(v)
            num_children_unvisited, num_children_visited = len(unvisited_children), len(visited_children)

            # No unvisited children: means intersect(v, new_clade) == v
            if num_children_unvisited == 0:
                self.subsequence.extend(v.index, clade_index)
                continue

            # v must have a parent node
            if v.first_in == NULL:
                assert v.index == self.root
                self.create_new_root()
            parent_of_v = v.first_in.u
            assert parent_of_v is not NULL

            # Exactly one visited and one unvisited child: delete v, as there are existing nodes
            # for both intersect(v, new_clade) and intersect(v, new_clade_complement)
            if num_children_visited == 1 and num_children_unvisited == 1:
                self.subsequence.clear_list(v.index)
                self.tree.collapse_node_with_indegree_one(v)
                continue

            # Exactly one child w is visited: there is an existing node for intersect(v, new_clade);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade_complement)
            if num_children_visited == 1:
                visited_edge = self.tree.nodes[visited_children.pop()].first_in
                self.tree.set_edge_parent(visited_edge, parent_of_v)
                self.times_visited[v.index] = 0
                continue

            # Exactly one child is w is unvisited: there is an existing node for intersect(v, new_clade_complement);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade)
            if num_children_unvisited == 1:
                unvisited_edge = self.tree.nodes[unvisited_children.pop()].first_in
                self.tree.set_edge_parent(unvisited_edge, parent_of_v)
                self.subsequence.extend(v.index, clade_index)  # v is now a subset of the new clade
                continue

            # Multiple visited and unvisited children: create new_node for intersect(v, new_clade_complement)
            # and replace v with intersect(v, new_clade)
            sibling_node = self.tree.add_node(-1)
            assert sibling_node.index < self.num_samples * 2
            self.times_visited[sibling_node.index] = 0
            self.subsequence.copy_list(v.index, sibling_node.index)
            self.subsequence.extend(v.index, clade_index)
            self.tree.add_edge(parent_of_v.index, sibling_node.index)
            for child in unvisited_children:
                unvisited_edge = self.tree.nodes[child].first_in
                self.tree.set_edge_parent(unvisited_edge, sibling_node)

    cdef Queue partial_traversal(self, int[:] leaves):
        """
        Returns the ancestors of a set of leaf nodes, up to their lowest common ancestor, in bottom-up topological ordering.
        :param leaves: indices of leaf nodes
        :return: queue of nodes such pop() returns all children in the queue before returning a parent
        """
        self.times_visited.clear()
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
            num_visits = self.times_visited[v.index] + 1
            self.times_visited[v.index] = num_visits
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
                if out_edge.v.index in self.times_visited:
                    result.push_to_front(out_edge.v)
                out_edge = out_edge.next_out
            place_in_queue = place_in_queue.prev
        return result

    cdef void add_edges_from_subsequence(self, int add_from, int add_to):
        cdef list_node * place_in_list = self.subsequence.head[add_from]
        while place_in_list != NULL:
            self.graph.add_edge(place_in_list.value, add_to)
            place_in_list = place_in_list.next

    cdef tuple get_visited_children(self, node* v):
        cdef list visited_children = []
        cdef list unvisited_children = []
        cdef int child
        for child in self.tree.successors(v.index):
            if self.times_visited[child] > 0:
                visited_children.append(child)
            else:
                unvisited_children.append(child)
        return visited_children, unvisited_children

    cdef void create_new_root(self):
        cdef node* new_root = self.tree.add_node(-1)
        assert new_root.index < self.num_samples * 2
        self.tree.add_edge(new_root.index, self.root)
        self.subsequence.copy_list(self.root, new_root.index)
        self.root = new_root.index

    def get_tree(self) -> DiGraph:
        return self.tree

    def get_graph(self) -> DiGraph:
        return self.graph

    def get_subsequence(self, node: int) -> cnp.ndarray:
        return self.subsequence.extract(node)
