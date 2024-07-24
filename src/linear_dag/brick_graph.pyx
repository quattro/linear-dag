# brick_graph.pyx
import numpy as np
from .data_structures cimport node, edge, list_node
from .data_structures cimport DiGraph, LinkedListArray, CountingArray, Stack, IntegerList
cimport numpy as cnp
from scipy.sparse import csr_matrix, csc_matrix
import time

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

    def __cinit__(self, int num_samples, int num_variants):
        cnp.import_array()
        self.num_samples = num_samples
        self.num_variants = num_variants
        self.graph = DiGraph(num_variants, num_variants)
        self.graph.initialize_all_nodes()
        self.initialize_tree()
        tree_num_nodes = self.tree.maximum_number_of_nodes
        self.times_visited = CountingArray(tree_num_nodes)
        self.times_revisited = CountingArray(tree_num_nodes)

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
    def from_genotypes(genotypes: csc_matrix) -> BrickGraph:
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
        if len(new_clade) == 0:
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
                self.subsequence.extend(node_idx, clade_index)
                continue

            visited_children, unvisited_children = self.get_visited_children(v)  # TODO avoid unvisited children?
            num_children_unvisited, num_children_visited = unvisited_children.length, visited_children.length

            # v must have a parent node
            if v.first_in == NULL:
                assert node_idx == self.root
                self.create_new_root()
            parent_of_v = v.first_in.u
            assert parent_of_v is not NULL

            # Exactly one visited and one unvisited child: delete v, as there are existing nodes
            # for both intersect(v, new_clade) and intersect(v, new_clade_complement)
            if num_children_visited == 1 and num_children_unvisited == 1:
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
                self.clade_size[node_idx] -= self.clade_size[i]
                continue

            # Exactly one child is w is unvisited: there is an existing node for intersect(v, new_clade_complement);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade)
            if num_children_unvisited == 1:
                i = unvisited_children.pop()
                unvisited_edge = self.tree.nodes[i].first_in
                self.tree.set_edge_parent(unvisited_edge, parent_of_v)
                self.subsequence.extend(node_idx, clade_index)  # v is now a subset of the new clade
                self.clade_size[node_idx] -= self.clade_size[i]
                continue

            # Multiple visited and unvisited children: create new_node for intersect(v, new_clade_complement)
            # and replace v with intersect(v, new_clade)
            sibling_node = self.tree.add_node(-1)
            assert sibling_node.index < self.num_samples * 2
            self.times_visited.set_element(sibling_node.index, 0)
            self.subsequence.copy_list(node_idx, sibling_node.index)
            self.subsequence.extend(node_idx, clade_index)
            self.tree.add_edge(parent_of_v.index, sibling_node.index)
            while unvisited_children.length > 0:
                child = unvisited_children.pop()
                unvisited_edge = self.tree.nodes[child].first_in
                self.tree.set_edge_parent(unvisited_edge, sibling_node)
                self.clade_size[node_idx] -= self.clade_size[child]
                self.clade_size[sibling_node.index] += self.clade_size[child]

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
        cdef list_node * place_in_list = self.subsequence.head[add_from]
        while place_in_list != NULL:
            self.graph.add_edge(place_in_list.value, add_to)
            place_in_list = place_in_list.next

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
