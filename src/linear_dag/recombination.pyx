cimport numpy as cnp
import numpy as np
from .data_structures cimport LinkedListArray, ModHeap, CountingArray, DiGraph, node, edge
import sys
cdef class Recombination(DiGraph):
    """
    Implements
    """
    cdef int[:] clique
    cdef LinkedListArray clique_rows
    cdef ModHeap clique_size_heap
    cdef int num_cliques

    def __init__(self, int num_nodes, int num_edges):
        cnp.import_array()  # Necessary for initializing the C API
        self.clique = -np.ones(num_edges, dtype=np.intc)
        self.clique_rows = LinkedListArray(num_edges)
        self.clique_size_heap = None
        self.num_cliques = 0
        super().__init__(num_nodes, num_edges)

    @property
    def get_heap(self):
        return self.clique_size_heap.act_heap, self.clique_size_heap.priority

    @property
    def get_clique_rows(self):
        return [np.asarray(self.clique_rows.extract(n)) for n in range(self.clique_rows.n)]

    @property
    def get_cliques(self):
        return np.asarray(self.clique)

    @staticmethod
    def from_graph(brick_graph: DiGraph) -> Recombination:
        result = Recombination(brick_graph.maximum_number_of_nodes, brick_graph.maximum_number_of_edges)
        result.add_edges_from(brick_graph.edge_list())
        result.compute_cliques()
        result.collect_cliques()
        return result

    cpdef void compute_cliques(self):
        cdef CountingArray right_parent_to_clique = CountingArray(self.number_of_nodes)
        cdef int left_parent, right_parent, clique_idx
        cdef edge* out_edge

        # Find trios of the form (left_parent, right_parent, child)
        for left_parent in range(self.maximum_number_of_nodes):
            if not self.is_node[left_parent]:
                continue

            out_edge = self.nodes[left_parent].first_out
            while out_edge is not NULL:
                if out_edge.next_in is NULL:
                    out_edge = out_edge.next_out
                    continue  # left_parent is the rightmost parent of this child

                right_parent = out_edge.next_in.u.index

                # keep track of previously seen right_parent for this left_parent
                if right_parent in right_parent_to_clique:
                    clique_idx = right_parent_to_clique[right_parent]
                else:
                    clique_idx = self.num_cliques
                    right_parent_to_clique[right_parent] = clique_idx
                    self.num_cliques += 1
                self.clique[out_edge.index] = clique_idx

                out_edge = out_edge.next_out

            # clears in O(1)
            right_parent_to_clique.clear()

    cpdef void collect_cliques(self):
        cdef int[:] what = np.where(np.asarray(self.clique) != -1)[0].astype(np.intc)
        cdef int[:] where = np.arange(self.num_cliques, dtype=np.intc)
        cdef int[:] which = np.take(self.clique, what)

        self.clique_rows.assign(what, where, which)
        self.clique_size_heap = ModHeap(self.clique_rows.length)

    cpdef void find_recombinations(self):
        cdef int c = self.max_clique()
        while c >= 0: # -1 when none remain
            self.factor_clique(c)
            c = self.max_clique()

    cpdef int max_clique(self):
        if self.clique_size_heap:
            return self.clique_size_heap.pop()
        else:
            return -1

    cpdef void factor_clique(self, int clique_index):
        print(clique_index)
        sys.stdout.flush()
        cdef int[:] clique_rows = self.clique_rows.extract(clique_index)
        if len(clique_rows) <= 1:
            return
        cdef node* new_node = self.add_node(-1)

        # Update neighboring trios
        cdef int p
        for p in range(2):
            self.update_trios(p, clique_rows, new_node.index)
        assert self.clique_rows.length[clique_index] == len(clique_rows)

        # Replace clique with a star
        self.collapse_clique(clique_index, new_node.index)

    cdef void update_trios(self, int p, int[:] edges, int new_node):
        cdef int num_trios = len(edges)

        cdef int[:] neighboring_trios = np.empty(num_trios, dtype=np.intc)
        cdef int neighbor
        cdef int num_neighbors = 0
        cdef int i
        for i in range(num_trios):
            neighbor = self.neighboring_trio(edges[i], p)
            if neighbor >= 0:
                neighboring_trios[num_neighbors] = neighbor
                num_neighbors += 1

        # Re-assign clique[neighboring_trios]
        affected_cliques, which_affected_cliques = np.unique(
            np.take(self.clique, neighboring_trios),
            return_inverse=True)
        num_new_cliques = len(affected_cliques)
        for i in range(num_neighbors):
            self.clique[neighboring_trios[i]] = which_affected_cliques[i] + self.num_cliques
        new_cliques = np.arange(self.num_cliques, self.num_cliques + num_new_cliques, dtype=np.intc)
        self.num_cliques += num_new_cliques

        # Assign neighbors to cliqueRows[new cliques]
        self.clique_rows.assign(neighboring_trios, new_cliques, which_affected_cliques)

        # Update clique size heap
        cdef int new_clique
        for new_clique in new_cliques:
            self.clique_size_heap.push(new_clique, self.clique_rows.length[new_clique])

        # Remove them from cliqueRows[old cliques]
        cdef int old_clique
        for i in range(num_new_cliques):
            old_clique = affected_cliques[i]
            self.clique_rows.remove_difference(new_cliques[i], old_clique)
            self.clique_size_heap.push(old_clique, self.clique_rows.length[old_clique])


    cdef int neighboring_trio(self, int edge_index, int p):
        cdef edge* e = self.edges[edge_index]
        assert e is not NULL
        assert e.next_in is not NULL

        if p == 0:
            e = e.prev_in
        elif p == 1:
            e = e.next_in.next_in
        else:
            raise ValueError
        if e is NULL:
            return -1
        return e.index

    cpdef collapse_clique(self, int c, int new_node):
        cdef int i
        cdef int[:] edge_indices = self.clique_rows.extract(c)
        cdef edge* e = self.edges[edge_indices[0]]

        # Add edges from parents of the trio to the new node
        cdef int parent
        parent = e.u.index
        self.add_edge(parent, new_node)
        parent = e.next_in.u.index
        self.add_edge(parent, new_node)

        # Replace trio edges with a single edge from new node to each child
        for i in edge_indices:
            e = self.edges[i]
            self.set_edge_parent(e, self.nodes[new_node])
            self.remove_edge(e.next_in)

        self.clique_rows.clear_list(c)
