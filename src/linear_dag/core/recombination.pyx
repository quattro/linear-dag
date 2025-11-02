cimport numpy as cnp
import numpy as np
from .data_structures cimport CountingArray, LinkedListArray, ModHeap
from .digraph cimport DiGraph, node, edge

import sys

cdef class Recombination(DiGraph):
    """
    Implements the find-recombinations algorithm. Usage:
    recombination = Recombination.from_graph(brick_graph)
    recombination.find_recombinations()
    """
    cdef long[:] clique
    cdef LinkedListArray clique_rows
    cdef ModHeap clique_size_heap
    cdef long num_cliques
    cdef CountingArray unique_cliques_tracker


    def __init__(self, long num_nodes, long num_edges):
        cnp.import_array()  # Necessary for initializing the C API
        # Call parent __init__ to initialize the graph structure
        # Note: __cinit__ was already called with these same arguments
        DiGraph.__init__(self, num_nodes, num_edges)
        
        self.clique = -np.ones(num_edges, dtype=np.int64)
        self.clique_rows = LinkedListArray(num_edges)
        self.clique_size_heap = None
        self.num_cliques = 0
        self.unique_cliques_tracker = CountingArray(num_edges)


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
        n, m = brick_graph.maximum_number_of_nodes, brick_graph.maximum_number_of_edges
        result = Recombination(n + m, m)
        result.copy_from(brick_graph)
        result.compute_cliques()
        result.collect_cliques()
        
        return result

    cpdef void compute_cliques(self):
        # cdef CountingArray right_parent_to_clique = CountingArray(self.number_of_nodes)
        cdef CountingArray right_parent_to_clique = CountingArray(self.maximum_number_of_nodes)
        cdef long left_parent, right_parent, clique_idx
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
        cdef long[:] what = np.where(np.asarray(self.clique) != -1)[0].astype(np.int64)
        cdef long[:] where = np.arange(self.num_cliques, dtype=np.int64)
        cdef long[:] which = np.take(self.clique, what)

        self.clique_rows.assign(what, where, which)
        self.clique_size_heap = ModHeap(np.asarray(self.clique_rows.length, dtype=np.int64))

    cpdef void find_recombinations(self):
        cdef long c
        cdef long[:] clique_rows
        cdef node* new_node
        cdef long p
        
        c = self.max_clique()
        
        while c >= 0: # -1 when none remain
            clique_rows = self.clique_rows.extract(c)
            
            if len(clique_rows) <= 1:
                c = self.max_clique()
                if c >= 0 and self.clique_rows.length[c] <= 1:
                    break
                continue
            
            new_node = self.add_node(-1)
            
            for p in range(2):
                self.update_trios(p, clique_rows, new_node.index)
            
            assert self.clique_rows.length[c] == len(clique_rows)
            
            self.collapse_clique(c, new_node.index, clique_rows)
            
            c = self.max_clique()
            if self.clique_rows.length[c] <= 1:
                break

    cpdef int max_clique(self):
        if self.clique_size_heap:
            return self.clique_size_heap.pop()
        else:
            return -1

    cpdef void factor_clique(self, int clique_index):
        cdef long[:] clique_rows = self.clique_rows.extract(clique_index)
        if len(clique_rows) <= 1:
            return
        cdef node* new_node = self.add_node(-1)

        # Update neighboring trios
        cdef long p
        for p in range(2):
            self.update_trios(p, clique_rows, new_node.index)
        assert self.clique_rows.length[clique_index] == len(clique_rows)

        # Replace clique with a star
        self.collapse_clique(clique_index, new_node.index, clique_rows)

    cdef void update_trios(self, long p, long[:] edges, long new_node):
        cdef long num_trios = len(edges)

        cdef long[:] neighboring_trios = np.empty(num_trios, dtype=np.int64)
        cdef long neighbor
        cdef long num_neighbors = 0
        cdef long i
        for i in range(num_trios):
            neighbor = self.neighboring_trio(edges[i], p)
            if neighbor >= 0:
                neighboring_trios[num_neighbors] = neighbor
                num_neighbors += 1
        neighboring_trios = neighboring_trios[:num_neighbors]

        # Re-assign clique[neighboring_trios]
        # Arrays to store unique cliques and mapping
        cdef long[:] affected_cliques = np.empty(num_neighbors, dtype=np.int64)
        cdef long[:] which_affected_cliques = np.empty(num_neighbors, dtype=np.int64)
        cdef long num_unique = 0
        cdef long clique_val, unique_idx
        
        # Find unique cliques using CountingArray
        self.unique_cliques_tracker.clear()
        for i in range(num_neighbors):
            clique_val = self.clique[neighboring_trios[i]]
            
            if not self.unique_cliques_tracker.contains(clique_val):
                # Not seen before - add to unique list
                affected_cliques[num_unique] = clique_val
                self.unique_cliques_tracker.set_element(clique_val, num_unique)
                which_affected_cliques[i] = num_unique
                num_unique += 1
            else:  # Already seen
                unique_idx = self.unique_cliques_tracker.get_element(clique_val)
                which_affected_cliques[i] = unique_idx
        
        affected_cliques = affected_cliques[:num_unique]
        
        num_new_cliques = len(affected_cliques)
        for i in range(num_neighbors):
            self.clique[neighboring_trios[i]] = which_affected_cliques[i] + self.num_cliques
        cdef long[:] new_cliques = np.arange(self.num_cliques, self.num_cliques + num_new_cliques, dtype=np.int64)
        self.num_cliques += num_new_cliques

        # Assign neighbors to cliqueRows[new cliques]
        self.clique_rows.assign(neighboring_trios, new_cliques, which_affected_cliques)

        # Update clique size heap
        cdef long new_clique
        for new_clique in new_cliques:
            self.clique_size_heap.push(new_clique, self.clique_rows.length[new_clique])

        # Remove them from cliqueRows[old cliques]
        cdef long old_clique
        for i in range(num_new_cliques):
            old_clique = affected_cliques[i]
            self.clique_rows.remove_difference(new_cliques[i], old_clique)
            self.clique_size_heap.push(old_clique, self.clique_rows.length[old_clique])


    cdef long neighboring_trio(self, long edge_index, long p):
        cdef edge* e = self.get_edge(edge_index)
        assert e is not NULL
        assert e.next_in is not NULL

        if p == 0:
            e = e.prev_in
        elif p == 1:
            e = e.next_in
        else:
            raise ValueError

        if e is NULL:
            return -1
        if e.next_in is NULL:
            return -1
        return e.index

    cpdef collapse_clique(self, long c, long new_node, long[:] edge_indices):
        cdef long i
        cdef edge* e = self.get_edge(edge_indices[0])

        # Add edges from parents of the trio to the new node
        cdef long left_parent = e.u.index
        cdef long right_parent = e.next_in.u.index
        self.add_edge(left_parent, new_node)
        self.add_edge(right_parent, new_node)

        # Replace trio edges with a single edge from new node to each child
        for i in edge_indices:
            e = self.get_edge(i)
            assert e.u.index == left_parent and e.next_in.u.index == right_parent
            self.set_edge_parent(e.next_in, &self.nodes[new_node])
            self.remove_edge(e)

        self.clique_rows.clear_list(c)
