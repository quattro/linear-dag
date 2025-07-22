# recombination.pxd

cimport numpy as cnp
from .data_structures cimport LinkedListArray, ModHeap, CountingArray, DiGraph, node, edge

cdef class Recombination(DiGraph):
    cdef long[:] clique
    cdef LinkedListArray clique_rows
    cdef ModHeap clique_size_heap
    cdef long num_cliques

    cpdef void compute_cliques(self)
    cpdef void collect_cliques(self)
    cpdef void find_recombinations(self)
    cpdef int max_clique(self)
    cpdef void factor_clique(self, int clique_index)
    cpdef collapse_clique(self, long c, long new_node, long[:] edge_indices)

    cdef void update_trios(self, long p, long[:] edges, long new_node)
    cdef long neighboring_trio(self, long edge_index, long p)