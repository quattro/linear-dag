# digraph.pxd

cdef struct node:
    int index
    edge* first_in
    edge* first_out

cdef struct edge:
    int index
    node* u
    node* v
    edge* next_in
    edge* next_out
    edge* prev_in
    edge* prev_out

cdef class DiGraph:
    cdef node* nodes
    cdef bint* is_node
    cdef edge** edge_arrays
    cdef node* available_node
    cdef edge* available_edge
    cdef long maximum_number_of_nodes
    cdef long number_of_available_nodes
    cdef long maximum_number_of_edges
    cdef long number_of_available_edges
    cdef long edge_array_length
    
    cdef edge* get_edge(self, long index)
    cdef void extend_edge_array(self)
    cdef node* add_node(self, long node_index)
    cdef edge* add_edge(self, long u_index, long v_index)
    cdef void remove_edge(self, edge* e)
    cdef void remove_node(self, node* u)
    cdef void set_edge_child(self, edge* e, node* v)
    cdef void set_edge_parent(self, edge* e, node* u)
    cdef void patch_in_pointers(self, edge* e)
    cdef void patch_out_pointers(self, edge* e)
    cdef void extend_node_array(self, int new_maximum_number_of_nodes)
    cdef void replace_node(self, node* u, node* v)
    cdef void collapse_node(self, node* v)
    cdef void collapse_node_with_indegree_one(self, node* v)
    cdef void collapse_node_with_outdegree_one(self, node* u)
    cdef long number_of_successors(self, node* u)
    cdef long number_of_predecessors(self, node* u)
    cpdef bint has_node(self, long node_index)
    cpdef long some_parent(self, long node_index)
    cpdef long some_child(self, long node_index)
    cpdef void initialize_all_nodes(self)
    cpdef long[:] out_degree(self)
    cpdef long[:] in_degree(self)
    cpdef long[:] reverse_topological_sort(self)
