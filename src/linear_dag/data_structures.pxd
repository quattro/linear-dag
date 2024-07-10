# data_structures.pxd


cdef struct stack_node:
    int value
    stack_node* next
    stack_node* prev

cdef class Stack:
    cdef stack_node * head
    cdef int length
    cdef stack_node* push(self, int value)
    cdef int pop(self)
    cdef void remove(self, stack_node * element)
    cdef void clear(self)

cdef struct queue_node:
    node* value
    queue_node* next
    queue_node* prev

cdef class Queue:
    cdef queue_node * head
    cdef queue_node * tail
    cdef int length
    cdef queue_node* push(self, node* value)
    cdef queue_node * push_to_front(self, node * value)
    cdef node* pop(self)
    cdef void clear(self)

cdef struct list_node:
    list_node* next
    int value

cdef class LinkedListArray:
    cdef list_node** head  # Array of pointers to the first element of each linked list
    cdef list_node** tail  # Array of pointers to the last element of each linked list
    cdef int[:] length  # Length of each linked list
    cdef int n  # Number of linked lists
    cdef void extend(self, int n, int value)
    cdef void remove(self, int n, list_node* element, list_node* predecessor)
    cdef void assign(self, int[:] what, int[:] where, int[:] which)
    cdef void remove_difference(self, int n, int m)
    cdef void clear_list(self, int n)
    cdef copy_list(self, int n, int m)
    cpdef int[:] extract(self, int n)

cdef class CountingArray:
    cdef int[:] count
    cdef int[:] last_cleared
    cdef int times_cleared
    cdef int length
    cpdef void clear(self)


cdef struct node:
    int index
    int value  # TODO delete?
    edge* first_in
    edge* first_out

cdef struct edge:
    node* u
    node* v
    edge* next_in
    edge* next_out
    edge* prev_in
    edge* prev_out
    int index

cdef class DiGraph:
    cdef node** nodes
    cdef bint* is_node
    cdef edge** edges
    cdef Stack available_nodes
    cdef Stack available_edges
    cdef int maximum_number_of_nodes
    cdef int maximum_number_of_edges

    cpdef int some_parent(self, int node_index)
    cpdef int some_child(self, int node_index)
    cpdef void initialize_all_nodes(self)
    cdef node* add_node(self, int node_index)
    cdef edge* add_edge(self, int u_index, int v_index)
    cdef void extend_edge_array(self, int new_maximum_number_of_edges)
    cdef void extend_node_array(self, int new_maximum_number_of_nodes)
    cdef void set_edge_child(self, edge* e, node* v)
    cdef void set_edge_parent(self, edge* e, node* u)
    cdef void remove_edge(self, edge* e)
    cdef void remove_node(self, node* u)
    cdef void replace_node(self, node* u, node* v)
    cdef void patch_out_pointers(self, edge* e)
    cdef void patch_in_pointers(self, edge* e)
    cdef void collapse_node(self, node * u)
    cdef void collapse_node_with_indegree_one(self, node * u)
    cdef int number_of_successors(self, node * u)
    cdef int number_of_predecessors(self, node * u)

cdef class Trie(DiGraph):
    cdef int[:] elements
    cdef int[:] termini
    cdef int[:] depth

    cdef void initialize_branches(self, int number_of_branches)
    cdef void clear_branch(self, int branch_index)
    cdef void insert_branch(self, int branch_index, int new_branch_index)
    cdef void extend_branch(self, int branch_index, int value)
    cdef int[:] read_branch(self, int branch_index)
    cdef void extend_node_array(self, int new_maximum_number_of_nodes)


cdef class HeapNode:
    cdef public int priority
    cdef public int index

cdef class ModHeap:
    cdef public list act_heap
    cdef int[:] priority
    cdef public int n

    cdef list _create_heap(self, int[:] priority)
    cpdef void push(self, int index, int priority)
    cpdef int pop(self)
