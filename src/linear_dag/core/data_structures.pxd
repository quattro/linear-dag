# data_structures.pxd

cdef class IntegerList:
    cdef int* value
    cdef int length
    cdef int maximum_length

    cdef void push(self, int value)
    cdef int pop(self)

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

cdef class InfiniteStack(Stack):
    cdef int last

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
    long value

cdef class LinkedListArray:
    cdef list_node** head  # Array of pointers to the first element of each linked list
    cdef list_node** tail  # Array of pointers to the last element of each linked list
    cdef int[:] length  # Length of each linked list
    cdef int n  # Number of linked lists
    cdef void extend(self, long n, long value)
    cdef void insert(self, long n, long value)
    cdef void remove(self, long n, list_node* element, list_node* predecessor)
    cdef void assign(self, long[:] what, long[:] where, long[:] which)
    cdef void remove_difference(self, long n, long m)
    cdef void clear_list(self, long n)
    cdef copy_list(self, long n, long m)
    cpdef long[:] extract(self, long n)

cdef class IntegerSet:
    cdef long length
    cdef int[:] last_cleared
    cdef int times_cleared

    cdef bint contains(self, int index)
    cdef void add(self, int index)
    cdef void remove(self, int index)
    cpdef void clear(self)

cdef class CountingArray(IntegerSet):
    cdef int[:] count

    cdef int get_element(self, int index)
    cdef int increment_element(self, int index, int increment)
    cdef void set_element(self, int index, int value)


cdef struct node:
    long index
    edge* first_in
    edge* first_out

cdef struct edge:
    node* u
    node* v
    edge* next_in
    edge* next_out
    edge* prev_in
    edge* prev_out
    long index

cdef class DiGraph:
    cdef node** nodes
    cdef bint* is_node
    cdef edge** edges
    cdef Stack available_nodes
    cdef Stack available_edges
    cdef long maximum_number_of_nodes
    cdef long maximum_number_of_edges

    cpdef bint has_node(self, long node_index)
    cpdef long some_parent(self, long node_index)
    cpdef long some_child(self, long node_index)
    cpdef void initialize_all_nodes(self)
    cdef node* add_node(self, long node_index)
    cdef edge* add_edge(self, long u_index, long v_index)
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
    cdef void collapse_node_with_outdegree_one(self, node * u)
    cdef long number_of_successors(self, node * u)
    cdef long number_of_predecessors(self, node * u)
    cpdef long[:] out_degree(self)
    cpdef long[:] in_degree(self)
    cpdef long[:] reverse_topological_sort(self)

cdef class HeapNode:
    cdef public long priority
    cdef public long index

cdef class ModHeap:
    cdef public list act_heap
    cdef long[:] priority
    cdef public long n

    cdef list _create_heap(self, long[:] priority)
    cpdef void push(self, long index, long priority)
    cpdef long pop(self)
