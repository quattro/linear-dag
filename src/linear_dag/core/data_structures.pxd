# data_structures.pxd

cdef class IntegerList:
    cdef int* value
    cdef int length
    cdef int maximum_length

    cdef void push(self, int value)
    cdef int pop(self)

cdef class Stack:
    cdef int[:] data
    cdef int top
    cdef int capacity
    cdef int length
    cdef void push(self, int value)
    cdef int pop(self)
    cdef void clear(self)

cdef class InfiniteStack(Stack):
    cdef int last

cdef struct list_node:
    long next  # Index of next node, -1 for NULL
    long value

cdef class LinkedListArray:
    cdef list_node* nodes  # Pool of nodes
    cdef long* head  # Array of head indices
    cdef long* tail  # Array of tail indices
    cdef long[:] length
    cdef long n  # Number of lists
    cdef long capacity  # Current capacity of node pool
    cdef long next_free  # Index of next free node
    cdef public double time_argsort
    cdef public double time_loop
    
    cdef void _resize(self)
    cdef long _get_free_node(self)
    cdef void _extend(self, long n, long value)
    cdef void _insert(self, long n, long value)
    cdef void _remove(self, long n, long node_idx, long predecessor_idx)
    cdef void _assign(self, long[:] what, long[:] where, long[:] which)
    cdef void _remove_difference(self, long n, long m)
    cdef void _clear_list(self, long n)
    cdef void _copy_list(self, long n, long m)
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
