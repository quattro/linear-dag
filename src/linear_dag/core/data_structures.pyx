#data_structures.pyx
#define NPY_NO_DEPRECATED_API

from typing import Type
import heapq
from libc.stdlib cimport free, malloc, realloc, qsort
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
cimport numpy as cnp
from .data_structures cimport list_node # in data_structures.pxd

cdef int MAXINT = 32767

cdef int compare_long(const void* a, const void* b) noexcept nogil:
    return <int>((<long*>a)[0] - (<long*>b)[0])

cdef class IntegerList:
    """Fixed-length list of integers supporting push() and pop()"""
    # cdef int* value
    # cdef int length
    # cdef int maximum_length

    def __cinit__(self, int maximum_length):
        self.maximum_length = maximum_length
        self.value = <int*> malloc(maximum_length * sizeof(int))
        self.length = 0

    def __dealloc__(self):
        free(self.value)

    cdef void push(self, int value):
        if self.length >= self.maximum_length:
            raise ValueError("IntegerList is full")
        self.value[self.length] = value
        self.length += 1

    cdef int pop(self):
        self.length -= 1
        if self.length < 0:
            raise ValueError("IntegerList is empty")
        return self.value[self.length]


cdef class Stack:
    """
    Stack of integers implemented as a dynamic array.
    """
    # cdef int[:] data
    # cdef int top
    # cdef int capacity
    
    def __init__(self, int capacity = 1024):
        self.capacity = capacity
        self.data = np.empty(capacity, dtype=np.int32)
        self.top = -1
        self.length = 0
    
    cdef void push(self, int value):
        if self.top + 1 == self.capacity:
            self.capacity *= 2
            self.data = np.resize(self.data, self.capacity)
        self.top += 1
        self.data[self.top] = value
        self.length += 1
    
    cdef int pop(self):
        if self.top == -1:
            raise ValueError("Stack is empty")
        self.top -= 1
        self.length -= 1
        return self.data[self.top + 1]
    
    cdef void clear(self):
        self.top = -1
        self.length = 0

cdef class InfiniteStack(Stack):
    """
    Stack of integers initialized (implicitly) as 0, 1, 2, ...
    """
    # cdef int last
    def __init__(self):
        super().__init__()
        self.last = -1

    cdef int pop(self):
        if self.top == -1:
            self.last += 1
            return self.last
        return Stack.pop(self)


cdef class IntegerSet:
    """Set of integers between 0 and n-1 supporting O(1) add, remove, check for inclusion, and clear."""
    # cdef int length
    # cdef int[:] last_cleared
    # cdef int times_cleared

    def __init__(self, long length):
        self.length = length
        self.last_cleared = np.zeros(length, dtype=np.intc)
        self.times_cleared = 1
        cnp.import_array()  # Necessary for initializing the C API

    cdef bint contains(self, int index):
        if index >= self.length or index < 0:
            return False
        return self.last_cleared[index] == self.times_cleared

    cdef void add(self, int index):
        if index >= self.length or index < 0:
            raise IndexError(f"Index {index} out of bounds")
        self.last_cleared[index] = self.times_cleared

    cdef void remove(self, int index):
        if index >= self.length or index < 0:
            raise IndexError(f"Index {index} out of bounds")
        self.last_cleared[index] = 0

    cpdef void clear(self):
        self.times_cleared += 1
        if self.times_cleared == MAXINT:
            self.last_cleared = np.zeros(self.length, dtype=np.intc)
            self.times_cleared = 1

cdef class CountingArray(IntegerSet):
    """Array that keeps track of the last time each element was modified, allowing it to be cleared in O(1) time"""
    # cdef int[:] count

    def __init__(self, long length):
        self.count = np.zeros(length, dtype=np.intc)
        super().__init__(length)

    def __contains__(self, int index) -> bint:  # index in counting_array
        return self.contains(index)

    def __setitem__(self, int index, int value):  # counting_array[index] = value
        self.set_element(index, value)

    def __getitem__(self, int index) -> int:  # value = counting_array[index]
        if index >= self.length or index < 0:
            raise IndexError(f"Index {index} out of bounds")
        return self.get_element(index)

    cdef int get_element(self, int index):
        if not self.contains(index):
            return 0
        return self.count[index]

    cdef int increment_element(self, int index, int increment):
        if not self.contains(index):
            self.set_element(index, increment)
            return increment
        self.count[index] += increment
        return self.count[index]

    cdef void set_element(self, int index, int value):
        self.add(index)
        self.count[index] = value


cdef class LinkedListArray:
    """Array of linked lists using indices instead of pointers to enable dynamic resizing."""

    def __cinit__(self, int n, long initial_capacity=1000):
        self.n = n
        self.capacity = initial_capacity
        self.next_free = 0
        
        self.nodes = <list_node*> malloc(self.capacity * sizeof(list_node))
        self.head = <long*> malloc(n * sizeof(long))
        self.tail = <long*> malloc(n * sizeof(long))
        self.length = np.zeros(n, dtype=np.int64)

    def __init__(self, int n, long initial_capacity=1000):
        cnp.import_array()
        if not self.nodes or not self.head or not self.tail:
            raise MemoryError("Could not allocate LinkedListArray.")

        cdef long i
        for i in range(n):
            self.head[i] = -1
            self.tail[i] = -1
            self.length[i] = 0
        
        # Initialize free list: each node points to the next
        for i in range(initial_capacity):
            self.nodes[i].next = i + 1
        self.nodes[initial_capacity - 1].next = -1  # Last node has no next

    def __dealloc__(self):
        free(self.nodes)
        free(self.head)
        free(self.tail)

    cdef void _resize(self):
        """Double the capacity of the node pool"""
        cdef long old_capacity = self.capacity
        cdef long new_capacity = self.capacity * 2
        cdef list_node* new_nodes = <list_node*> realloc(self.nodes, new_capacity * sizeof(list_node))
        if not new_nodes:
            raise MemoryError("Could not resize node pool")
        
        self.nodes = new_nodes
        self.capacity = new_capacity
        
        # Initialize new nodes in free list and link to existing free list
        cdef long i
        for i in range(old_capacity, new_capacity):
            self.nodes[i].next = i + 1
        # Link last new node to old free list
        self.nodes[new_capacity - 1].next = self.next_free
        
        # Point next_free to the first new node
        self.next_free = old_capacity

    cdef long _get_free_node(self):
        """Get the next free node index from the pool"""
        if self.next_free == -1:
            self._resize()
        cdef long idx = self.next_free
        self.next_free = self.nodes[idx].next
        return idx

    cdef void _extend(self, long n, long value):
        if n < 0 or n >= self.n:
            raise ValueError("LinkedList index out of bounds.")

        cdef long new_idx = self._get_free_node()
        self.nodes[new_idx].value = value
        self.nodes[new_idx].next = -1

        if self.tail[n] == -1:
            self.head[n] = new_idx
            self.tail[n] = new_idx
        else:
            self.nodes[self.tail[n]].next = new_idx
            self.tail[n] = new_idx

        self.length[n] += 1

    cdef void _insert(self, long n, long value):
        """Insert a new element into a sorted list"""
        if n < 0 or n >= self.n:
            raise ValueError("LinkedList index out of bounds.")

        cdef long new_idx = self._get_free_node()
        self.nodes[new_idx].value = value

        cdef long place_idx = self.head[n]
        cdef long prev_idx = -1
        while place_idx != -1:
            if value > self.nodes[place_idx].value:
                break
            prev_idx = place_idx
            place_idx = self.nodes[place_idx].next

        if prev_idx == -1:
            self.head[n] = new_idx
        else:
            self.nodes[prev_idx].next = new_idx

        if place_idx == -1:
            self.tail[n] = new_idx
        self.nodes[new_idx].next = place_idx

        self.length[n] += 1

    cdef void _remove(self, long n, long node_idx, long predecessor_idx):
        """Removes a node at given index from list n and returns it to the free pool"""
        if predecessor_idx == -1:
            self.head[n] = self.nodes[node_idx].next
            if self.head[n] == -1:
                self.tail[n] = -1
        else:
            self.nodes[predecessor_idx].next = self.nodes[node_idx].next
            if self.nodes[node_idx].next == -1:
                self.tail[n] = predecessor_idx

        self.length[n] -= 1
        
        # Return node to free list
        self.nodes[node_idx].next = self.next_free
        self.next_free = node_idx

    cdef void _assign(self, long[:] what, long[:] where, long[:] which):
        """
        Assign values in 'what' to lists 'where' according to indices 'which'.
        """
        cdef long[:] tmp = np.empty_like(what)
        cdef long[:] indptrs = np.zeros(len(where) + 2, dtype=np.int64)
        
        # Count occurrences of each 'which'
        cdef long i
        for i in which:
            indptrs[i+2] += 1
        cdef int total = 0
        for i in range(len(where)):
            total += indptrs[i+2]
            indptrs[i+2] = total

        # Group 'what' by 'which'
        # Afterwards, each element of indptrs is shifted 1 to the left
        for i in range(len(what)):
            tmp[indptrs[which[i]+1]] = what[i]
            indptrs[which[i]+1] += 1
        
        # Sort each group and assign to lists
        cdef long start, end, k
        for i in range(len(where)):
            start = indptrs[i]
            end = indptrs[i+1]
            qsort(&tmp[start], end - start, sizeof(long), compare_long)
            for k in range(start, end):
                self._extend(where[i], tmp[k])
            

            
            

    cdef void _remove_difference(self, long n, long m):
        """Remove intersection of lists n and m from list m"""
        if n == m:
            self._clear_list(n)
            return

        cdef long node_n_idx = self.head[n]
        cdef long node_m_idx = self.head[m]
        cdef long next_m_idx
        cdef long prev_m_idx = -1
        
        while node_m_idx != -1 and node_n_idx != -1:
            if self.nodes[node_n_idx].value < self.nodes[node_m_idx].value:
                node_n_idx = self.nodes[node_n_idx].next
            elif self.nodes[node_n_idx].value > self.nodes[node_m_idx].value:
                prev_m_idx = node_m_idx
                node_m_idx = self.nodes[node_m_idx].next
            else:
                next_m_idx = self.nodes[node_m_idx].next
                self._remove(m, node_m_idx, prev_m_idx)
                node_m_idx = next_m_idx

    cdef void _clear_list(self, long n):
        """Clear list n and return all nodes to the free pool"""
        cdef long node_idx = self.head[n]
        cdef long next_idx
        
        # Traverse list and return each node to free pool
        while node_idx != -1:
            next_idx = self.nodes[node_idx].next
            self.nodes[node_idx].next = self.next_free
            self.next_free = node_idx
            node_idx = next_idx
        
        self.head[n] = -1
        self.tail[n] = -1
        self.length[n] = 0

    cdef void _copy_list(self, long n, long m):
        """Append a copy of list n to list m (does NOT clear m first - matches old behavior)"""
        cdef long node_idx = self.head[n]
        while node_idx != -1:
            self._extend(m, self.nodes[node_idx].value)
            node_idx = self.nodes[node_idx].next

    cpdef long[:] extract(self, long n):
        """Returns list n as an array."""
        if n < 0 or n >= self.n:
            raise ValueError("Index out of bounds.")

        if self.head[n] == -1:
            return np.array([], dtype=np.int64)

        cdef long list_length = self.length[n]
        cdef long[:] result = np.empty(list_length, dtype=np.int64)

        cdef long node_idx = self.head[n]
        cdef long i = 0
        while node_idx != -1:
            result[i] = self.nodes[node_idx].value
            node_idx = self.nodes[node_idx].next
            i += 1

        return result
    
    # Python-accessible wrappers for testing
    def extend(self, long n, long value):
        self._extend(n, value)
    
    def insert(self, long n, long value):
        self._insert(n, value)
    
    def assign(self, long[:] what, long[:] where, long[:] which):
        self._assign(what, where, which)
    
    def remove_difference(self, long n, long m):
        self._remove_difference(n, m)
    
    def clear_list(self, long n):
        self._clear_list(n)
    
    def copy_list(self, long n, long m):
        self._copy_list(n, m)


cdef class HeapNode:
    # cdef public int priority
    # cdef public int index

    def __init__(self, long priority, long index):
        self.priority = priority
        self.index = index

    def __lt__(self, HeapNode other):
        return self.priority < other.priority

    def __int__(self):
        return self.index

cdef class ModHeap:
    """
    Implements a heap with added support for modifying the priority of a key. pop returns the highest-priority node.
    The size of the heap is <= n+k, where n is the number of keys and k is the number of times that a key has had its
    priority modified.
    """
    # cdef public list act_heap
    # cdef long[:] priority
    # cdef public long n

    def __init__(self, long[:] priority):
        cnp.import_array()  # Necessary for initializing the C API
        self.n = len(priority)
        self.priority = np.copy(priority).astype(np.int64) # Copies the input array
        self.act_heap = self._create_heap(np.copy(self.priority))
    cdef list _create_heap(self, long[:] priority):
        cdef long i
        cdef list heap = []
        for i in range(self.n):
            # Allocate a new Node and append it to the heap list
            node = HeapNode(-priority[i], i)
            heap.append(node)
        heapq.heapify(heap)
        return heap

    cpdef void push(self, long index, long priority):
        """
        Either push a new node to the heap, or modify priority of an existing one
        """
        cdef HeapNode node = HeapNode(-priority, index)
        heapq.heappush(self.act_heap, node)
        self.priority[index] = priority

    cpdef long pop(self):
        cdef HeapNode node
        while self.act_heap:
            node = heapq.heappop(self.act_heap)
            if self.priority[node.index] == -node.priority:
                self.priority[node.index] = 0
                return node.index
        return -1
