#data_structures.pyx
#define NPY_NO_DEPRECATED_API

from typing import Type
import heapq
from libc.stdlib cimport free, malloc, realloc
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
cimport numpy as cnp
from .data_structures cimport node, edge, stack_node, queue_node, list_node # in data_structures.pxd

cdef int MAXINT = 32767

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
    Stack of integers implemented as a linked list.
    """
    # cdef stack_node* head
    # cdef int length

    def __init__(self):
        self.head = NULL
        self.length = 0

    def __dealloc__(self):
        cdef stack_node* element = self.head
        cdef stack_node* next_element
        while element is not NULL:
            next_element = element.next
            free(element)
            element = next_element

    cdef stack_node* push(self, int value):
        cdef stack_node* new_node = <stack_node*> malloc(sizeof(stack_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate memory for a new stack node.")
        new_node.value = value
        new_node.next = self.head
        new_node.prev = NULL
        if self.head is not NULL:
            self.head.prev = new_node
        self.head = new_node
        self.length += 1
        return new_node

    cdef int pop(self):
        cdef stack_node* element = self.head
        cdef int result
        if element is NULL:
            raise ValueError("Attempted to pop from an empty stack")
        self.head = element.next
        if self.head is not NULL:
            self.head.prev = NULL
        result = element.value
        free(element)
        self.length -= 1
        return result

    cdef void remove(self, stack_node* element):
        assert element is not NULL
        if element.prev is NULL:
            self.head = element.next
        else:
            element.prev.next = element.next
        if element.next is not NULL:
            element.next.prev = element.prev
        self.length -= 1
        free(element)

    cdef void clear(self):
        while self.head is not NULL:
            self.pop()

cdef class InfiniteStack(Stack):
    """
    Stack of integers initialized (implicitly) as 0, 1, 2, ...
    """
    # cdef int last
    def __init__(self):
        super().__init__()
        self.last = -1

    cdef int pop(self):
        if self.head is NULL:
            self.last += 1
            return self.last
        return Stack.pop(self)

cdef class Queue:
    """
    Queue of nodes implemented as a doubly linked list. A queue_node has pointers to a next
    queue node and a previous queue node, together with a value which is a DiGraph node*
    """
    # cdef queue_node* head
    # cdef queue_node* tail
    # cdef int length

    def __init__(self):
        self.head = NULL
        self.tail = NULL
        self.length = 0

    def __dealloc__(self):
        cdef queue_node* element = self.head
        cdef queue_node* next_element
        while element is not NULL:
            next_element = element.next
            free(element)
            element = next_element

    cdef queue_node* push(self, node* value):
        cdef queue_node* new_node = <queue_node*> malloc(sizeof(queue_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate memory for a new queue node.")
        new_node.value = value
        new_node.next = NULL
        if self.tail is not NULL:
            self.tail.next = new_node
        else:
            self.head = new_node
        new_node.prev = self.tail
        self.tail = new_node
        self.length += 1
        return new_node

    cdef queue_node* push_to_front(self, node* value):
        cdef queue_node* new_node = <queue_node*> malloc(sizeof(queue_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate memory for a new queue node.")
        new_node.value = value
        new_node.next = NULL
        if self.head is not NULL:
            self.head.prev = new_node
        else:
            self.tail = new_node
        new_node.next = self.head
        self.head = new_node
        self.length += 1
        return new_node

    cdef node* pop(self):
        cdef queue_node* element = self.head
        if element is NULL:
            return <node*> NULL
        self.head = element.next
        if self.head is NULL:
            self.tail = NULL
        else:
            self.head.prev = NULL
        cdef node* result = element.value
        free(element)
        self.length -= 1
        return result

    cdef void clear(self):
        while self.head is not NULL:
            self.pop()

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
    """Array of linked lists of integers."""
    # cdef list_node** head
    # cdef list_node** tail
    # cdef int[:] length
    # cdef int n  # Number of lists

    def __cinit__(self, int n):
        self.n = n
        self.head = <list_node**> malloc(n * sizeof(list_node *))
        self.tail = <list_node**> malloc(n * sizeof(list_node *))
        self.length = np.zeros(n, dtype=np.intc)

    def __init__(self, int n):
        cnp.import_array()  # Necessary for initializing the C API
        if not self.head or not self.tail:
            raise MemoryError("Could not allocate LinkedListArray.")

        cdef long i
        for i in range(n):
            self.head[i] = NULL
            self.tail[i] = NULL
            self.length[i] = 0

    def __dealloc__(self):
        for i in range(self.n):
            current_node = self.head[i]
            while current_node != NULL:
                next_node = current_node.next
                free(current_node)
                current_node = next_node
        free(self.head)
        free(self.tail)

    cdef void extend(self, long n, long value):
        if n < 0 or n >= self.n:
            raise ValueError("LinkedList index out of bounds.")

        new_node = <list_node *> malloc(sizeof(list_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate list_node.")
        new_node.value = value
        new_node.next = NULL

        if self.tail[n] == NULL:
            self.head[n] = new_node
            self.tail[n] = new_node
        else:
            self.tail[n].next = new_node
            self.tail[n] = new_node

        self.length[n] += 1

    cdef void insert(self, long n, long value):
        """Insert a new element into a sorted list"""
        if n < 0 or n >= self.n:
            raise ValueError("LinkedList index out of bounds.")

        new_node = <list_node *> malloc(sizeof(list_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate list_node.")
        new_node.value = value

        cdef list_node* place_in_list = self.head[n]
        cdef list_node* previous = NULL
        while place_in_list is not NULL:
            if value > place_in_list.value:
                break
            previous = place_in_list
            place_in_list = place_in_list.next

        if previous is NULL:
            self.head[n] = new_node
        else:
            previous.next = new_node

        if place_in_list is NULL:
            self.tail[n] = new_node
        new_node.next = place_in_list

        self.length[n] += 1

    cdef void remove(self, long n, list_node * node, list_node * predecessor):
        """Removes a given node from list n; requires knowing its predecessor, if any"""
        # Check if the node to remove is the first node in the list
        if predecessor == NULL:
            assert self.head[n] == node, "If predecessor is NULL, node should be the head of list n"
            self.head[n] = node.next
            if self.head[n] == NULL:
                self.tail[n] = NULL
        else:
            assert predecessor.next == node, "predecessor.next should be the node to remove"
            predecessor.next = node.next
            # Check if the node to remove is the last node in the list
            if node.next == NULL:
                self.tail[n] = predecessor

        self.length[n] -= 1
        free(node)

    cdef void assign(self, long[:] what, long[:] where, long[:] which):
        """
        Assign values in 'what' to lists 'where' according to indices 'which'.
        """
        # Sort 'what' to keep linked lists sorted
        cdef long[:] order = np.argsort(what).astype(np.int64)

        # Iterate over 'which' in sorted order and extend lists
        cdef long i
        for i in range(len(order)):
            self.extend(where[which[order[i]]], what[order[i]])

    cdef void remove_difference(self, long n, long m):
        """Compute the intersection between two lists n and m and remove it from list m, assuming they are sorted."""
        if n == m:
            self.clear_list(n)
            return

        cdef list_node* node_n = self.head[n]
        cdef list_node* node_m = self.head[m]
        cdef list_node* next_m
        cdef list_node* prev_m = NULL
        while node_m != NULL and node_n != NULL:
            if node_n.value < node_m.value:
                node_n = node_n.next
            elif node_n.value > node_m.value:
                prev_m = node_m
                node_m = node_m.next
            else:
                next_m = node_m.next
                self.remove(m, node_m, prev_m)
                node_m = next_m

    cdef void clear_list(self, long n):
        cdef list_node * node = self.head[n]
        cdef list_node * next_node
        while node != NULL:
            next_node = node.next
            free(node)
            node = next_node
        self.head[n] = NULL
        self.tail[n] = NULL
        self.length[n] = 0

    cdef copy_list(self, long n, long m):
        """Replaces the current list m with a copy of list n"""
        cdef list_node * node = self.head[n]
        while node is not NULL:
            self.extend(m, node.value)
            node = node.next

    cpdef long[:] extract(self, long n):
        """
        Returns list n as an array.
        """
        if n < 0 or n >= self.n:
            raise ValueError("Index out of bounds.")

        # Check if the linked list at index n is empty
        if self.head[n] is NULL:
            return np.array([], dtype=np.int64)  # Return an empty NumPy array for an empty list

        # Calculate the length of the n-th linked list
        cdef long list_length = self.length[n]

        # Initialize a NumPy array of the appropriate length
        cdef cnp.ndarray result = np.empty(list_length, dtype=np.int64)

        # Traverse the linked list and fill the NumPy array
        cdef list_node * current_node = self.head[n]
        cdef long i = 0
        while current_node is not NULL:
            result[i] = current_node.value
            current_node = current_node.next
            i += 1

        return result

# cdef struct node:
#     int index
#     edge* first_in
#     edge* first_out
#
# cdef struct edge:
#     int index
#     node* u
#     node* v
#     edge* next_in
#     edge* next_out
#     edge* prev_in
#     edge* prev_out

cdef class DiGraph:
    """
    Unweighted directed graph implemented using linked lists of edges. Data structure is optimized for fast modification,
    i.e., adding or removing nodes and edges. Nodes and edges are stored in arrays. Each node
    stores pointers to its first out-edge and its first in-edge. Each edge (u,v) contains pointers to its next out-edge
    (u,w), its previous out-edge, its next in-edge (w,v), and its previous in-edge; also to u and v themselves. Nodes
    and edges are indexed, with nodes[i].index == i and likewise for edges; order of indices is arbitrary.
    Unused edge/node indices are stored in stacks, and when a new node or edge is added, an element is popped arbitrarily.
    """
    # cdef node**
    # cdef bint* is_node
    # cdef edge** edges
    # cdef Stack available_nodes
    # cdef Stack available_edges
    # cdef int maximum_number_of_nodes
    # cdef int maximum_number_of_edges

    def __cinit__(self, int number_of_nodes, int number_of_edges):
        """
        Initialize a directed graph with specified number of nodes and edges.

        :param number_of_nodes: Total number of nodes in the graph.
        :param number_of_edges: Total number of edges in the graph.
        """
        self.nodes = <node**> malloc(number_of_nodes * sizeof(node *))
        self.is_node = <bint*> malloc(number_of_nodes * sizeof(bint))
        self.edges = <edge**> malloc(number_of_edges * sizeof(edge *))

    def __init__(self, int number_of_nodes, int number_of_edges):

        if not self.nodes or not self.edges:
            raise MemoryError("Could not allocate memory for DiGraph.")

        self.available_nodes = Stack()
        self.available_edges = Stack()

        cdef long i
        for i in reversed(range(number_of_nodes)):
            self.nodes[i] = <node*> self.available_nodes.push(i)
            self.is_node[i] = False
        for i in reversed(range(number_of_edges)):
            self.edges[i] = NULL
            self.available_edges.push(i)
        self.maximum_number_of_nodes = number_of_nodes
        self.maximum_number_of_edges = number_of_edges

    def __dealloc__(self):
        cdef long i
        for i in range(self.maximum_number_of_edges):
            if self.edges[i] is not NULL:
                free(self.edges[i])
        free(self.edges)

        # Before freeing nodes, must clear the available_nodes stack and the pointers to them in self.nodes
        while self.available_nodes.length > 0:
            i = self.available_nodes.pop()
            self.nodes[i] = NULL
        for i in range(self.maximum_number_of_nodes):
            if self.nodes[i] is not NULL:
                free(self.nodes[i])
        free(self.nodes)
        free(self.is_node)

    @property
    def number_of_nodes(self) -> int:
        return self.maximum_number_of_nodes - self.available_nodes.length
    @property
    def number_of_edges(self) -> int:
        return self.maximum_number_of_edges - self.available_edges.length

    def parents(self, node_index: int) -> int:
        cdef edge* e = self.nodes[node_index].first_in
        while e is not NULL:
            yield e.u.index
            e = e.next_in

    def children(self, node_index: int) -> int:
        cdef edge* e = self.nodes[node_index].first_out
        while e is not NULL:
            yield e.v.index
            e = e.next_out

    cpdef bint has_node(self, long node_index):
        if node_index >= self.maximum_number_of_nodes or node_index < 0:
            return 0
        return self.is_node[node_index]

    cpdef long some_parent(self, long node_index):
        return self.nodes[node_index].first_in.u.index

    cpdef long some_child(self, long node_index):
        return self.nodes[node_index].first_out.v.index

    def copy(self) -> Type[DiGraph]:
        cdef DiGraph duplicate = DiGraph(self.maximum_number_of_nodes, self.maximum_number_of_edges)
        duplicate.add_edges_from(self.edge_list())
        return duplicate

    def tree_descendants(self, int node_index) -> int:
        """For a tree, yields descendants of a node in DFS order. For a DAG, descendants are yielded
        multiple times (however many paths there are to that node). If node_index is contained in a directed cycle,
        method never terminates."""
        cdef node* u = self.nodes[node_index]
        if u is NULL:
            raise ValueError(f"Attempted to look up descendants of non-existent node with index {node_index}")
        cdef edge* e = u.first_out
        yield node_index
        while e is not NULL:
            yield from self.tree_descendants(e.v.index)
            e = e.next_out

    @classmethod
    def from_csr(cls, A: csr_matrix):
        """
        Create a DiGraph from a Compressed Sparse Row (CSR) format matrix.
        """
        cdef int[:] indptr = A.indptr.astype(np.intc)
        cdef int[:] indices = A.indices.astype(np.intc)
        cdef int[:] data = A.data.astype(np.intc)
        cdef int max_num_nodes = A.shape[0]
        if A.shape[1] != max_num_nodes:
            raise ValueError("Input matrix should be square")
        cdef int num_edges = np.sum(np.asarray(data) == 1)
        cdef int num_nodes = len(indptr) - 1
        cdef DiGraph G = cls(max_num_nodes, num_edges)
        cdef long node_idx, neighbor_idx, edge_idx
        cdef edge* e

        # Ensure nodes needed for now are initialized
        for i in range(num_nodes):
            index = G.add_node(-1).index
            assert index == i

        for node_idx in range(num_nodes):
            for i in range(indptr[node_idx], indptr[node_idx + 1]):
                if data[i] != 1:
                    continue
                neighbor_idx = indices[i]
                G.add_edge(node_idx, neighbor_idx)

        return G

    @classmethod
    def from_csc(cls, A: csc_matrix):
        """
        Create a DiGraph from a Compressed Sparse Row (CSR) format matrix.
        """
        return cls.from_csr(csr_matrix(A))

    def edge_list(self) -> list[tuple[int, int]]:
        """
        Returns a list of edges.
        """
        edgelist = []
        cdef long i
        cdef edge* e
        for i in range(self.maximum_number_of_edges):
            e = self.edges[i]
            if e is NULL:
                continue
            if e.u is NULL:
                continue
            edgelist.append((e.u.index, e.v.index))

        return edgelist

    def to_csr(self) -> csr_matrix:
        n = self.maximum_number_of_nodes
        edges = self.edge_list()
        rows, cols = zip(*edges)
        return csr_matrix((np.ones(len(edges)), (list(rows), list(cols))),
                       shape=(n, n))

    def to_csc(self) -> csc_matrix:
        return csc_matrix(self.to_csr())

    def add_edges_from(self, list[tuple[int, int]] edges) -> None:
        for i, j in edges:
            self.add_edge(i, j)

    cpdef void initialize_all_nodes(self):
        while self.available_nodes.length > 0:
            self.add_node(-1)

    cdef node* add_node(self, long node_index):
        """Adds a new node with the specified index, which should be smaller than maximum_number_of_nodes.
        If node_index is -1, a new node is created with an arbitrary index, even if must be greater than
        maximum_number_of_nodes."""

        if node_index >= self.maximum_number_of_nodes or node_index < -1:
            raise IndexError("Node index out of bounds")
        cdef stack_node* element
        if node_index == -1:
            if self.available_nodes.length == 0:
                self.extend_node_array(self.maximum_number_of_nodes * 2)
            node_index = self.available_nodes.pop()
        else:
            if self.is_node[node_index]:
                raise ValueError("Attempted to add a node that already exists")
            element = <stack_node*> self.nodes[node_index]
            self.available_nodes.remove(element)

        self.nodes[node_index] = <node*> malloc(sizeof(node))
        if self.nodes[node_index] is NULL:
            raise MemoryError("Could not allocate memory for a new node")
        self.is_node[node_index] = True
        cdef node* new_node = self.nodes[node_index]

        new_node.first_in = NULL
        new_node.first_out = NULL
        new_node.index = node_index
        return new_node

    cdef edge* add_edge(self, long u_index, long v_index):
        if self.available_edges.length == 0:
            self.extend_edge_array(self.maximum_number_of_edges * 2)
        if u_index == v_index:
            raise ValueError("Self edges are not supported")

        if max(u_index, v_index) >= self.maximum_number_of_nodes:
            self.extend_node_array(2 * max(u_index, v_index))

        if not self.is_node[u_index]:
            self.add_node(u_index)
        if not self.is_node[v_index]:
            self.add_node(v_index)

        edge_index = self.available_edges.pop()
        if self.edges[edge_index] is NULL:
            self.edges[edge_index]  = <edge*> malloc(sizeof(edge))
            if self.edges[edge_index] is NULL:
                raise MemoryError("Could not allocate memory for a new edge")
        cdef edge* new_edge = self.edges[edge_index]
        new_edge.u = NULL
        new_edge.v = NULL
        new_edge.index = edge_index
        self.set_edge_parent(new_edge, self.nodes[u_index])
        self.set_edge_child(new_edge, self.nodes[v_index])

        return new_edge

    cdef void extend_node_array(self, int new_maximum_number_of_nodes):
        if new_maximum_number_of_nodes <= self.maximum_number_of_nodes:
            raise ValueError("New maximum number of nodes must be greater than the current maximum.")

        self.nodes = <node**> realloc(self.nodes, new_maximum_number_of_nodes * sizeof(node *))
        if self.nodes == NULL:
            raise MemoryError("Could not reallocate memory for nodes array.")

        self.is_node = <bint *> realloc(self.is_node, new_maximum_number_of_nodes * sizeof(bint))
        if self.is_node == NULL:
            raise MemoryError("Could not reallocate memory for is_node array.")

        for i in reversed(range(self.maximum_number_of_nodes, new_maximum_number_of_nodes)):
            self.is_node[i] = False
            self.nodes[i] = <node*> self.available_nodes.push(i)

        self.maximum_number_of_nodes = new_maximum_number_of_nodes

    cdef void extend_edge_array(self, int new_maximum_number_of_edges):
        if new_maximum_number_of_edges <= self.maximum_number_of_edges:
            raise ValueError("New maximum number of edges must be greater than the current maximum.")

        self.edges = <edge**> realloc(self.edges, new_maximum_number_of_edges * sizeof(edge *))
        if self.edges == NULL:
            raise MemoryError("Could not reallocate memory for edges array.")

        for i in range(self.maximum_number_of_edges, new_maximum_number_of_edges):
            self.edges[i] = NULL
            self.available_edges.push(i)

        self.maximum_number_of_edges = new_maximum_number_of_edges

    cdef void set_edge_child(self, edge* e, node* v):
        self.patch_in_pointers(e)
        e.v = v
        e.prev_in = NULL
        e.next_in = v.first_in
        if v.first_in is not NULL:
            v.first_in.prev_in = e
        v.first_in = e

    cdef void set_edge_parent(self, edge* e, node* u):
        self.patch_out_pointers(e)
        e.u = u
        e.prev_out = NULL
        e.next_out = u.first_out
        if u.first_out is not NULL:
            u.first_out.prev_out = e
        u.first_out = e

    cdef void remove_edge(self, edge* e):
        self.patch_out_pointers(e)
        self.patch_in_pointers(e)
        self.available_edges.push(e.index)
        self.edges[e.index] = NULL
        free(e)

    cdef void remove_node(self, node* u):
        if not self.is_node[u.index]:
            raise ValueError("Tried to remove a node which does not exist")
        while u.first_out is not NULL:
            self.remove_edge(u.first_out)
        while u.first_in is not NULL:
            self.remove_edge(u.first_in)
        self.nodes[u.index] = <node*> self.available_nodes.push(u.index)
        self.is_node[u.index] = False
        free(u)

    cdef void replace_node(self, node* u, node* v):
        """Replaces every edge (u,w) or (w,u) with (v,w) or (w,v)"""
        cdef edge* e = u.first_out
        cdef edge* next_edge
        while e is not NULL:
            next_edge = e.next_out
            self.set_edge_parent(e, v)
            e = next_edge

        e = u.first_in
        while e is not NULL:
            next_edge = e.next_in
            self.set_edge_child(e, v)
            e = next_edge

    cdef void patch_out_pointers(self, edge * e):
        """
        Skip over an edge in its out-edge pointer list
        :param e:
        :return:
        """
        if e.u is NULL:
            return

        if e.prev_out is not NULL:
            e.prev_out.next_out = e.next_out
        else:
            e.u.first_out = e.next_out

        if e.next_out is not NULL:
            e.next_out.prev_out = e.prev_out

    cdef void patch_in_pointers(self, edge * e):
        """
        Skip over an edge in its in-edge pointer list
        :param e:
        :return:
        """
        if e.v is NULL:
            return

        if e.prev_in is not NULL:
            e.prev_in.next_in = e.next_in
        else:
            e.v.first_in = e.next_in

        if e.next_in is not NULL:
            e.next_in.prev_in = e.prev_in

    cdef void collapse_node(self, node* v):
        """Removes a node v and preserves paths passing through it"""
        cdef edge * in_edge = v.first_in
        cdef edge * out_edge = v.first_out
        while in_edge is not NULL:
            out_edge = v.first_out
            while out_edge is not NULL:
                self.add_edge(in_edge.u.index, out_edge.v.index)
                out_edge = out_edge.next_out
            in_edge = in_edge.next_in

        self.remove_node(v)

    cdef void collapse_node_with_indegree_one(self, node * v):
        """Removes a node v and preserves paths from its first parent to all its children;
        other parents are ignored"""
        cdef edge * removable_edge = v.first_in
        cdef edge * out_edge
        cdef edge * next_edge
        cdef node * u

        if removable_edge is NULL:
            raise ValueError("Node does not have an in-edge.")

        u = removable_edge.u
        assert u is not NULL

        # Redirect each out-edge of v to point to u
        out_edge = v.first_out
        while out_edge is not NULL:
            next_edge = out_edge.next_out
            self.set_edge_parent(out_edge, u)
            out_edge = next_edge

        self.remove_node(v)

    cdef void collapse_node_with_outdegree_one(self, node * u):
        """Removes a node u and preserves paths from its all its parents to its first child;
        all other children are ignored. """
        cdef edge * removable_edge = u.first_out
        cdef edge * in_edge
        cdef edge * next_edge
        cdef edge * prev_edge
        cdef node * v

        if removable_edge is NULL:
            raise ValueError("Node does not have an out-edge.")

        v = removable_edge.v
        assert v is not NULL

        # Insert the first in-edge of u into the in-edge list of v, replacing removable_edge
        in_edge = u.first_in
        prev_edge = removable_edge.prev_in
        in_edge.prev_in = prev_edge
        if prev_edge is not NULL:
            prev_edge.next_in = in_edge
        else:
            v.first_in = in_edge

        # Redirect each in-edge of u to point to v; find last such edge
        while in_edge is not NULL:
            in_edge.v = v
            prev_edge = in_edge
            in_edge = in_edge.next_in

        # Patch the end of the inserted list
        next_edge = removable_edge.next_in
        prev_edge.next_in = next_edge
        if next_edge is not NULL:
            next_edge.prev_in = prev_edge

        # Ensure that the redirected edges are not removed when removing u
        u.first_in = NULL
        self.remove_node(u)


    cdef long number_of_successors(self, node* u):
        """
        Counts successors of a node by iterating over its out-edges.
        """
        cdef int counter = 0
        cdef edge* out_edge = u.first_out
        while out_edge is not NULL:
            out_edge = out_edge.next_out
            counter += 1
        return counter

    cdef long number_of_predecessors(self, node* u):
        """
        Counts predecessors of a node by iterating over its in-edges.
        """
        cdef int counter = 0
        cdef edge* in_edge = u.first_in
        while in_edge is not NULL:
            in_edge = in_edge.next_in
            counter += 1
        return counter

    def successors(self, u_idx: int):
        """
        Iterate over successors of a node
        """
        cdef edge* e = self.nodes[u_idx].first_out
        while e is not NULL:
            yield e.v.index
            e = e.next_out
    def predecessors(self, v_idx: int):
        """
        Iterate over predecessors of a node
        """
        cdef edge * e = self.nodes[v_idx].first_in
        while e is not NULL:
            yield e.u.index
            e = e.next_in

    cpdef long[:] out_degree(self):
        cdef long[:] result = np.zeros(self.maximum_number_of_nodes, dtype=np.int64)
        cdef node* u
        cdef edge* e
        cdef long i
        for i in range(self.maximum_number_of_edges):
            e = self.edges[i]
            if e is NULL:
                continue
            u = e.u
            result[u.index] += 1
        return result

    cpdef long[:] in_degree(self):
        cdef long[:] result = np.zeros(self.maximum_number_of_nodes, dtype=np.int64)
        cdef node* v
        cdef edge* e
        cdef long i
        for i in range(self.maximum_number_of_edges):
            e = self.edges[i]
            if e is NULL:
                continue
            v = e.v
            result[v.index] += 1
        return result

    cpdef long[:] reverse_topological_sort(self):
        """
        Returns an array of nodes, L, in order such that if L[i] is a descendant of L[j] then i < j
        """
        cdef long num_nodes = self.number_of_nodes
        cdef cnp.ndarray num_unvisited_children = np.asarray(self.out_degree())

        # nodes_to_visit initialized with nodes having out-degree 0
        cdef Stack nodes_to_visit = Stack()
        cdef long i
        for i in np.where(num_unvisited_children == 0)[0]:
            if self.is_node[i]:
                nodes_to_visit.push(i)

        cdef long[:] result = np.empty(num_nodes, dtype=np.int64)
        cdef long node_idx, parent_index
        cdef edge* in_edge
        i = 0
        while nodes_to_visit.length > 0:
            node_idx = nodes_to_visit.pop()
            result[i] = node_idx
            i += 1

            # Add parents to nodes_to_visit once all of their other children have been visited
            in_edge = self.nodes[node_idx].first_in
            while in_edge is not NULL:
                parent_index = in_edge.u.index
                num_unvisited_children[parent_index] -= 1
                if num_unvisited_children[parent_index] == 0:
                    nodes_to_visit.push(parent_index)
                in_edge = in_edge.next_in

        if i != num_nodes:
            raise ValueError("DiGraph is not acyclic")

        return result


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
