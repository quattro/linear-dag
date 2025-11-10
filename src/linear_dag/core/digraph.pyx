#digraph.pyx
#define NPY_NO_DEPRECATED_API

from typing import Type
from libc.stdlib cimport free, malloc, realloc
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
cimport numpy as cnp
from .data_structures cimport Stack

cdef int MAXINT = 32767

cdef struct node:
    int index # needed?
    edge* first_in # can be an index?
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
    """
    Unweighted directed graph implemented using linked lists of edges. Data structure is optimized for fast modification,
    i.e., adding or removing nodes and edges. Nodes and edges are stored in arrays. Each node
    stores pointers to its first out-edge and its first in-edge. Each edge (u,v) contains pointers to its next out-edge
    (u,w), its previous out-edge, its next in-edge (w,v), and its previous in-edge; also to u and v themselves. Nodes
    and edges are indexed, with nodes[i].index == i and likewise for edges; order of indices is arbitrary.
    Unused edge/node indices are stored in stacks, and when a new node or edge is added, an element is popped arbitrarily.
    """

    def __cinit__(self, int number_of_nodes, int number_of_edges):
        """
        Initialize a directed graph with specified number of nodes and edges.

        :param number_of_nodes: Total number of nodes in the graph.
        :param number_of_edges: Total number of edges in the graph.
        """
        self.nodes = <node*> malloc(number_of_nodes * sizeof(node))
        self.edge_arrays = <edge**> malloc(64 * sizeof(edge*))
        cdef int i
        for i in range(64):
            self.edge_arrays[i] = NULL
        self.edge_array_length = number_of_edges

    def __init__(self, int number_of_nodes, int number_of_edges):
        if number_of_edges < 0:
            raise ValueError("Number of edges must be non-negative")

        if not self.nodes or not self.edge_arrays:
            raise MemoryError("Could not allocate memory for DiGraph.")

        cdef long i
        for i in range(number_of_nodes):
            self.nodes[i].index = i + 1
            self.nodes[i].first_in = <edge*> &self.nodes[((i - 1) + number_of_nodes) % number_of_nodes]
            self.nodes[i].first_out = <edge*> &self.nodes[(i + 1) % number_of_nodes]
        self.available_node = &self.nodes[0]

        self.maximum_number_of_edges = 0
        self.number_of_available_edges = 0
        self.extend_edge_array()
        assert self.maximum_number_of_edges == number_of_edges

        self.maximum_number_of_nodes = number_of_nodes
        self.number_of_available_nodes = number_of_nodes

    def __dealloc__(self):
        cdef long i
        for i in range(64):
            if self.edge_arrays[i] is not NULL:
                free(self.edge_arrays[i])
        free(self.edge_arrays)
        free(self.nodes)

    cdef edge* get_edge(self, long edge_idx):
        if edge_idx >= self.maximum_number_of_edges:
            raise IndexError("Edge index out of bounds")
        cdef long which_array = 0
        cdef long cum_size = self.edge_array_length
        while edge_idx >= cum_size:
            which_array += 1
            cum_size += self.edge_array_length * (1 << which_array)
        assert which_array < 64
        cdef int arr_idx = edge_idx - (cum_size - self.edge_array_length * (1 << which_array))
        cdef edge* edge = &self.edge_arrays[which_array][arr_idx]
        assert edge.index == edge_idx
        return edge

    @property
    def number_of_nodes(self) -> int:
        return self.maximum_number_of_nodes - self.number_of_available_nodes
    
    @property
    def number_of_edges(self) -> int:
        return self.maximum_number_of_edges - self.number_of_available_edges
    
    @property
    def max_nodes(self) -> int:
        return self.maximum_number_of_nodes
    
    @property
    def max_edges(self) -> int:
        return self.maximum_number_of_edges

    def parents(self, node_index: int) -> int:
        if not self.is_node(node_index):
            raise ValueError("Node index is not a node")
        cdef edge* e = self.nodes[node_index].first_in
        while e is not NULL:
            yield e.u.index
            e = e.next_in

    def children(self, node_index: int) -> int:
        cdef edge* e = self.nodes[node_index].first_out
        while e is not NULL:
            yield e.v.index
            e = e.next_out

    cdef bint is_node(self, long node_index):
        """Check if a node is initialized by checking if its index matches the array position"""
        if node_index >= self.maximum_number_of_nodes or node_index < 0:
            return 0
        return self.nodes[node_index].index == node_index

    cpdef bint has_node(self, long node_index):
        if node_index >= self.maximum_number_of_nodes or node_index < 0:
            return 0
        return self.is_node(node_index)

    cpdef long some_parent(self, long node_index):
        return self.nodes[node_index].first_in.u.index

    cpdef long some_child(self, long node_index):
        return self.nodes[node_index].first_out.v.index

    cdef node* add_node(self, long node_index):
        """Adds a new node with the specified index, which should be smaller than maximum_number_of_nodes.
        If node_index is -1, a new node is created with an arbitrary index, even if must be greater than
        maximum_number_of_nodes."""
        if node_index >= self.maximum_number_of_nodes or node_index < -1:
            raise IndexError("Node index out of bounds")
            
        if self.has_node(node_index):
            raise ValueError("Attempted to add a node that already exists")

        if node_index == -1:
            if self.number_of_available_nodes == 0:
                self.extend_node_array(self.maximum_number_of_nodes * 2)
            node_index = self.available_node - self.nodes
        
        if self.number_of_available_nodes == 1:
            assert (self.available_node - self.nodes) == node_index
            assert <node*> self.available_node.first_out == self.available_node
            self.available_node = NULL
        elif node_index == (self.available_node - self.nodes):
            self.available_node = <node*> self.available_node.first_out
            
        cdef node* new_node = &self.nodes[node_index]
        cdef node* previous_node = <node*> new_node.first_in
        cdef node* next_node = <node*> new_node.first_out

        assert previous_node is not NULL
        assert <node*> previous_node.first_out == new_node
        previous_node.first_out = <edge*> next_node

        assert next_node is not NULL
        assert <node*> next_node.first_in == new_node
        next_node.first_in = <edge*> previous_node
        
        new_node.first_in = NULL
        new_node.first_out = NULL
        new_node.index = node_index

        self.number_of_available_nodes -= 1

        return new_node
    
    def create_node(self, int node_index) -> int:
        cdef node* u = self.add_node(node_index)
        return u.index

    cdef void remove_node(self, node* u):
        if not self.is_node(u.index):
            raise ValueError("Tried to remove a node which does not exist")
        
        cdef long array_position = (u - self.nodes)
        
        while u.first_out is not NULL:
            self.remove_edge(u.first_out)
        while u.first_in is not NULL:
            self.remove_edge(u.first_in)
        
        u.index = array_position + 1
        self.number_of_available_nodes += 1

        cdef node* available_node = self.available_node
        if available_node is NULL:
            self.available_node = u
            u.first_out = <edge*> u
            u.first_in = <edge*> u
            return

        cdef node* next_node = <node*> available_node.first_out
        assert next_node is not NULL

        available_node.first_out = <edge*> u
        u.first_in = <edge*> available_node

        assert <node*> next_node.first_in == available_node
        next_node.first_in = <edge*> u
        u.first_out = <edge*> next_node

    
    def delete_node(self, int node_index):
        self.remove_node(&self.nodes[node_index])

    cdef edge* add_edge(self, long u_index, long v_index):
        if self.number_of_available_edges == 0:
            self.extend_edge_array()
        
        if u_index == v_index:
            raise ValueError("Self edges are not supported")

        if not self.is_node(u_index):
            self.add_node(u_index)
        if not self.is_node(v_index):
            self.add_node(v_index)

        self.number_of_available_edges -= 1
        cdef edge* new_edge = self.available_edge
        self.available_edge = self.available_edge.next_in
        if self.available_edge == new_edge:
            assert self.number_of_available_edges == 0

        self.set_edge_parent(new_edge, &self.nodes[u_index])
        self.set_edge_child(new_edge, &self.nodes[v_index])

        return new_edge

    def create_edge(self, long u_index, long v_index):
        self.add_edge(u_index, v_index)
    
    def has_edge_slow(self, long u_index, long v_index):
        cdef int i
        cdef edge* e
        for i in range(self.maximum_number_of_edges):
            e = self.get_edge(i)
            if e.u is not NULL and e.v is not NULL:
                if e.u.index == u_index and e.v.index == v_index:
                    return True
        return False

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

    cdef void remove_edge(self, edge* e):
        self.patch_out_pointers(e)
        self.patch_in_pointers(e)
        e.u = NULL
        e.v = NULL
        e.next_in = self.available_edge
        self.available_edge = e
        self.number_of_available_edges += 1
    
    def delete_edge(self, int u_idx, int v_idx):
        cdef node* u = &self.nodes[u_idx]
        cdef edge* e = u.first_out
        while e is not NULL:
            if e.v.index == v_idx:
                self.remove_edge(e)
                return
            e = e.next_out
        raise ValueError(f"Edge {u_idx} -> {v_idx} does not exist")

    def copy(self) -> Type[DiGraph]:
        cdef DiGraph duplicate = DiGraph(self.maximum_number_of_nodes, self.maximum_number_of_edges)
        duplicate.copy_from(self)
        return duplicate

    cpdef void copy_from(self, DiGraph other):
        """Copy all edges from another DiGraph into this one."""
        cdef long i
        cdef edge* e
        for i in range(other.maximum_number_of_edges):
            e = other.get_edge(i)
            if e.u is not NULL and e.v is not NULL:
                self.add_edge(e.u.index, e.v.index)

    def tree_descendants(self, int node_index) -> int:
        """For a tree, yields descendants of a node in DFS order. For a DAG, descendants are yielded
        multiple times (however many paths there are to that node). If node_index is contained in a directed cycle,
        method never terminates."""
        cdef node* u = &self.nodes[node_index]
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
        cdef long node_idx, neighbor_idx
        cdef edge* e

        # Ensure nodes needed for now are initialized
        cdef int i
        cdef node* new_node
        for i in range(num_nodes):
            new_node = G.add_node(-1)
            assert new_node.index == i

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
            e = self.get_edge(i)
            if e.u is NULL:
                continue
            assert e.v is not NULL
            edgelist.append((e.u.index, e.v.index))
        return edgelist

    def maximum_node_index(self) -> int:
        cdef long i
        cdef node* u
        cdef long max_index = -1
        for i in range(self.maximum_number_of_nodes):
            u = &self.nodes[i]
            if not self.is_node(i):
                continue
            if u.index > max_index:
                max_index = u.index
        return max_index

    def to_csr(self) -> csr_matrix:
        n = self.maximum_node_index() + 1
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
        while self.number_of_available_nodes > 0:
            self.add_node(-1)

    # TODO - nontrivial because all edges need to be redirected
    cdef void extend_node_array(self, int new_maximum_number_of_nodes):
        raise NotImplementedError

    cdef void extend_edge_array(self):
        cdef long which_arr
        cdef long array_len = self.edge_array_length
        for which_arr in range(64):
            if self.edge_arrays[which_arr] is NULL:
                break
            array_len *= 2
        assert which_arr < 64 # out of memory way before this
        self.edge_arrays[which_arr] = <edge*> malloc(array_len * sizeof(edge))
              
        cdef edge* e
        cdef int i
        cdef long global_edge_idx = self.maximum_number_of_edges
        
        for i in range(array_len):
            e = &self.edge_arrays[which_arr][i]
            e.index = global_edge_idx + i
            e.u = NULL
            e.v = NULL
            e.next_in = &self.edge_arrays[which_arr][(i+1) % array_len]
            e.next_out = NULL
            e.prev_in = NULL
            e.prev_out = NULL
        self.available_edge = e
        self.maximum_number_of_edges += array_len
        self.number_of_available_edges += array_len
        

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
            e = self.get_edge(i)
            if e.u is NULL:
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
            e = self.get_edge(i)
            if e.v is NULL:
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
        cdef Stack nodes_to_visit = Stack(num_nodes)
        cdef long i
        for i in np.where(num_unvisited_children == 0)[0]:
            if self.is_node(i):
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


def create_test_digraph(int num_nodes, int num_edges):
    """
    Python entrypoint to create a DiGraph object for testing.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes to allocate
    num_edges : int
        Number of edges to allocate
        
    Returns
    -------
    DiGraph
        A DiGraph object with allocated space for the specified nodes and edges
    """
    return DiGraph(num_nodes, num_edges)


