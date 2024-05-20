from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as cnp
cnp.import_array()  # Necessary for initializing the C API


cdef struct stack_node:
    edge* value
    stack_node* next

cdef class Stack:
    """
    Stack of edges implemented as a linked list.
    """
    cdef stack_node* head
    cdef int length

    def __cinit__(self):
        self.head = NULL
        self.length = 0

    def __dealloc__(self):
        cdef stack_node* element = self.head
        cdef stack_node* next_element
        while element is not NULL:
            next_element = element.next
            free(element)
            element = next_element

    cdef void push(self, edge* value):
        cdef stack_node* new_node = <stack_node*> malloc(sizeof(stack_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate memory for a new stack node.")
        new_node.value = value
        new_node.next = self.head
        self.head = new_node
        self.length += 1

    cdef edge* pop(self):
        cdef stack_node* element = self.head
        cdef edge* result
        if element is NULL:
            return <edge*> NULL
        self.head = element.next
        result = element.value
        free(element)
        self.length -= 1
        return result

    cdef void clear(self):
        while self.head is not NULL:
            self.pop()

cdef struct queue_node:
    node* value
    queue_node* next

cdef class Queue:
    """
    Queue of nodes implemented as a linked list.
    """
    cdef queue_node* head
    cdef queue_node* tail
    cdef int length

    def __cinit__(self):
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

    cdef void push(self, node* value):
        cdef queue_node* new_node = <queue_node*> malloc(sizeof(queue_node))
        if new_node is NULL:
            raise MemoryError("Could not allocate memory for a new queue node.")
        new_node.value = value
        new_node.next = NULL
        self.tail.next = new_node
        self.tail = new_node
        self.length += 1

    cdef node* pop(self):
        cdef queue_node* element = self.head
        cdef node* result
        if element is NULL:
            return <node*> NULL
        self.head = element.next
        result = element.value
        free(element)
        self.length -= 1
        return result

    cdef void clear(self):
        while self.head is not NULL:
            self.pop()

cdef struct node:
    int index
    edge* first_in
    edge* first_out
    edge* last_in
    edge* last_out

cdef struct edge:
    node* u
    node* v
    edge* next_in
    edge* next_out
    edge* prev_in
    edge* prev_out

cdef class DiGraph:
    """
    Unweighted directed graph implemented as arrays of nodes and edges, with linked lists connecting in- and out-
    edges of each node. For in-edges, these linked lists are sorted by their origin node. Implements path-preserving
    operations.
    """
    cdef node** nodes
    cdef edge** edges
    cdef int number_of_nodes
    cdef int number_of_edges
    cdef int maximum_node_index

    def __cinit__(self, int number_of_nodes, int number_of_edges):
        """
        Initialize a directed graph with specified number of nodes and edges.

        :param number_of_nodes: Total number of nodes in the graph.
        :param number_of_edges: Total number of edges in the graph.
        """
        self.nodes = <node**> malloc(number_of_nodes * sizeof(node *))
        if not self.nodes:
            raise MemoryError("Could not allocate memory for DiGraph.")
        self.edges = <edge**> malloc(number_of_edges * sizeof(edge *))
        if not self.edges:
            raise MemoryError("Could not allocate memory for DiGraph.")
        self.maximum_node_index = -1
        self.number_of_edges = number_of_edges
        self.number_of_nodes = number_of_nodes

    def __dealloc__(self):
        cdef int u
        for i in range(self.number_of_nodes):
            if self.nodes[i] is not NULL:
                free(self.nodes[i])
        for i in range(self.number_of_edges):
            if self.edges[i] is not NULL:
                free(self.edges[i])
        free(self.nodes)
        free(self.edges)

    @classmethod
    def from_csr(cls, int[:] indptr, int[:] indices, int[:] data, int max_num_nodes):
        """
        Create a graph from a Compressed Sparse Row (CSR) format.

        :param indptr: Array that indicates the start of each row in indices
        :param indices: Array containing column indices of non-zero elements in matrix form.
        :param data: Array containing weights of the edges corresponding to column indices. Edges of weight different
        from one are ignored.
        :param max_num_nodes: Number of nodes in the graph.
        :return: An instance of DiGraph constructed from the given CSR data.
        """
        cdef int num_edges = len(data)
        cdef int num_nodes = len(indptr) - 1
        cdef DiGraph G = cls(max_num_nodes, num_edges)
        cdef int node_idx, neighbor_idx, edge_idx = 0
        cdef edge* e
        G.maximum_node_index = num_nodes - 1

        # Ensure nodes needed for now are initialized
        for i in range(num_nodes):
            G.nodes[i] = <node*> malloc(sizeof(node))
            if G.nodes[i] is NULL:
                raise MemoryError("Failed to allocate node")
            G.nodes[i].first_in = NULL
            G.nodes[i].first_out = NULL
            G.nodes[i].last_in = NULL
            G.nodes[i].last_out = NULL
            G.nodes[i].index = i

        # Ensure edges are initialized
        for i in range(num_edges):
            G.edges[i] = <edge*> malloc(sizeof(edge))
            if G.edges[i] is NULL:
                raise MemoryError("Failed to allocate edge")

        for node_idx in range(num_nodes):
            for neighbor_idx in range(indptr[node_idx], indptr[node_idx + 1]):
                if edge_idx >= num_edges:
                    raise IndexError("Edge index out of range")
                if data[neighbor_idx] != 1:
                    continue

                e = G.edges[edge_idx]
                e.u = G.nodes[node_idx]
                if indices[neighbor_idx] >= num_nodes:
                    raise IndexError("Node index out of range")
                e.v = G.nodes[indices[neighbor_idx]]

                # Update next_out/last_out of the origin node
                if e.u.first_out is NULL:
                    e.u.first_out = e
                    e.prev_out = NULL
                else:
                    e.u.last_out.next_out = e
                    e.prev_out = e.u.last_out

                e.u.last_out = e
                e.next_out = NULL

                # Update next_in/last_in of the destination node
                if e.v.first_in is NULL:
                    e.v.first_in = e
                    e.prev_in = NULL
                else:
                    e.v.last_in.next_in = e
                    e.prev_in = e.v.last_in

                e.v.last_in = e
                e.next_in = NULL

                edge_idx += 1

        return G

    cdef node* add_node(self):  # TODO recycle indices
        self.maximum_node_index += 1
        if self.maximum_node_index >= self.number_of_nodes:
            raise IndexError("Attempted to add too many nodes")
        cdef node* new_node
        if self.nodes[self.maximum_node_index] is NULL:
            self.nodes[self.maximum_node_index] = <node*> malloc(sizeof(node))
        new_node = self.nodes[self.maximum_node_index]

        new_node.first_in = NULL
        new_node.first_out = NULL
        new_node.last_in = NULL
        new_node.last_out = NULL
        new_node.index = self.maximum_node_index
        return new_node

    cdef edge* add_edge(self, node* u, node* v, int index):  # TODO put it into array
        cdef edge* new_edge
        if self.edges[index] is NULL:
            self.edges[index] = <edge*> malloc(sizeof(edge))
        new_edge = self.edges[index]

        new_edge.u = u
        new_edge.next_out = NULL
        new_edge.prev_out = u.last_out
        u.last_out.next_out = new_edge
        u.last_out = new_edge

        new_edge.v = v
        new_edge.next_in = NULL
        new_edge.prev_in = v.last_in
        v.last_in.next_in = new_edge
        v.last_in = new_edge

        return new_edge

    cdef void remove_edge(self, edge* e):
        e.u = NULL
        e.v = NULL
        e.prev_in = NULL
        e.prev_out = NULL
        e.next_in = NULL
        e.next_out = NULL

    cdef void remove_node(self, node* u):
        u.first_in = NULL
        u.first_out = NULL
        u.last_in = NULL
        u.last_out = NULL
        u.index = -1
        if self.nodes[self.maximum_node_index] == u:
            self.maximum_node_index -= 1

    cdef void patch_out_pointers(self, edge * e):
        """
        Skip over an edge in its out-edge pointer list
        :param e:
        :return:
        """
        if e.prev_out is not NULL:
            e.prev_out.next_out = e.next_out
        else:
            e.u.first_out = e.next_out

        if e.next_out is not NULL:
            e.next_out.prev_out = e.prev_out
        else:
            e.u.last_out = e.prev_out

    cdef void patch_in_pointers(self, edge * e):
        """
        Skip over an edge in its in-edge pointer list
        :param e:
        :return:
        """
        if e.prev_in is not NULL:
            e.prev_in.next_in = e.next_in
        else:
            e.v.first_in = e.next_in

        if e.next_in is not NULL:
            e.next_in.prev_in = e.prev_in
        else:
            e.v.last_in = e.prev_in


    cdef void remove_node_with_indegree_one(self, node* v):
        cdef edge * removable_edge = v.first_in
        cdef edge * out_edge
        cdef edge * prev_edge
        cdef node * u

        # Check that v has exactly one in-edge
        if removable_edge is NULL or removable_edge.next_in is not NULL:
            raise ValueError("Node does not have in-degree of exactly one.")

        u = removable_edge.u

        # Redirect each out-edge of v to originate from u
        out_edge = v.first_out
        prev_edge = removable_edge.prev_out
        while out_edge is not NULL:
            out_edge.u = u
            if prev_edge is NULL:
                u.first_out = out_edge
            else:
                prev_edge.next_out = out_edge
            out_edge.prev_out = prev_edge
            out_edge = out_edge.next_out

        # skip over removable_edge in the out-edge list of u
        self.patch_out_pointers(removable_edge)

        self.remove_edge(removable_edge)
        self.remove_node(v)

    cdef void remove_node_with_outdegree_one(self, node* u):
        cdef edge * removable_edge = u.first_out
        cdef edge * in_edge
        cdef edge * prev_edge
        cdef node * v

        # Check that v has exactly one out-edge
        if removable_edge is NULL or removable_edge.next_out is not NULL:
            raise ValueError("Node does not have out-degree of exactly one.")

        v = removable_edge.v

        # Redirect each in-edge of u to point to v
        in_edge = u.first_in
        prev_edge = removable_edge.prev_in
        while in_edge is not NULL:
            in_edge.v = v
            if prev_edge is NULL:
                v.first_in = in_edge
            else:
                prev_edge.next_in = in_edge
            in_edge.prev_in = prev_edge
            in_edge = in_edge.next_in

        # skip over removable_edge in the in-edge list of v
        self.patch_in_pointers(removable_edge)

        self.remove_edge(removable_edge)
        self.remove_node(u)

    cdef int number_of_successors(self, node* u):
        """
        Counts successors of a node by iterating over its out-edges.
        """
        cdef int counter = 0
        cdef edge* out_edge = u.first_out
        while out_edge is not NULL:
            out_edge = out_edge.next_out
            counter += 1
        return counter

    cdef int number_of_predecessors(self, node* u):
        """
        Counts predecessors of a node by iterating over its in-edges.
        """
        cdef int counter = 0
        cdef edge* in_edge = u.first_in
        while in_edge is not NULL:
            in_edge = in_edge.next_in
            counter += 1
        return counter

    cpdef recombine_all(self, int threshold):
        """
        Passes through nodes in the graph and calls recombine_node.
        :param threshold:
        :return:
        """
        cdef node* u
        cdef int i = -1
        for u in self.nodes[:self.maximum_node_index+1]:
            i += 1
            if u is NULL:
                continue
            if self.number_of_successors(u) < threshold:
                continue
            self.recombine_node(u, threshold=threshold)

    cdef void recombine_node(self, node* u, int threshold):
        """
        Processes outgoing edges from node u in DiGraph G:
        - Finds unique last-visited predecessors of successors of u.
        - Factor bicliques sharing more than n children, where n is the threshold parameter.
        - Removes predecessors if they have an out-degree of 1 (and are not in the original set of nodes), and removes
        children if they have an in-degree of 1 (and are not in the original set of nodes)
        """
        # Group successors by their last predecessor
        num_successors = self.number_of_successors(u)
        cdef cnp.ndarray[int, ndim=1] predecessors = np.empty(num_successors, dtype=np.intc)
        cdef edge** out_edges = <edge**> malloc(num_successors * sizeof(edge*))
        cdef edge* out_edge = u.first_out
        cdef int neighbor_index
        for neighbor_index in range(num_successors):
            out_edges[neighbor_index] = out_edge
            if out_edge.prev_in is not NULL:
                predecessors[neighbor_index] = out_edge.prev_in.v.index
            else:
                predecessors[neighbor_index] = -1
            neighbor_index += 1
            out_edge = out_edge.next_out

        # Recombine u with each last predecessor
        cdef long[:] sorting_indices = np.argsort(predecessors)
        cdef int predecessor = predecessors[sorting_indices[0]]
        cdef Stack shared_neighbor_edges = Stack()
        cdef long i
        for i in sorting_indices:
            if predecessors[i] == -1:
                continue

            if predecessors[i] != predecessor:
                self.recombine_pair(shared_neighbor_edges, threshold)
                # recombine_pair also empties the stack
                predecessor = predecessors[i]

            shared_neighbor_edges.push(out_edges[i])

        self.recombine_pair(shared_neighbor_edges, threshold)
        free(out_edges)

    cdef node* recombine_pair(self, Stack edges_vw, int threshold):
        """
        Recombines nodes u and v with shared successors w for (v,w) in edges_vw. For all such w,
        (v,w).prev_in should be (u,w). Creates a new recombination node new, redirects edges (u,w) to (new,w),
        and replaces the first two edges (v,w) with (u,new) and (v,new). (shared_children should have length >=2). Only
        creates the recombination if the number of shared children are g.t. the threshold; otherwise, empties the stack
        of edges and returns NULL.
        :param edges_vw: stack of edges from v to the shared successors of u and v, which is emptied
        :param threshold (>=1): number of shared children must be greater than threshold to create recombination.
        :return: recombination node that was created.
        """
        if edges_vw.length <= threshold:
            edges_vw.clear()
            return <node *> NULL
        cdef node* w
        cdef edge* e = edges_vw.pop()
        cdef node * v = e.u
        if e.prev_in is NULL:
            edges_vw.clear()
            return <node*> NULL
        print(edges_vw.length)
        cdef node * u = e.prev_in.u
        cdef node* new = self.add_node()

        new.first_in = e.prev_in # remember where to put the new edge (u, new)

        while e is not NULL:
            f = e.prev_in
            assert f.u == u

            w = e.v
            assert f.v == w

            self.patch_in_pointers(f)
            self.patch_out_pointers(f)
            self.patch_out_pointers(e)
            # in-pointer lists for e remain intact

            # reassign (u,w) to (new,w)
            e.u = new

            # delete (u,w)
            self.remove_edge(f)
            e = edges_vw.pop()

        # place for (v, new)
        new.last_in = f

        # Create edges (u,new) and (v,new) at the expected locations
        new.first_in.v = new # new.first_in is (u,new)
        new.last_in.v = new # new.last_in is (v,new)
        new.first_in.u = u
        new.last_in.u = v
        new.first_in.prev_in = NULL
        new.first_in.next_in = new.last_in
        new.last_in.prev_in = new.first_in
        new.last_in.next_in = NULL

        # append (u,new) to the end of the out-edge list for u, which does not remain sorted
        if u.last_out is not NULL:
            u.last_out.next_out = new.first_in
        else:
            u.first_out = new.first_in
        new.first_in.prev_out = u.last_out
        u.last_out = new.first_in

        # append (v,new) to the end of the out-edge list for v
        if v.last_out is not NULL:
            v.last_out.next_out = new.last_in
        else:
            v.first_out = new.last_in
        new.last_in.prev_out = v.last_out
        v.last_out = new.last_in

        return new

    def test_recombine(self, int u, int v, int[:] edges):
        cdef Stack edge_stack = Stack()
        for i in edges:
            edge_stack.push(self.edges[i])
        self.recombine_pair(edge_stack, 1)

    def test_remove(self, int u):
        if self.nodes[u].first_in is not NULL:
            if self.nodes[u].first_in.next_in is NULL:
                self.remove_node_with_indegree_one(self.nodes[u])
                return

        if self.nodes[u].first_out is not NULL:
            if self.nodes[u].first_out.next_out is NULL:
                self.remove_node_with_outdegree_one(self.nodes[u])
                return

    def get_edges(self):
        result: list = []
        for edge in self.edges[:self.number_of_edges]:
            if edge is NULL:
                continue
            if edge.u is NULL:
                continue
            result.append((edge.u.index, edge.v.index))

        return result


cpdef tuple intersect_clades(tree: DiGraph, new_clade: int[:]):
    """
    Modifies a tree such that for a given set of nodes, every clade of the tree either contains it as a subset, is
    a subset of it and its ancestors, or is disjoint from it. The new clade is added to the tree if not already present.
    :param tree:
    :param new_clade: integer-valued numpy array
    :return: tuple with (1) lowest common ancestor of new_clade in the previous tree and (2) each node u such that u
    has a descendent in new_clade, but some sibling of u does not
    """
    cdef int initial_num_nodes = len(new_clade)
    cdef Queue active_nodes = Queue()
    cdef int i
    for i in new_clade:
        active_nodes.push(tree.nodes[i])

    # Bottom-up traversal tracking how many nodes from new_clade are descended from each node
    cdef node* v
    cdef int[:] num_visits = np.zeros(tree.number_of_nodes, dtype=np.intc)
    while active_nodes.length > 0:
        v = active_nodes.pop()

        num_visits[v.index] += 1
        if num_visits[v.index] == initial_num_nodes:  # lowest common ancestor found
            break

        active_nodes.push(v.first_in.u)

    # Top-down traversal pruning unvisited descendents of the LCA
    cdef node* lowest_common_ancestor = v
    active_nodes.clear()
    active_nodes.push(lowest_common_ancestor)
    cdef Queue stranded_nodes = Queue()
    cdef Queue nodes_to_return = Queue()
    cdef edge* e
    cdef node* u
    cdef bint some_child_unvisited
    while active_nodes.length > 0:
        u = active_nodes.pop()
        e = u.first_out
        some_child_unvisited = False

        while e is not NULL:
            if num_visits[e.v.index] == 0:
                some_child_unvisited = True
                tree.remove_edge(e)
                stranded_nodes.push(e.v)
            else:
                active_nodes.push(e.v)

        if not some_child_unvisited:
            continue

        e = u.first_out
        while e is not NULL:
            # Only visited children remain
            nodes_to_return.push(e.v)

        if nodes_to_return.length == 1:
            tree.remove_node_with_outdegree_one(u)

    cdef node* new_node
    if stranded_nodes.length > 0:
        new_node = tree.add_node()
        tree.add_edge(new_node, lowest_common_ancestor)
        if lowest_common_ancestor.first_in is not NULL:
            tree.add_edge(lowest_common_ancestor.first_in.u, new_node)
            tree.remove_edge(lowest_common_ancestor.first_in)

    u = stranded_nodes.pop()
    while u is not NULL:
        tree.add_edge(new_node, u)
        u = stranded_nodes.pop()

    cdef int[:] result = np.empty(nodes_to_return.length, dtype=np.intc)
    for i in range(nodes_to_return.length):
        result[i] = nodes_to_return.pop().index
    return lowest_common_ancestor.index, result
