# brick_graph.pyx
import numpy as np
from .data_structures cimport list_node
from .data_structures cimport LinkedListArray, CountingArray, Stack, IntegerList, IntegerSet
from .digraph cimport node, edge, DiGraph
cimport numpy as cnp
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import os
import h5py
cdef int MAXINT = 2147483647
cdef int TRAVERSAL_METHOD_THRESHOLD = 10
cdef long PACKED_EDGE_CHUNK_SIZE = 262144


cdef class PackedEdges:
    """Owned, chunked endpoint storage for a reduction-union graph."""

    cdef list _chunks
    cdef object _current_array
    cdef cnp.int32_t[:, ::1] _current
    cdef long _chunk_size
    cdef long _used
    cdef long _length
    cdef long _number_of_nodes
    cdef long _allocated_nbytes
    cdef long _first_parent
    cdef long _first_child
    cdef bint _has_first
    cdef bint _finished
    cdef bint _released

    def __cinit__(self, long number_of_nodes, long chunk_size=PACKED_EDGE_CHUNK_SIZE):
        if number_of_nodes < 0:
            raise ValueError("number_of_nodes must be non-negative")
        if number_of_nodes > MAXINT:
            raise OverflowError("Packed endpoint node indices must fit in int32")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._chunks = []
        self._current_array = None
        self._chunk_size = chunk_size
        self._used = 0
        self._length = 0
        self._number_of_nodes = number_of_nodes
        self._allocated_nbytes = 0
        self._has_first = False
        self._finished = False
        self._released = False

    cdef void append_insertion_order(self, long parent, long child):
        """Record insertion order while reproducing DiGraph's edge-index rotation."""
        if self._finished:
            raise RuntimeError("Cannot append to finalized packed edges")
        if not self._has_first:
            self._first_parent = parent
            self._first_child = child
            self._has_first = True
            return
        self._append(parent, child)

    cdef void _append(self, long parent, long child):
        cdef object chunk
        if self._current_array is None or self._used == self._chunk_size:
            chunk = np.empty((self._chunk_size, 2), dtype=np.int32)
            self._chunks.append(chunk)
            self._current_array = chunk
            self._current = chunk
            self._used = 0
            self._allocated_nbytes += chunk.nbytes
        self._current[self._used, 0] = <cnp.int32_t> parent
        self._current[self._used, 1] = <cnp.int32_t> child
        self._used += 1
        self._length += 1

    cdef void finish(self):
        if self._finished:
            return
        if self._has_first:
            self._append(self._first_parent, self._first_child)
        if self._chunks and self._used < self._chunk_size:
            self._chunks[-1] = self._current_array[:self._used]
        self._finished = True

    @property
    def chunks(self):
        if self._released:
            return ()
        return tuple(self._chunks)

    @property
    def number_of_edges(self):
        return self._length

    @property
    def number_of_nodes(self):
        return self._number_of_nodes

    @property
    def allocated_nbytes(self):
        return self._allocated_nbytes

    @property
    def released(self):
        return bool(self._released)

    def take_chunks(self):
        """Transfer ownership of endpoint chunks to a single consumer."""
        if not self._finished:
            raise RuntimeError("Packed edges must be finalized before consumption")
        if self._released:
            raise RuntimeError("Packed edges have already been consumed")
        chunks = self._chunks
        self._chunks = []
        self._current = None
        self._current_array = None
        self._released = True
        return chunks

cdef class BrickGraph:
    """
    Implements the brick graph algorithm. Usage:
    brick_graph, sample_indices, variant_indices = BrickGraph.from_genotypes(genotype_matrix)
    """
    cdef DiGraph graph
    cdef DiGraph tree
    cdef long[:] clade_size
    cdef long root
    cdef LinkedListArray subsequence
    cdef CountingArray times_visited
    cdef CountingArray times_revisited
    cdef long num_samples
    cdef long num_variants
    cdef long direction
    cdef bint save_to_disk
    cdef str out
    cdef object _add_edge_to_file  # Function to add edge to HDF5
    cdef object _save_batch        # Function to save current batch
    cdef object _cleanup          # Function to cleanup HDF5 file
    cdef object _hdf5_file        # HDF5 file object

    @staticmethod
    def forward_backward(genotypes: csc_matrix, bint add_samples = True, bint save_to_disk = False, str out = None):
        """
        Runs the forward and backward brick graph algorithms on a genotype matrix.
        :param genotypes: sparse genotype matrix in csc_matrix format; rows=samples, columns=variants. Order of variants
        matters, order of samples does not.
        :param add_samples: whether to add nodes to the brick graph for the sample haplotypes.
        :param save_to_disk: If True, stream inferred edges to disk; otherwise return both graphs.
        :param out: Output prefix used when save_to_disk is True.
        """
        num_samples, num_variants = genotypes.shape

        cdef int[:] indices = genotypes.indices
        cdef int[:] indptr = genotypes.indptr
        cdef int[:] carriers

        # Forward pass
        cdef BrickGraph forward_pass = BrickGraph(num_samples, num_variants, save_to_disk=save_to_disk, out=f'{out}_forward_graph.h5')
        forward_pass.direction = 1
        cdef long i
        for i in range(num_variants):
            carriers = indices[indptr[i]:indptr[i + 1]]
            forward_pass.intersect_clades(carriers, i)

        # Add samples
        cdef long[:] sample_indices
        cdef node* u
        if add_samples:
            sample_indices =  np.arange(num_variants, num_variants+num_samples, dtype=np.int64)
            for i in range(num_samples):
                u = forward_pass.graph.add_node(sample_indices[i])
                forward_pass.add_edges_from_subsequence(i, sample_indices[i])
                forward_pass.subsequence.clear_list(i)
                assert forward_pass.graph.has_node(sample_indices[i])
                assert forward_pass.graph.number_of_successors(u) == 0
        else:
            sample_indices = np.array([])
        cdef DiGraph forward_graph = forward_pass.graph
        del forward_pass
        if save_to_disk:
            del forward_graph

        # Backward pass
        cdef BrickGraph backward_pass = BrickGraph(num_samples, num_variants, save_to_disk=save_to_disk, out=f'{out}_backward_graph.h5')
        backward_pass.direction = -1
        for i in reversed(range(num_variants)):
            carriers = indices[indptr[i]:indptr[i+1]]
            backward_pass.intersect_clades(carriers, i)
        cdef DiGraph backward_graph = backward_pass.graph
        del backward_pass

        if not save_to_disk:
            return forward_graph, backward_graph, sample_indices
        else:
            del backward_graph
            return sample_indices


    @staticmethod
    def forward_backward_from_hdf5(str genotype_path, bint add_samples = True, str out = None, long batch_nnz = 1000000):
        """Run sequential passes from bounded batches of CSC carrier indices."""
        cdef long num_samples
        cdef long num_variants
        cdef cnp.ndarray[cnp.int64_t, ndim=1] indptr_array
        cdef cnp.int64_t[:] indptr
        cdef cnp.ndarray[cnp.int32_t, ndim=1] batch_indices_array
        cdef int[:] batch_indices
        cdef int[:] carriers
        cdef long batch_start
        cdef long batch_end
        cdef long index_start
        cdef long index_end
        cdef long target
        cdef long i
        cdef BrickGraph forward_pass
        cdef BrickGraph backward_pass
        cdef long[:] sample_indices
        cdef node* u

        if out is None:
            raise ValueError("out is required when streaming forward/backward graphs to disk")
        if batch_nnz <= 0:
            raise ValueError("batch_nnz must be positive")

        with h5py.File(genotype_path, 'r') as f:
            num_samples, num_variants = f['shape'][:]
            indptr_array = np.asarray(f['indptr'][:], dtype=np.int64)
            indptr = indptr_array

            forward_pass = BrickGraph(
                num_samples,
                num_variants,
                save_to_disk=True,
                out=f'{out}_forward_graph.h5',
            )
            forward_pass.direction = 1
            batch_start = 0
            while batch_start < num_variants:
                target = indptr[batch_start] + batch_nnz
                batch_end = np.searchsorted(indptr_array, target, side='right') - 1
                if batch_end <= batch_start:
                    batch_end = batch_start + 1
                if batch_end > num_variants:
                    batch_end = num_variants
                index_start = indptr[batch_start]
                index_end = indptr[batch_end]
                batch_indices_array = np.asarray(f['indices'][index_start:index_end], dtype=np.int32)
                batch_indices = batch_indices_array
                for i in range(batch_start, batch_end):
                    carriers = batch_indices[indptr[i] - index_start:indptr[i + 1] - index_start]
                    forward_pass.intersect_clades(carriers, i)
                batch_start = batch_end

            if add_samples:
                sample_indices = np.arange(num_variants, num_variants + num_samples, dtype=np.int64)
                for i in range(num_samples):
                    u = forward_pass.graph.add_node(sample_indices[i])
                    forward_pass.add_edges_from_subsequence(i, sample_indices[i])
                    forward_pass.subsequence.clear_list(i)
                    assert forward_pass.graph.has_node(sample_indices[i])
                    assert forward_pass.graph.number_of_successors(u) == 0
            else:
                sample_indices = np.array([], dtype=np.int64)

            carriers = None
            batch_indices = None
            batch_indices_array = None
            del forward_pass

            backward_pass = BrickGraph(
                num_samples,
                num_variants,
                save_to_disk=True,
                out=f'{out}_backward_graph.h5',
            )
            backward_pass.direction = -1
            batch_end = num_variants
            while batch_end > 0:
                target = indptr[batch_end] - batch_nnz
                batch_start = np.searchsorted(indptr_array, target, side='left')
                if batch_start >= batch_end:
                    batch_start = batch_end - 1
                index_start = indptr[batch_start]
                index_end = indptr[batch_end]
                batch_indices_array = np.asarray(f['indices'][index_start:index_end], dtype=np.int32)
                batch_indices = batch_indices_array
                for i in reversed(range(batch_start, batch_end)):
                    carriers = batch_indices[indptr[i] - index_start:indptr[i + 1] - index_start]
                    backward_pass.intersect_clades(carriers, i)
                batch_end = batch_start

            del backward_pass

        return sample_indices


    @staticmethod
    def combine_graphs(forward_graph: DiGraph, backward_graph: DiGraph, num_variants: int):
        # For variants i,j with paths i->j and also j->i, combine them into a single node
        cdef long[:] variant_indices = combine_cliques(forward_graph, backward_graph)

        # Transitive reduction of the union of the forward and reverse graphs
        cdef DiGraph brick_graph = reduction_union(forward_graph, backward_graph)

        return brick_graph, variant_indices[:num_variants]

    @staticmethod
    def combine_graphs_packed(forward_graph: DiGraph, backward_graph: DiGraph, num_variants: int):
        """Combine graphs into packed endpoints for direct recombination construction."""
        cdef long[:] variant_indices = combine_cliques(forward_graph, backward_graph)
        cdef PackedEdges packed_edges = reduction_union_packed(forward_graph, backward_graph)
        return packed_edges, variant_indices[:num_variants]


    @staticmethod
    def from_genotypes(genotypes: csc_matrix, add_samples: bint = True) -> tuple[DiGraph, int[:], int[:]]:
        """
        Runs the brick graph algorithm on a genotype matrix
        :param genotypes: sparse genotype matrix in csc_matrix format; rows=samples, columns=variants. Order of variants
        matters, order of samples does not.
        :param add_samples: whether to add nodes to the brick graph for the sample haplotypes.
        """
        forward_graph, backward_graph, sample_indices = BrickGraph.forward_backward(genotypes, add_samples)
        brick_graph, variant_indices = BrickGraph.combine_graphs(forward_graph, backward_graph, genotypes.shape[1])
        return brick_graph, sample_indices, variant_indices


    def __cinit__(self, int num_samples, int num_variants, bint save_to_disk=False, str out=None):

        if save_to_disk:
            assert out is not None
            # Initialize HDF5 file and get helper functions
            self._add_edge_to_file, self._save_batch, self._cleanup, self._hdf5_file = build_sparse_matrix_with_hdf5(
                out,
                num_variants + num_samples,
                batch_size=100000
            )
        else:
            self._add_edge_to_file = None
            self._save_batch = None
            self._cleanup = None
            self._hdf5_file = None

        cnp.import_array()
        self.num_samples = num_samples
        self.num_variants = num_variants
        # Disk mode streams every inferred edge through ``self._add_edge_to_file``.
        # Keep the full node arena used by sample bookkeeping, but avoid an
        # otherwise unused edge arena proportional to the partition width.
        cdef int graph_edge_capacity = 1 if save_to_disk else num_variants + num_samples
        self.graph = DiGraph(num_variants + num_samples, graph_edge_capacity)
        # self.graph.initialize_all_nodes()
        self.initialize_tree()
        tree_num_nodes = self.tree.maximum_number_of_nodes
        self.times_visited = CountingArray(tree_num_nodes)
        self.times_revisited = CountingArray(tree_num_nodes)
        self.direction = 0
        self.save_to_disk = save_to_disk
        self.out = out

    @property
    def _native_graph_stats(self):
        """Expose native graph allocation counters for focused regression tests."""
        return self.graph.number_of_edges, self.graph.max_edges


    def __dealloc__(self):
        if self.save_to_disk:
            self._cleanup()

    cpdef void add_edge(self, variant_idx, node_idx):

        if self.save_to_disk:
            self._add_edge_to_file(variant_idx, node_idx)
        else:
            self.graph.add_edge(variant_idx, node_idx)


    cpdef void initialize_tree(self):
        self.tree = DiGraph(self.num_samples * 2, self.num_samples * 2 - 1)
        self.root = self.num_samples
        self.clade_size = np.zeros(self.num_samples * 2, dtype=np.int64)
        cdef list edges = [(self.root, i) for i in range(self.num_samples)]
        self.tree.add_edges_from(edges)
        for i in range(self.num_samples):
            self.clade_size[i] = 1
        self.clade_size[self.root] = self.num_samples
        # Allocate capacity based on expected usage: num_lists * avg_list_length
        cdef long initial_capacity = self.tree.maximum_number_of_nodes
        self.subsequence = LinkedListArray(self.tree.maximum_number_of_nodes, initial_capacity)


    cpdef void intersect_clades(self, int[:] new_clade, long clade_index):
        """
        Adds a new clade to a rooted tree and splits existing clades if they intersect with the new clade. Returns the
        lowest common ancestor from the previous tree of nodes in the new clade.
        """
        cdef long new_clade_size = len(new_clade)
        if new_clade_size == 0:
            return

        # Find LCA of the clade while tracking in self.num_visits the number of carriers descended from each node
        cdef node * lowest_common_ancestor
        if new_clade_size * TRAVERSAL_METHOD_THRESHOLD < self.num_samples:
            lowest_common_ancestor = self.partial_traversal(new_clade)
        else:
            lowest_common_ancestor = self.partial_traversal2(new_clade)
        assert lowest_common_ancestor is not NULL

        self.add_edges_from_subsequence(lowest_common_ancestor.index, clade_index)

        cdef IntegerList traversal = IntegerList(2 * len(new_clade))
        self.times_revisited.clear()
        cdef int i
        for i in new_clade:
            traversal.push(i)
            self.times_revisited.set_element(i, 1)

        cdef edge* out_edge
        cdef edge* visited_edge
        cdef edge* unvisited_edge
        cdef node* sibling_node
        cdef node* new_root
        cdef node* parent_of_v
        cdef node* v = NULL
        cdef long node_idx
        cdef bint v_is_root
        cdef Stack visited_children, unvisited_children
        cdef long num_children_visited, num_children_unvisited, visits

        while traversal.length > 0:
            node_idx = traversal.pop()
            v = &self.tree.nodes[node_idx]
            # Push a node when all its visited children have been found
            if v.first_in != NULL:
                i = v.first_in.u.index
                visits = self.times_revisited.increment_element(i, self.times_revisited.get_element(node_idx))
                if visits == self.times_visited.get_element(i):
                    traversal.push(i)

            # No unvisited children: means intersect(v, new_clade) == v
            if self.times_visited[node_idx] == self.clade_size[node_idx]:
                continue

            visited_children, unvisited_children = self.get_visited_children(v)
            num_children_unvisited, num_children_visited = unvisited_children.length, visited_children.length
            assert num_children_unvisited > 0

            # If v is the LCA, then its clade is not partitioned, but rather a subclade is produced
            if node_idx == lowest_common_ancestor.index:
                assert num_children_visited > 1
                child_node = self.tree.add_node(-1)
                assert child_node.index < 2 * self.num_samples
                self.clade_size[child_node.index] = new_clade_size

                self.tree.add_edge(node_idx, child_node.index)
                while visited_children.length > 0:
                    child = visited_children.pop()
                    visited_edge = self.tree.nodes[child].first_in
                    self.tree.set_edge_parent(visited_edge, child_node)
                lowest_common_ancestor = child_node  # LCA in new tree
                break

            assert v.first_in is not NULL
            parent_of_v = v.first_in.u

            # Exactly one visited and one unvisited child: delete v, as there are existing nodes
            # for both intersect(v, new_clade) and intersect(v, new_clade_complement)
            if num_children_visited == 1 and num_children_unvisited == 1:
                self.subsequence.copy_list(node_idx, visited_children.pop())
                self.subsequence.copy_list(node_idx, unvisited_children.pop())
                self.subsequence.clear_list(node_idx)
                self.clade_size[node_idx] = 0
                self.tree.collapse_node_with_indegree_one(v)
                continue

            # Exactly one child w is visited: there is an existing node for intersect(v, new_clade);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade_complement)
            if num_children_visited == 1:
                i = visited_children.pop()
                visited_edge = self.tree.nodes[i].first_in
                self.tree.set_edge_parent(visited_edge, parent_of_v)
                self.times_visited.set_element(node_idx, 0)
                self.subsequence.copy_list(node_idx, i)
                self.clade_size[node_idx] -= self.clade_size[i]
                continue

            # Exactly one child is w is unvisited: there is an existing node for intersect(v, new_clade_complement);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade)
            if num_children_unvisited == 1:
                i = unvisited_children.pop()
                unvisited_edge = self.tree.nodes[i].first_in
                self.tree.set_edge_parent(unvisited_edge, parent_of_v)
                self.subsequence.copy_list(node_idx, i)
                self.clade_size[node_idx] -= self.clade_size[i]
                continue

            # Multiple visited and unvisited children: create new_node for intersect(v, new_clade_complement)
            # and replace v with intersect(v, new_clade)
            sibling_node = self.tree.add_node(-1)
            assert sibling_node.index < self.num_samples * 2
            self.times_visited.set_element(sibling_node.index, 0)
            self.subsequence.copy_list(node_idx, sibling_node.index)
            self.tree.add_edge(parent_of_v.index, sibling_node.index)
            while unvisited_children.length > 0:
                child = unvisited_children.pop()
                unvisited_edge = self.tree.nodes[child].first_in
                self.tree.set_edge_parent(unvisited_edge, sibling_node)
                self.clade_size[node_idx] -= self.clade_size[child]
                self.clade_size[sibling_node.index] += self.clade_size[child]

        self.subsequence.clear_list(lowest_common_ancestor.index)
        self.subsequence.extend(lowest_common_ancestor.index, clade_index)

    cdef long visit_node(self, node* v, node* predecessor):
        """
        Visits a node in the tree, incrementing the number of visits for each node.
        """
        cdef edge* e = v.first_out
        cdef node* w
        cdef long count = self.times_visited.get_element(v.index)
        
        while e != NULL:
            w = e.v
            if w == predecessor:
                # Don't recurse into predecessor, just read its already-computed count
                count += self.times_visited.get_element(w.index)
            else:
                count += self.visit_node(w, v)
            e = e.next_out
        self.times_visited.set_element(v.index, count)
        return count

    cdef node* partial_traversal(self, int[:] leaves):
        """
        Finds the lowest common ancestor of an array of leaves in the tree. For all descendants of the LCA, counts
        the number of leaves that are descended from them.
        """
        self.times_visited.clear()
        cdef int num_leaves = len(leaves)
        if num_leaves == 0:
            return <node*> NULL

        # Bottom-up traversal from every leaf node to the root
        cdef int i
        cdef node * v
        cdef int num_visits
        for i in leaves:
            v = &self.tree.nodes[i]
            while True:
                num_visits = self.times_visited.get_element(v.index) + 1
                self.times_visited.set_element(v.index, num_visits)
                if num_visits == num_leaves: # reached LCA
                    assert i == leaves[num_leaves-1]
                    break
                if v.first_in is NULL: # reached root
                    break
                v = v.first_in.u

        cdef node * lowest_common_ancestor = v
        return lowest_common_ancestor

    cdef node* partial_traversal2(self, int[:] leaves):
        """
        Finds the lowest common ancestor of an array of leaves in the tree. 
        For all descendants of the LCA, counts the number of leaves that are descended from them
        and stores the result in self.times_visited.
        """
        self.times_visited.clear()
        cdef long num_leaves = len(leaves)
        cdef long i
        cdef node* u
        cdef node* v
        cdef node* lowest_common_ancestor = NULL
        
        if num_leaves == 0:
            return <node*> NULL
        
        for i in leaves:
            self.times_visited.set_element(i, 1)

        u = &self.tree.nodes[leaves[0]]
        v = NULL
        cdef long num_visits = 0
        while True:
            num_visits = self.visit_node(u, v)
            if num_visits == num_leaves:
                break
            if u.first_in is NULL:
                break
            v = u
            u = u.first_in.u

        lowest_common_ancestor = u
        return lowest_common_ancestor

    cdef void add_edges_from_subsequence(self, long subsequence_index, long node_index):
        """
        Adds edges in self.graph from every node u_k in a subsequence to a node, but only if for all succeeding
{{ ... }}
        nodes u_j, j>k, there is no path u_k->u_j.
        """
        cdef long tree_node = subsequence_index
        cdef long node_idx
        cdef long last_variant_found = -MAXINT * self.direction
        cdef long variant_idx
        while True:
            node_idx = self.subsequence.head[tree_node]
            while node_idx != -1:
                variant_idx = self.subsequence.nodes[node_idx].value
                if variant_idx * self.direction > last_variant_found * self.direction:
                    self.add_edge(variant_idx, node_index)
                    last_variant_found = variant_idx
                node_idx = self.subsequence.nodes[node_idx].next
            if self.tree.nodes[tree_node].first_in is NULL:
                break
            tree_node = self.tree.nodes[tree_node].first_in.u.index

    cdef tuple[Stack, Stack] get_visited_children(self, node* v):
        """
        Separates the children of node v into those with and without variant carriers as descendants.
        """
        cdef Stack visited_children = Stack()
        cdef Stack unvisited_children = Stack()
        cdef long child
        cdef edge* e = v.first_out
        while e is not NULL:
            child = e.v.index
            e = e.next_out
            if self.times_visited.get_element(child) > 0:
                visited_children.push(child)
            else:
                unvisited_children.push(child)
        return visited_children, unvisited_children

cpdef long[:] combine_cliques(DiGraph forward_graph, DiGraph backward_graph):
    """
    Finds sequences u<v<...<w connected via edges (u,v),(v,...,w) in the forward graph and (w,...,v),(v,u) in the
    backward graph. Collapses these into a single node and returns an array of node assignments.
    """
    cdef long num_nodes = backward_graph.maximum_number_of_nodes
    cdef long[:] result = np.arange(num_nodes, dtype=np.int64)
    cdef long node_index
    cdef long neighbor_index
    cdef long neighbor_of_neighbor
    cdef IntegerSet neighbors = IntegerSet(forward_graph.maximum_number_of_nodes)
    cdef node * current_node
    cdef node * backward_node
    cdef edge * current_edge
    cdef edge * back_edge
    for node_index in range(num_nodes):
        if not forward_graph.has_node(node_index) or not backward_graph.has_node(node_index):
            continue

        # If some neighbor of the current node has a back-edge to the current node as well, then it must be the first
        # neighbor due to the order in which edges are added (last in, first out)
        current_edge = forward_graph.nodes[node_index].first_out
        if current_edge is NULL:
            continue
        neighbor_index = current_edge.v.index
        if not backward_graph.has_node(neighbor_index):
            continue

        # Similarly, back edge is the first if it exists
        back_edge = backward_graph.nodes[neighbor_index].first_out
        if back_edge is NULL:
            continue
        neighbor_of_neighbor = back_edge.v.index
        if not neighbor_of_neighbor == node_index:
            continue

        # Remove node_index from each graph, assigning its incoming edges in the forward graph and its outgoing
        # edges in the backward graph to its neighbor
        contract_edge(current_edge, back_edge, forward_graph, backward_graph, neighbors)
        result[node_index] = neighbor_index

    # If a clique u<v<w has size >2, assign u to w instead of v
    for node_index in reversed(range(num_nodes)):
        result[node_index] = result[result[node_index]]

    return result

cdef void contract_edge(edge* forward_edge,
                        edge* backward_edge,
                        DiGraph forward_graph,
                        DiGraph backward_graph,
                        IntegerSet neighbors):
    """
    Contract the edges between u and v in the forward graph and v and u in the backward graph. In the forward graph,
    in-neighbors w of u are added as in-neighbors of v if for all w' with an edge (w', u) in the backward graph,
    (w, w') is not an edge of the forward graph. This ensures that there is no other path w, w', ..., v in the
    forward graph. In the backward graph, out-neighbors of u are handled similarly. Additionally, edges are added
    between forward graph in-neighbors w of u and out-neighbors w' of u if w' < v; similarly, between backward graph
    in-neighbors w of u and out-neighbors w' of u if w < v.
    """
    u_idx = forward_edge.u.index
    v_idx = forward_edge.v.index
    assert u_idx == backward_edge.v.index
    assert v_idx == backward_edge.u.index
    cdef edge* e_in
    cdef edge* e_out
    cdef node* w

    # Add edges between forward graph in-neighbors w of u and out-neighbors w' of u if w' < v
    e_in = forward_edge.u.first_in
    while e_in is not NULL:
        e_out = forward_edge.u.first_out
        while e_out is not NULL:
            if e_out.v.index < v_idx:
                add_nontransitive_edge(forward_graph, e_in.u.index, e_out.v.index, u_idx)
            e_out = e_out.next_out
        e_in = e_in.next_in

    # Add edges between backward graph in-neighbors w of u and out-neighbors w' of u if w < v
    e_in = backward_edge.v.first_in
    while e_in is not NULL:
        if e_in.u.index < v_idx:
            e_out = backward_edge.v.first_out
            while e_out is not NULL:
                add_nontransitive_edge(backward_graph, e_in.u.index, e_out.v.index, u_idx)
                e_out = e_out.next_out
        e_in = e_in.next_in

    # For edges (w, u) in the forward graph, add an edge (w, v) if for all w' with an edge (w', u) in the backward
    # graph, (w, w') is not an edge of the forward graph
    # The previous step does not interfere with this because if w' has an edge (w', u) (backward), there is not also
    # an edge (u, w') (forward).
    neighbors.clear()
    search_two_hops_backward(neighbors, backward_graph, forward_graph, u_idx)
    assert neighbors.contains(u_idx)
    cdef edge* e_wu = forward_edge.u.first_in
    cdef edge* next_edge
    while e_wu is not NULL:
        w = e_wu.u
        next_edge = e_wu.next_in
        if not neighbors.contains(w.index):
            forward_graph.set_edge_child(e_wu, forward_edge.v)
        e_wu = next_edge

    # For edges (u, w) in the backward graph, add an edge (v, w) if for all w' with an edge (u, w') in the forward
    # graph, (w', w) is not an edge of the backward graph
    neighbors.clear()
    search_two_hops(neighbors, forward_graph, backward_graph, u_idx)
    assert neighbors.contains(u_idx)
    cdef edge* e_uw = backward_edge.v.first_out
    while e_uw is not NULL:
        w = e_uw.v
        next_edge = e_uw.next_out
        if not neighbors.contains(w.index):
            backward_graph.set_edge_parent(e_uw, backward_edge.u)
        e_uw = next_edge

    forward_graph.remove_node(forward_edge.u)
    backward_graph.remove_node(backward_edge.v)

cdef edge* add_nontransitive_edge(DiGraph graph, long u_idx, long v_idx, long skip_node):
    """
    Adds an edge between u and v if there is not already a path u, w_1,...,w_n, v with u < w_k < v or u > w_k > v.
    Skips over skip_node when searching.
    :return: the edge, or NULL if a path was found
    """
    # Search for descendants of u
    cdef long direction = 1 if u_idx < v_idx else -1
    cdef edge * e
    cdef long node
    cdef Stack nodes_to_visit = Stack(graph.maximum_number_of_nodes)
    nodes_to_visit.push(u_idx)
    while nodes_to_visit.length > 0:
        node = nodes_to_visit.pop()

        e = graph.nodes[node].first_out
        while e is not NULL:
            assert direction * e.v.index > direction * u_idx
            if e.v.index == skip_node:
                pass
            elif direction * e.v.index < direction * v_idx:
                nodes_to_visit.push(e.v.index)
            elif e.v.index == v_idx:
                return <edge *> NULL
            e = e.next_out

    # v was not found
    # print(u_idx, v_idx)
    return graph.add_edge(u_idx, v_idx)


cpdef DiGraph reduction_union(DiGraph forward_reduction, DiGraph backward_reduction):
    """
    Computes the transitive reduction of the union of the partial orderings defined by two DAGs, which are the
    transitive reductions of the intersections of some partial ordering with a total ordering and its negation.
    :param forward_reduction: the transitive reduction of intersect(partial ordering, total ordering)
    :param backward_reduction: the transitive reduction of intersect(partial ordering, total ordering reversed)
    :param prune_loops: whether to prune edges (v,u) in backward_reduction if (u,v) is in forward_reduction
    :return: the transitive reduction of the partial ordering
    """
    cdef long num_nodes = max(forward_reduction.maximum_number_of_nodes, backward_reduction.maximum_number_of_nodes)
    cdef IntegerSet reachable_in_two_hops = IntegerSet(num_nodes)
    cdef long node_index
    cdef node * current_node
    cdef edge * out_edge
    cdef DiGraph result = DiGraph(num_nodes, forward_reduction.number_of_edges + backward_reduction.number_of_edges)

    for node_index in range(num_nodes):
        # Set of nodes that is reachable in two hops from this one
        reachable_in_two_hops.clear()
        if forward_reduction.is_node(node_index):
            search_two_hops(reachable_in_two_hops, forward_reduction, backward_reduction, node_index)
        if backward_reduction.is_node(node_index):
            search_two_hops(reachable_in_two_hops, backward_reduction, forward_reduction, node_index)

        # Add neighbors that aren't reachable in two hops
        if forward_reduction.is_node(node_index):
            current_node = &forward_reduction.nodes[node_index]
            add_nonredundant_neighbors(result, current_node, reachable_in_two_hops)
        if backward_reduction.is_node(node_index):
            current_node = &backward_reduction.nodes[node_index]
            add_nonredundant_neighbors(result, current_node, reachable_in_two_hops)
        
        if not result.has_node(node_index):
            result.add_node(node_index)

    return result


cpdef PackedEdges reduction_union_packed(
    DiGraph forward_reduction,
    DiGraph backward_reduction,
    long chunk_size=PACKED_EDGE_CHUNK_SIZE,
):
    """Compute a reduction union into packed int32 endpoint chunks.

    Endpoints are stored in the edge-index order that ``reduction_union``
    exposes. ``DiGraph`` assigns the first inserted edge to the last slot of
    its initial arena, so this is the insertion sequence rotated left by one.
    """
    cdef long num_nodes = max(forward_reduction.maximum_number_of_nodes, backward_reduction.maximum_number_of_nodes)
    cdef IntegerSet reachable_in_two_hops = IntegerSet(num_nodes)
    cdef long node_index
    cdef node * current_node
    cdef PackedEdges result = PackedEdges(num_nodes, chunk_size)

    for node_index in range(num_nodes):
        reachable_in_two_hops.clear()
        if forward_reduction.is_node(node_index):
            search_two_hops(reachable_in_two_hops, forward_reduction, backward_reduction, node_index)
        if backward_reduction.is_node(node_index):
            search_two_hops(reachable_in_two_hops, backward_reduction, forward_reduction, node_index)

        if forward_reduction.is_node(node_index):
            current_node = &forward_reduction.nodes[node_index]
            add_nonredundant_neighbors_packed(result, current_node, reachable_in_two_hops)
        if backward_reduction.is_node(node_index):
            current_node = &backward_reduction.nodes[node_index]
            add_nonredundant_neighbors_packed(result, current_node, reachable_in_two_hops)

    result.finish()
    return result

# Subroutines of reduction_union
cdef void search_two_hops(IntegerSet result, DiGraph first_graph, DiGraph second_graph, long starting_node_index):
    """
    Searches from a starting node u to find nodes w such that for some v, (u,v) is an edge of first_graph and (v,w)
    is an edge of second_graph.
    """
    cdef node * starting_node = &first_graph.nodes[starting_node_index]
    cdef edge * first_hop
    cdef edge * second_hop
    first_hop = starting_node.first_out
    while first_hop is not NULL:
        if not second_graph.has_node(first_hop.v.index):
            first_hop = first_hop.next_out
            continue
        second_hop = second_graph.nodes[first_hop.v.index].first_out
        while second_hop is not NULL:
            result.add(second_hop.v.index)
            second_hop = second_hop.next_out
        first_hop = first_hop.next_out

cdef void search_two_hops_backward(IntegerSet result, DiGraph first_graph, DiGraph second_graph, long starting_node_index):
    """
    Searches from a starting node u to find nodes w such that for some v, (v,u) is an edge of first_graph and (w,v)
    is an edge of second_graph.
    """
    cdef node * starting_node = &first_graph.nodes[starting_node_index]
    cdef edge * first_hop
    cdef edge * second_hop
    first_hop = starting_node.first_in
    while first_hop is not NULL:
        if not second_graph.has_node(first_hop.u.index):
            first_hop = first_hop.next_in
            continue
        second_hop = second_graph.nodes[first_hop.u.index].first_in
        while second_hop is not NULL:
            result.add(second_hop.u.index)
            second_hop = second_hop.next_in
        first_hop = first_hop.next_in

cdef void add_nonredundant_neighbors(DiGraph result, node * starting_node, IntegerSet neighbors_to_exclude):
    """
    Copies neighbors of starting_node to the graph result, except for those in neighbors_to_exclude.
    """
    cdef edge * out_edge = starting_node.first_out
    while out_edge is not NULL:
        if not neighbors_to_exclude.contains(out_edge.v.index):
            result.add_edge(starting_node.index, out_edge.v.index)
        out_edge = out_edge.next_out


cdef void add_nonredundant_neighbors_packed(
    PackedEdges result,
    node * starting_node,
    IntegerSet neighbors_to_exclude,
):
    """Append nonredundant neighbor endpoints in reduction-union insertion order."""
    cdef edge * out_edge = starting_node.first_out
    while out_edge is not NULL:
        if not neighbors_to_exclude.contains(out_edge.v.index):
            result.append_insertion_order(starting_node.index, out_edge.v.index)
        out_edge = out_edge.next_out

cpdef tuple read_brick_graph_h5(filename):
    """
    Read in brick graph from h5 file.
    :param filename: path to .h5 file
    :return: graph (DiGraph), sample_indices, variant_indices
    """
    with h5py.File(filename, 'r') as f:
        A = csc_matrix((f['data'][:], f['indices'][:], f['indptr'][:]), shape=(f.attrs['n'], f.attrs['n']))
        variant_indices = f['variant_indices'][:]
        sample_indices = f['sample_indices'][:]
    graph = DiGraph.from_csc(A)
    # graph.initialize_all_nodes()
    return graph, sample_indices, variant_indices

cpdef tuple get_graph_statistics(str brick_graph_dir, list partition_identifiers):
    """
    Get merged graph statistics from directory of brick graph partitions.
    :param brick_graph_dir: location of brick graph partitions
    :return: num_samples, number_of_nodes, number_of_edges
    """
    cdef long number_of_nodes = 0
    cdef long number_of_edges = 0
    for f in partition_identifiers:
        path = f'{brick_graph_dir}/{f}.h5'
        with h5py.File(path, 'r') as f:
            num_samples = f['sample_indices'].shape[0]
            number_of_nodes += f.attrs['n']
            number_of_edges += f['data'].shape[0]
    number_of_nodes -= num_samples * (len(partition_identifiers)-1)
    return num_samples, number_of_nodes, number_of_edges

cdef void add_neighbors(DiGraph graph_to_modify, node* v, long[:] node_ids):
    """
    Add edges from the parents of node v to node v to graph_to_modify using the index mapping node_ids.
    :param graph_to_modify: DiGraph object
    :param v: pointer to node v
    :param node_ids: index mapping such that node_ids[i] maps to the index of node i in graph_to_modify
    :return: None
    """
    cdef edge* e = v.first_in
    if e is NULL:
        return
    node_idx = node_ids[v.index]
    while e.next_in is not NULL:
        e = e.next_in
    while e is not NULL:
        graph_to_modify.add_edge(node_ids[e.u.index], node_idx)
        e = e.prev_in


cdef long add_neighbors_defer_first(
    DiGraph graph_to_modify,
    node* v,
    long[:] node_ids,
    bint* has_deferred_edge,
    long* deferred_parent,
    long* deferred_child,
):
    """Add incoming edges while retaining the first edge for allocator-order compatibility."""
    cdef edge* e = v.first_in
    cdef long node_idx
    cdef long parent_idx
    cdef long num_edges = 0
    if e is NULL:
        return 0
    node_idx = node_ids[v.index]
    while e.next_in is not NULL:
        e = e.next_in
    while e is not NULL:
        parent_idx = node_ids[e.u.index]
        if not has_deferred_edge[0]:
            deferred_parent[0] = parent_idx
            deferred_child[0] = node_idx
            has_deferred_edge[0] = True
        else:
            graph_to_modify.add_edge(parent_idx, node_idx)
        num_edges += 1
        e = e.prev_in
    return num_edges


cdef tuple _fill_merged_graph(
    DiGraph result,
    str brick_graph_dir,
    list partition_identifiers,
    long num_samples,
    bint preserve_copy_order,
    bint collect_index_mapping,
):
    cdef DiGraph graph
    cdef long number_of_nodes
    cdef long i
    cdef long var
    cdef long sample_idx
    cdef long non_sample_counter = num_samples
    cdef long sample_counter
    cdef long edge_count = 0
    cdef long[:] new_node_ids
    cdef long[:] samples_view
    cdef unsigned char[:] sample_mask
    cdef list variant_indices = []
    cdef object index_mapping = [] if collect_index_mapping else None
    cdef bint has_deferred_edge = False
    cdef long deferred_parent = -1
    cdef long deferred_child = -1

    for filename in partition_identifiers:
        graph, samples, variants = read_brick_graph_h5(f'{brick_graph_dir}/{filename}.h5')
        number_of_nodes = graph.maximum_number_of_nodes
        samples_view = np.asarray(samples, dtype=np.int64)
        if len(samples_view) != num_samples:
            raise ValueError("Brick graph partitions have inconsistent sample counts")

        # Match the previous ascending-node mapping in O(nodes + samples).
        sample_mask = np.zeros(number_of_nodes, dtype=np.uint8)
        for sample_idx in samples_view:
            if sample_idx < 0 or sample_idx >= number_of_nodes:
                raise ValueError("Sample index is outside the brick graph")
            sample_mask[sample_idx] = 1

        sample_counter = 0
        new_node_ids = np.empty(number_of_nodes, dtype=np.int64)
        for i in range(number_of_nodes):
            if sample_mask[i]:
                new_node_ids[i] = sample_counter
                sample_counter += 1
            else:
                new_node_ids[i] = non_sample_counter
                non_sample_counter += 1
        if sample_counter != num_samples:
            raise ValueError("Brick graph sample indices contain duplicates")
        if non_sample_counter > result.maximum_number_of_nodes:
            raise ValueError("Merged graph node capacity is too small")

        for var in variants:
            variant_indices.append(new_node_ids[var])
        if collect_index_mapping:
            index_mapping.append(np.asarray(new_node_ids))

        # Add incoming edges in the same parent order as the previous merge.
        # The old merge graph's allocator rotated the first inserted edge to
        # the final source slot; Recombination.copy_from then visited that edge
        # last. Deferring one edge reproduces that effective insertion order
        # without constructing and copying an intermediate DiGraph.
        for i in range(number_of_nodes):
            if not graph.is_node(i):
                continue
            if preserve_copy_order:
                edge_count += add_neighbors_defer_first(
                    result,
                    &graph.nodes[i],
                    new_node_ids,
                    &has_deferred_edge,
                    &deferred_parent,
                    &deferred_child,
                )
            else:
                edge_count += graph.number_of_predecessors(&graph.nodes[i])
                add_neighbors(result, &graph.nodes[i], new_node_ids)

    if preserve_copy_order and has_deferred_edge:
        result.add_edge(deferred_parent, deferred_child)

    return variant_indices, non_sample_counter, index_mapping, edge_count


cpdef tuple merge_brick_graphs_into(
    DiGraph result,
    str brick_graph_dir,
    list partition_identifiers,
    long num_samples,
    long number_of_nodes,
    bint preserve_copy_order=True,
):
    """Fill an existing graph directly from partition files without retaining mappings."""
    cdef long node_idx
    for node_idx in range(number_of_nodes):
        result.add_node(node_idx)

    variant_indices, merged_number_of_nodes, _, edge_count = _fill_merged_graph(
        result,
        brick_graph_dir,
        partition_identifiers,
        num_samples,
        preserve_copy_order,
        False,
    )
    return variant_indices, num_samples, merged_number_of_nodes, edge_count


cpdef tuple merge_brick_graphs(str brick_graph_dir, list partition_identifiers):
    """
    Merge multiple brick graphs with shared sample nodes and other nodes disjoint.
    :param brick_graph_dir: location of brick graph partitions
    :return: merged graph, variant indices, index mapping
    """
    num_samples, number_of_nodes, number_of_edges = get_graph_statistics(brick_graph_dir, partition_identifiers) # get statistics to initialize DiGraph object
    cdef DiGraph result = DiGraph(number_of_nodes, number_of_edges)
    variant_indices, _, index_mapping, edge_count = _fill_merged_graph(
        result,
        brick_graph_dir,
        partition_identifiers,
        num_samples,
        False,
        True,
    )
    if edge_count != number_of_edges:
        raise ValueError("Merged edge count does not match partition metadata")
    return result, variant_indices, num_samples, index_mapping


def build_sparse_matrix_with_hdf5(filename, n, batch_size=100000):

    f = h5py.File(filename, 'w')
    f.attrs['n'] = n
    f.create_dataset('rows', (0,), maxshape=(None,), dtype=np.int32)
    f.create_dataset('cols', (0,), maxshape=(None,), dtype=np.int32)

    rows, cols = [], []

    def add_edge_to_file(i, j):
        rows.append(i)
        cols.append(j)

        if len(rows) >= batch_size:
            save_batch()
            rows.clear()
            cols.clear()

    def save_batch():
        if not rows:
            return

        current_size = f['rows'].shape[0]
        new_size = current_size + len(rows)

        f['rows'].resize((new_size,))
        f['rows'][current_size:new_size] = rows

        f['cols'].resize((new_size,))
        f['cols'][current_size:new_size] = cols

    def cleanup():
        save_batch()
        f.close()

    return add_edge_to_file, save_batch, cleanup, f


@staticmethod
def read_graph_from_disk(file_path):
    """
    Read graph from HDF5 file adding edges in the order they were inferred.
    """
    with h5py.File(file_path, 'r') as f:
        n = f.attrs['n']
        rows = f['rows'][:]
        cols = f['cols'][:]

    digraph = DiGraph(n, len(rows))
    # digraph.initialize_all_nodes()

    # add edges in order they were stored
    for i in range(len(rows)):
        digraph.add_edge(rows[i], cols[i])

    return digraph
