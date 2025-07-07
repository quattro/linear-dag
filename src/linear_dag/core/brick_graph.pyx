# brick_graph.pyx
import numpy as np
from .data_structures cimport node, edge, list_node
from .data_structures cimport DiGraph, LinkedListArray, CountingArray, Stack, IntegerList, IntegerSet
cimport numpy as cnp
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
import os
import h5py
cdef int MAXINT = 2147483647


cdef class BrickGraph:
    """
    Implements the brick graph algorithm. Usage:
    brick_graph, sample_indices, variant_indices = BrickGraph.from_genotypes(genotype_matrix)
    """
    cdef DiGraph graph
    cdef DiGraph tree
    cdef int[:] clade_size
    cdef int root
    cdef LinkedListArray subsequence
    cdef CountingArray times_visited
    cdef CountingArray times_revisited
    cdef int num_samples
    cdef int num_variants
    cdef int direction
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
        :param save_to_disk: If False, saves the forward and backward graph as a sparse matrix to disk. If True, returns forward and backward graphs.
        :param out: If save_to_disk is True, the path save the forward and backward graphs.
        """

        num_samples, num_variants = genotypes.shape

        # Forward pass
        cdef BrickGraph forward_pass = BrickGraph(num_samples, num_variants, save_to_disk=save_to_disk, out=f'{out}_forward_graph.h5')
        forward_pass.direction = 1
        cdef int i
        for i in range(num_variants):
            carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i + 1]]
            forward_pass.intersect_clades(carriers, i)

        # Add samples
        cdef int[:] sample_indices
        if add_samples:
            sample_indices =  np.arange(num_variants, num_variants+num_samples, dtype=np.intc)
            for i in range(num_samples):
                forward_pass.add_edges_from_subsequence(i, sample_indices[i])
                forward_pass.subsequence.clear_list(i)
        else:
            sample_indices = np.array([])
        cdef DiGraph forward_graph = forward_pass.graph

        # Backward pass
        cdef BrickGraph backward_pass = BrickGraph(num_samples, num_variants, save_to_disk=save_to_disk, out=f'{out}_backward_graph.h5')
        backward_pass.direction = -1
        for i in reversed(range(num_variants)):
            carriers = genotypes.indices[genotypes.indptr[i]:genotypes.indptr[i+1]]
            backward_pass.intersect_clades(carriers, i)
        cdef DiGraph backward_graph = backward_pass.graph

        if not save_to_disk:
            return forward_graph, backward_graph, sample_indices
        else:
            del forward_graph # calls __dealloc__ to save to disk
            del backward_graph
            return sample_indices


    @staticmethod
    def combine_graphs(forward_graph: DiGraph, backward_graph: DiGraph, num_variants: int):

        # For variants i,j with paths i->j and also j->i, combine them into a single node
        cdef int[:] variant_indices = combine_cliques(forward_graph, backward_graph)

        # Transitive reduction of the union of the forward and reverse graphs
        cdef DiGraph brick_graph = reduction_union(forward_graph, backward_graph)

        return brick_graph, variant_indices[:num_variants]


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
        self.graph = DiGraph(num_variants + num_samples, num_variants + num_samples)
        self.graph.initialize_all_nodes()
        self.initialize_tree()
        tree_num_nodes = self.tree.maximum_number_of_nodes
        self.times_visited = CountingArray(tree_num_nodes)
        self.times_revisited = CountingArray(tree_num_nodes)
        self.direction = 0
        self.save_to_disk = save_to_disk
        self.out = out


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
        self.clade_size = np.zeros(self.num_samples * 2, dtype=np.intc)
        cdef list edges = [(self.root, i) for i in range(self.num_samples)]
        self.tree.add_edges_from(edges)
        for i in range(self.num_samples):
            self.clade_size[i] = 1
        self.clade_size[self.root] = self.num_samples
        self.subsequence = LinkedListArray(self.tree.maximum_number_of_nodes)


    cpdef void intersect_clades(self, int[:] new_clade, int clade_index):
        """
        Adds a new clade to a rooted tree and splits existing clades if they intersect with the new clade. Returns the
        lowest common ancestor from the previous tree of nodes in the new clade.
        """
        cdef int new_clade_size = len(new_clade)
        if new_clade_size == 0:
            return

        # Find LCA of the clade while tracking in self.num_visits the number of carriers descended from each node
        cdef node * lowest_common_ancestor = self.partial_traversal(new_clade)
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
        cdef int node_idx
        cdef bint v_is_root
        cdef Stack visited_children, unvisited_children
        cdef int num_children_visited, num_children_unvisited, visits

        while traversal.length > 0:
            node_idx = traversal.pop()
            v = self.tree.nodes[node_idx]

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
            v = self.tree.nodes[i]
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

    cdef void add_edges_from_subsequence(self, int subsequence_index, int node_index):
        """
        Adds edges in self.graph from every node u_k in a subsequence to a node, but only if for all succeeding
        nodes u_j, j>k, there is no path u_k->u_j.
        """
        cdef int tree_node = subsequence_index
        cdef list_node * place_in_list
        cdef int last_variant_found = -MAXINT * self.direction
        cdef int variant_idx
        while True:
            place_in_list = self.subsequence.head[tree_node]
            while place_in_list != NULL:
                variant_idx = place_in_list.value
                if variant_idx * self.direction > last_variant_found * self.direction:
                    self.add_edge(variant_idx, node_index)
                    last_variant_found = variant_idx
                place_in_list = place_in_list.next
            if self.tree.nodes[tree_node].first_in is NULL:
                break
            tree_node = self.tree.nodes[tree_node].first_in.u.index

    cdef tuple[Stack, Stack] get_visited_children(self, node* v):
        """
        Separates the children of node v into those with and without variant carriers as descendants.
        """
        cdef Stack visited_children = Stack()
        cdef Stack unvisited_children = Stack()
        cdef int child
        cdef edge* e = v.first_out
        while e is not NULL:
            child = e.v.index
            e = e.next_out
            if self.times_visited.get_element(child) > 0:
                visited_children.push(child)
            else:
                unvisited_children.push(child)
        return visited_children, unvisited_children

cpdef int[:] combine_cliques(DiGraph forward_graph, DiGraph backward_graph):
    """
    Finds sequences u<v<...<w connected via edges (u,v),(v,...,w) in the forward graph and (w,...,v),(v,u) in the
    backward graph. Collapses these into a single node and returns an array of node assignments.
    """
    cdef int num_nodes = backward_graph.maximum_number_of_nodes
    cdef int[:] result = np.arange(num_nodes, dtype=np.intc)
    cdef int node_index
    cdef int neighbor_index
    cdef int neighbor_of_neighbor
    cdef IntegerSet neighbors = IntegerSet(forward_graph.maximum_number_of_nodes)
    cdef node * current_node
    cdef node * backward_node
    cdef edge * current_edge
    cdef edge * back_edge
    for node_index in range(num_nodes):
        if not forward_graph.is_node[node_index]:
            continue
        assert backward_graph.is_node[node_index]

        # If some neighbor of the current node has a back-edge to the current node as well, then it must be the first
        # neighbor due to the order in which edges are added (last in, first out)
        current_edge = forward_graph.nodes[node_index].first_out
        if current_edge is NULL:
            continue
        neighbor_index = current_edge.v.index
        assert backward_graph.is_node[neighbor_index]

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

cdef edge* add_nontransitive_edge(DiGraph graph, int u_idx, int v_idx, int skip_node):
    """
    Adds an edge between u and v if there is not already a path u, w_1,...,w_n, v with u < w_k < v or u > w_k > v.
    Skips over skip_node when searching.
    :return: the edge, or NULL if a path was found
    """
    # Search for descendants of u
    cdef int direction = 1 if u_idx < v_idx else -1
    cdef edge * e
    cdef int node
    cdef Stack nodes_to_visit = Stack()
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
    cdef int num_nodes = max(forward_reduction.maximum_number_of_nodes, backward_reduction.maximum_number_of_nodes)
    cdef IntegerSet reachable_in_two_hops = IntegerSet(num_nodes)
    cdef int node_index
    cdef node * current_node
    cdef edge * out_edge
    cdef DiGraph result = DiGraph(num_nodes, forward_reduction.number_of_edges + backward_reduction.number_of_edges)

    for node_index in range(num_nodes):
        # Set of nodes that is reachable in two hops from this one
        reachable_in_two_hops.clear()
        if forward_reduction.is_node[node_index]:
            search_two_hops(reachable_in_two_hops, forward_reduction, backward_reduction, node_index)
        if backward_reduction.is_node[node_index]:
            search_two_hops(reachable_in_two_hops, backward_reduction, forward_reduction, node_index)

        # Add neighbors that aren't reachable in two hops
        if forward_reduction.is_node[node_index]:
            current_node = forward_reduction.nodes[node_index]
            add_nonredundant_neighbors(result, current_node, reachable_in_two_hops)
        if backward_reduction.is_node[node_index]:
            current_node = backward_reduction.nodes[node_index]
            add_nonredundant_neighbors(result, current_node, reachable_in_two_hops)

    return result

# Subroutines of reduction_union
cdef void search_two_hops(IntegerSet result, DiGraph first_graph, DiGraph second_graph, int starting_node_index):
    """
    Searches from a starting node u to find nodes w such that for some v, (u,v) is an edge of first_graph and (v,w)
    is an edge of second_graph.
    """
    cdef node * starting_node = first_graph.nodes[starting_node_index]
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

cdef void search_two_hops_backward(IntegerSet result, DiGraph first_graph, DiGraph second_graph, int starting_node_index):
    """
    Searches from a starting node u to find nodes w such that for some v, (v,u) is an edge of first_graph and (w,v)
    is an edge of second_graph.
    """
    cdef node * starting_node = first_graph.nodes[starting_node_index]
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
    return graph, sample_indices, variant_indices

cpdef tuple get_graph_statistics(str brick_graph_dir):
    """
    Get merged graph statistics from directory of brick graph partitions.
    :param brick_graph_dir: location of brick graph partitions
    :return: num_samples, number_of_nodes, number_of_edges
    """
    cdef int number_of_nodes = 0
    cdef int number_of_edges = 0
    cdef list files = os.listdir(brick_graph_dir)
    for f in files:
        with h5py.File(f, 'r') as f:
            num_samples = f['sample_indices'].shape[0]
            number_of_nodes += f.attrs['n']
            number_of_edges += f['data'].shape[0]
    number_of_nodes -= num_samples * (len(files)-1)
    return num_samples, number_of_nodes, number_of_edges

cdef add_neighbors(DiGraph graph_to_modify, node* v, node_ids):
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


cpdef tuple merge_brick_graphs(str brick_graph_dir):
    """
    Merge multiple brick graphs with shared sample nodes and other nodes disjoint.
    :param brick_graph_dir: location of brick graph partitions
    :return: merged graph, variant indices, index mapping
    """
    num_samples, number_of_nodes, number_of_edges = get_graph_statistics(brick_graph_dir) # get statistics to initialize DiGraph object
    cdef DiGraph result = DiGraph(number_of_nodes, number_of_edges)
    cdef DiGraph graph
    cdef int i
    cdef int u, v
    cdef int non_sample_counter = num_samples
    cdef int sample_counter
    cdef int[:] new_node_ids
    cdef list variant_indices = []
    cdef list index_mapping = []
    cdef list files = os.listdir(brick_graph_dir)

    ind_arr = np.array([int(f.split('_')[0]) for f in files])
    order = ind_arr.argsort()
    files = np.array(files)[order].tolist() # sort files by index

    for f in files:

        graph, samples, variants = read_brick_graph_h5(f'{brick_graph_dir}/{f}')

        # Get new node ids corresponding to the merged graph
        sample_counter = 0
        new_node_ids = np.zeros(graph.number_of_nodes, dtype=np.intc)
        for i in range(graph.number_of_nodes):
            if i in samples:
                new_node_ids[i] = sample_counter
                sample_counter += 1
            else:
                new_node_ids[i] = non_sample_counter
                non_sample_counter += 1
        for var in variants:
            variant_indices.append(new_node_ids[var])
        index_mapping.append(new_node_ids)

        # Add edges from graph to result while preserving the order of the parent nodes for each child node
        for node_idx in range(graph.number_of_nodes):
            if not graph.is_node[node_idx]:
                continue
            add_neighbors(result, graph.nodes[node_idx], new_node_ids)

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
    digraph.initialize_all_nodes()

    # add edges in order they were stored
    for i in range(len(rows)):
        digraph.add_edge(rows[i], cols[i])

    return digraph
