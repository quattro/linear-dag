cimport numpy as cnp
import numpy as np
from .data_structures cimport LinkedListArray, ModHeap, DiGraph

cdef class Trios:
    """
    Class to encapsulate trio-clique information.

    **Arguments:**

    - `n`: Number of nodes

    **Attributes:**

    - `parents`: parents[r,p] is parent p of trio r for p in [0, 1]
    - `child`: child[r] is the child of trio r
    - `clique`: clique[r] is the clique to which trio r belongs
    - `neighbors`: neighbors[r,p] is the adjacent trio s, s.t. child[r]==child[s] & parent[r,p]==parent[s, 1-p],
        or -1 if none such exists
    - `cliqueRows`: lists of rows r s.t. clique[r]==c
    - `num_cliques`: number of cliques so far (for creating new ones)
    - `num_nodes`: number of nodes so far (for creating new ones)
    """
    cdef int n
    cdef int[:, :] parents
    cdef int[:] child
    cdef int[:] clique
    cdef int[:, :] neighbors
    cdef int[:, :] edge_list
    # cdef DiGraph brick_graph # TODO
    cdef LinkedListArray clique_rows
    cdef ModHeap clique_size_heap
    cdef int num_cliques
    cdef int num_nodes
    cdef int num_edges
    cdef int num_trios

    def __init__(self, int n):
        cnp.import_array()  # Necessary for initializing the C API
        self.n = n
        # self.brick_graph = DiGraph(n)
        self.parents = -np.ones((n, 2), dtype=np.intc)
        self.child = -np.ones(n, dtype=np.intc)
        self.clique = -np.ones(n, dtype=np.intc)
        self.neighbors = -np.ones((n, 2), dtype=np.intc)
        self.edge_list = -np.ones((n, 2), dtype=np.intc)
        self.clique_rows = LinkedListArray(n)
        self.clique_size_heap = None
        self.num_cliques = 0
        self.num_nodes = 0
        self.num_edges = 0
        self.num_trios = 0

    def get_num(self):
        return self.n, self.num_cliques, self.num_nodes, self.num_trios, self.num_edges

    cpdef int max_clique(self):
        if self.clique_size_heap:
            return self.clique_size_heap.pop()
        else:
            return -1

    def get_heap(self):
        return self.clique_size_heap.act_heap, self.clique_size_heap.priority


    cpdef convert_matrix(self, cnp.ndarray indices, cnp.ndarray indptr):
            cdef int num_nodes = len(indptr) - 1
            cdef int parent, previous_parent, previous_row, count
            cdef int current_trio = -1
            # Iterate over child nodes, creating trios involving adjacent pairs of parents for that child
            for child in range(num_nodes):
                if indptr[child] == indptr[child + 1]:  # no parent exists
                    continue
                parent = indices[indptr[child]]

                # Add singleton edges to edge_list
                if indptr[child] + 1 == indptr[child + 1]:
                    self.add_edge(parent, child)
                    continue

                previous_row = -1
                for jj in range(indptr[child] + 1, indptr[child + 1]):
                    previous_parent = parent
                    parent = indices[jj]

                    current_trio += 1
                    if current_trio >= self.n:
                        raise ValueError('Exceeded allowed space for trios')

                    self.parents[current_trio, 0] =  previous_parent
                    self.parents[current_trio, 1] =  parent
                    self.child[current_trio] =  child

                    self.neighbors[current_trio, 0] = previous_row
                    if previous_row != -1:
                        self.neighbors[previous_row, 1] = current_trio
                    previous_row = current_trio

            self.num_trios = current_trio + 1
            self.num_nodes = len(indptr) - 1

            self.compute_cliques()
            self.collect_cliques()

            return self.edge_list[:self.num_edges,:]

    def add_trio(self, int row, int parent1, int parent2, int child, int clique, int left_neighbor,
                 int right_neighbor):
        # Assuming n is the current number of trios and is maintained correctly
        if row > self.n:
            raise MemoryError("Exceeded allocated space for trios.")

        # Update the class properties with the new trio information
        self.parents[row, 0] = parent1
        self.parents[row, 1] = parent2
        self.child[row] = child
        self.clique[row] = clique
        self.neighbors[row, 0] = left_neighbor
        self.neighbors[row, 1] = right_neighbor
        self.clique_rows.extend(clique, row)

    def add_trios(self, trios_data, index_offset):
        """
        trios_data: A list of tuples, each containing (parent1, parent2, child, clique, left_neighbor, right_neighbor) for a trio.
        """
        # Ensure there is enough space for new trios
        required_space = len(trios_data)
        if required_space > self.n:
            raise MemoryError("Exceeded allocated space for trios.")

        cdef Py_ssize_t row
        # Update class properties with new trios information
        for row, trio in enumerate(trios_data):
            parent1, parent2, child, clique, left_neighbor, right_neighbor = trio
            self.parents[row, 0] = parent1 - index_offset
            self.parents[row, 1] = parent2 - index_offset
            self.child[row] = child - index_offset
            self.clique[row] = clique - index_offset
            self.neighbors[row, 0] = left_neighbor - index_offset
            self.neighbors[row, 1] = right_neighbor - index_offset

        self.num_nodes = max(np.max(self.parents), np.max(self.child)) + 1

    cpdef void add_edge(self, int parent, int child):
        self.edge_list[self.num_edges, 0] = parent
        self.edge_list[self.num_edges, 1] = child
        self.num_edges += 1

    def get_cliques(self):
        return np.asarray(self.clique)

    cpdef void compute_cliques(self):
        _, unique_indices = np.unique(self.parents[:self.num_trios, :], axis=0, return_inverse=True)
        unique_indices = unique_indices.astype(np.intc)
        cdef int i
        for i in range(self.num_trios):
            self.clique[i] = unique_indices[i]

    cpdef collect_cliques(self):
        self.num_cliques = np.max(self.clique) + 1
        cdef int[:] what = np.arange(self.num_trios, dtype=np.intc)
        cdef int[:] where = np.arange(self.num_cliques, dtype=np.intc)
        cdef int[:] which = self.clique[:self.num_trios]

        self.clique_rows.assign(what, where, which)
        self.clique_size_heap = ModHeap(self.clique_rows.length)

    cpdef extract_trio(self, int trio_index):
        # Check if the index is within bounds
        if trio_index < 0 or trio_index >= self.n:
            raise ValueError("Trio index out of bounds")

        # Extract and return the trio data
        parent1 = self.parents[trio_index, 0]
        parent2 = self.parents[trio_index, 1]
        child = self.child[trio_index]
        clique = self.clique[trio_index]
        left_neighbor = self.neighbors[trio_index, 0]
        right_neighbor = self.neighbors[trio_index, 1]

        return parent1, parent2, child, clique, left_neighbor, right_neighbor

    cpdef void find_recombinations(self):
        cdef int c = self.max_clique()
        while c >= 0: # -1 when none remain
            self.factor_clique(c)
            c = self.max_clique()

    cpdef void factor_clique(self, int clique_index):
        cdef int p, i
        cdef int clique_size = self.clique_rows.length[clique_index]
        if clique_size <= 1:
            return
        cdef int[:] clique_rows = self.clique_rows.extract(clique_index)
        cdef bint[:, :] has_shared_duo_list = np.zeros((clique_size, 2), dtype=np.intc)
        cdef int new_node = self.num_nodes
        self.num_nodes += 1

        # Update neighboring trios
        for p in range(2):
            has_shared_duo_list[:, p] = self.update_trios(p, clique_rows, new_node)

        # Update neighbor lists; add edges for child nodes with no neighboring trios
        for i in range(clique_size):
            row = clique_rows[i]
            if has_shared_duo_list[i, 0] or has_shared_duo_list[i, 1]:
                self.bypass_trios(row)
            else:
                self.add_edge(new_node, self.child[row])

        self.collapse_clique(clique_index, new_node, clique_rows)


    cdef bint[:] update_trios(self, int p, int[:] rows, int new_node):
        cdef int nn = len(rows)
        # Identify rows with p-neighbors
        cdef bint[:] has_neighbor = np.zeros(nn, dtype=np.intc)
        if nn == 0:
            return has_neighbor
        cdef int i
        for i in range(nn):
            if self.neighbors[rows[i], p] >= 0:
                has_neighbor[i] = True
        if np.sum(has_neighbor) == 0:
            return has_neighbor

        # Identify neighbors and reassign their parent nodes
        cdef int[:] neighbors = np.zeros(np.sum(has_neighbor), dtype=np.intc)
        cdef int idx = 0
        for i in range(nn):
            if has_neighbor[i]:
                neighbors[idx] = self.neighbors[rows[i], p]
                self.parents[neighbors[idx], 1-p] = new_node
                idx += 1

        # Re-assign clique[neighbors]
        affected_cliques, which_affected_cliques = np.unique(np.take(self.clique, neighbors), return_inverse=True)
        for i in range(idx):
            self.clique[neighbors[i]] = which_affected_cliques[i] + self.num_cliques

        # Assign neighbors to cliqueRows[new cliques]
        self.clique_rows.assign(neighbors, self.num_cliques + np.arange(len(affected_cliques), dtype=np.intc), which_affected_cliques.astype(np.intc))

        # Update clique size heap
        for i in range(len(affected_cliques)):
            self.clique_size_heap.push(self.num_cliques + i, self.clique_rows.length[self.num_cliques + i])

        # Remove them from cliqueRows[old cliques]
        for i in range(len(affected_cliques)):
            self.clique_rows.remove_difference(self.num_cliques + i, affected_cliques[i])
            self.clique_size_heap.push(affected_cliques[i], self.clique_rows.length[affected_cliques[i]])

        self.num_cliques += len(affected_cliques)

        return has_neighbor

    cpdef bypass_trios(self, int t):
        cdef int left = self.neighbors[t, 0]
        cdef int right = self.neighbors[t, 1]
        if left >= 0:
            self.neighbors[left, 1] = right
        if right >= 0:
            self.neighbors[right, 0] = left

        # Set the adjacency of t to -1, indicating no adjacent trios
        self.neighbors[t, 1] = -1
        self.neighbors[t, 0] = -1

    cpdef collapse_clique(self, int c, int node, int[:] ts):

        # Update all trios in ts, except for the first one
        cdef int t
        for t in ts[1:]:
            self.parents[t, 0] = -1
            self.parents[t, 1] = -1
            self.child[t] = -1
            self.neighbors[t, 0] = -1
            self.neighbors[t, 1] = -1
            self.clique[t] = -1

        # Update the first trio
        cdef int t_first = ts[0]
        self.child[t_first] = node
        self.clique[t_first] = -1
        self.neighbors[t_first, 0] = -1
        self.neighbors[t_first, 1] = -1
        # Parents remain unchanged

        self.clique_rows.clear_list(c)  # eliminate the clique

    cpdef count_edges(self):
        cdef int trio_idx, p
        cdef int counter = 0

        # It doesn't matter if p is 0 or 1, choose p = 0
        p = 0

        # Iterate over each trio
        for trio_idx in range(self.n):
            if self.child[trio_idx] < 0:
                continue

            if self.neighbors[trio_idx, p] >= 0:
                counter += 1
            else:
                counter += 2

        return counter

    cpdef fill_edgelist(self):

        # It doesn't matter if p is 0 or 1, choose p = 0
        cdef int p = 0

        # Add either 1 or 2 edges for each non-null trio
        cdef int trio_idx
        for trio_idx in range(self.n):
            if self.child[trio_idx] < 0:
                continue  # Trio is null

            self.add_edge(self.parents[trio_idx, p], self.child[trio_idx])

            # If neighbor[1-p] is -1, it means that this parent-child duo doesn't appear elsewhere and must be added now
            if self.neighbors[trio_idx, 1-p] < 0:
                self.add_edge(self.parents[trio_idx, 1-p], self.child[trio_idx])

        return self.edge_list[:self.num_edges, :]

    cpdef int clique_size(self, int n):
        return self.clique_rows.length[n]

    def check_properties(self, int num_rows_to_check):
        # Flag to indicate if the property holds for all elements
        property_holds = True

        # First check: clique membership
        for i in range(min(len(self.clique), num_rows_to_check)):
            # Get the clique ID for the current element
            clique_id = self.clique[i]
            if clique_id == -1:
                continue

            # Check if i is contained in self.cliqueRows[clique_id]
            try:
                if i not in self.clique_rows.extract(clique_id):
                    property_holds = False
                    print(f"Property does not hold for element {i} in clique {clique_id}.")
                    break  # Exit early if a single violation is found
            except ValueError as ve:
                print(f"Error when evaluating property for element {i} in clique {clique_id}")
                return

        if not property_holds:
            print("Property does not hold for at least one element in self.clique.")
            return  # Exit the function if the first property check failed

        # Second check: neighbors consistency
        cdef int r, p, s
        for r in range(min(self.n, num_rows_to_check)):
            for p in range(2):
                s = self.neighbors[r, p]
                if s < 0:  # Ignore if there's no neighbor
                    continue

                if not (self.child[r] == self.child[s] and self.parents[r, p] == self.parents[s, 1 - p]):
                    property_holds = False
                    print(f"Neighbors property does not hold for row {r}, parent {p}, neighbor {s}.")
                    break  # Exit early if a single violation is found

            if not property_holds:
                break  # Exit the outer loop if a violation is found

        # Print a statement based on whether the property holds for all elements
        if property_holds:
            print("All properties hold for the Trios instance.")
        else:
            print("At least one property does not hold for the Trios instance.")
