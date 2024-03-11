cimport numpy as np
from libc.stdlib cimport free, malloc
import numpy as np
import heapq

# from .mod_heapc import ModHeap

cdef struct ListNode:
    ListNode* next
    int value

cdef int gcd(int a, int b):
    cdef int temp
    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a


cdef class Node:
    cdef public int priority
    cdef public int index

    def __init__(self, int priority, int index):
        self.priority = priority
        self.index = index

    def __lt__(self, Node other):
        return self.priority < other.priority

    def __int__(self):
        return self.index

cdef class ModHeap:
    cdef public list act_heap
    cdef int[:] priority
    cdef public int n

    def __init__(self, int[:] priority):
        self.n = len(priority)
        self.priority = np.copy(priority).astype(np.intc) # Copies the input array
        self.act_heap = self._create_heap(np.copy(self.priority))
        heapq.heapify(self.act_heap)
    cdef list _create_heap(self, int[:] priority):
        cdef int i
        cdef list heap = []
        for i in range(self.n):
            # Allocate a new Node and append it to the heap list
            node = Node(-priority[i], i)
            heap.append(node)
        heapq.heapify(heap)
        return heap

    cpdef push(self, int index, int priority):
        cdef Node node = Node(-priority, index)
        heapq.heappush(self.act_heap, node)
        self.priority[index] = priority

    cpdef int pop(self):
        cdef Node node = heapq.heappop(self.act_heap)
        while self.priority[node.index] != -node.priority:
            node = heapq.heappop(self.act_heap)
        self.priority[node.index] = 0
        return node.index

cdef class LinkedListArray:
    cdef ListNode** first  # Array of pointers to the first element of each linked list
    cdef ListNode** last  # Array of pointers to the last element of each linked list
    cdef int[:] length  # Length of each linked list
    cdef int n  # Number of linked lists

    def __cinit__(self, int n):
        self.n = n
        self.first = <ListNode**> malloc(n * sizeof(ListNode *))
        self.last = <ListNode**> malloc(n * sizeof(ListNode *))
        self.length = np.zeros(n, dtype=np.intc)
        if not self.first or not self.last:
            raise MemoryError("Could not allocate LinkedListArray.")
        for i in range(n):
            self.first[i] = NULL
            self.last[i] = NULL
            self.length[i] = 0

    def __dealloc__(self):
        # Free all linked lists
        for i in range(self.n):
            current_node = self.first[i]
            while current_node != NULL:
                next_node = current_node.next
                free(current_node)
                current_node = next_node
        # Free the arrays
        free(self.first)
        free(self.last)

    cdef void extend(self, int n, int value):
        # Ensure we're working within bounds
        if n < 0 or n >= self.n:
            raise ValueError("LinkedList index out of bounds.")

        # Create new node
        new_node = <ListNode *> malloc(sizeof(ListNode))
        if not new_node:
            raise MemoryError("Could not allocate ListNode.")
        new_node.value = value
        new_node.next = NULL

        # Append to the list
        if self.last[n] == NULL:  # List is empty
            self.first[n] = new_node
            self.last[n] = new_node
        else:
            # Check that value is greater than the last value in the list
            if value <= self.last[n].value:
                free(new_node)  # Important to avoid memory leak
                raise ValueError("New value must be greater than the last value in the list.")
            self.last[n].next = new_node
            self.last[n] = new_node

        self.length[n] += 1

    cdef void remove(self, int n, ListNode * node, ListNode * predecessor):
        # Check if the node to remove is the first node in the list
        if predecessor == NULL:
            self.first[n] = node.next
            if self.first[n] == NULL:  # List becomes empty
                self.last[n] = NULL
        else:
            predecessor.next = node.next
            # If 'node' is the last node, update 'last' pointer of the list
            if node.next == NULL:
                self.last[n] = predecessor

        # Decrement the length of the list
        if n >= 0:  # Ensure list_index is valid
            self.length[n] -= 1

        # Finally, free the node
        free(node)

    cpdef assign(self, int[:] what, int[:] where, int[:] which):
        # Sort 'what' to keep linked lists sorted
        cdef int[:] order = np.argsort(what).astype(np.int32)

        # Check that existing lists are empty for each 'n' in 'where'
        cdef int i, idx
        for i in range(len(where)):
            idx = where[i]
            if idx >= self.n:
                raise ValueError("Length of linked list array exceeded.")
            if self.first[idx] is not NULL:
                raise ValueError(f"List at index {idx} is not empty.")

        # Iterate over 'which' in sorted order and extend lists
        for i in range(len(order)):
            self.extend(where[which[order[i]]], what[order[i]])

    cpdef remove_difference(self, int n, int m):
        if n == m:
            self.clear_list(n)
            return

        cdef ListNode* node_n = self.first[n]
        cdef ListNode* node_m = self.first[m]
        cdef ListNode* next_m
        cdef ListNode* prev_m = NULL
        while node_m != NULL and node_n != NULL:
            if node_n.value < node_m.value:
                # Move to the next node in list 'n'
                node_n = node_n.next
            elif node_n.value > node_m.value:
                # Move to the next node in list 'm'
                prev_m = node_m
                node_m = node_m.next
            else:  # node_n.value == node_m.value
                # Remove the current node from list 'm' and move to the next
                next_m = node_m.next
                self.remove(m, node_m, prev_m)
                node_m = next_m

    cpdef clear_list(self, int n):
        cdef ListNode * node = self.first[n]
        cdef ListNode * next_node
        while node != NULL:
            next_node = node.next
            free(node)  # assuming you've allocated nodes with malloc
            node = next_node
        self.first[n] = NULL
        self.length[n] = 0

    def extract(self, int n):
        # Check if the index n is within bounds
        if n < 0 or n >= self.n:
            raise ValueError("Index out of bounds.")

        # Check if the linked list at index n is empty
        if self.first[n] is NULL:
            return np.array([], dtype=np.intc)  # Return an empty NumPy array for an empty list

        # Calculate the length of the n-th linked list
        cdef int list_length = self.length[n]

        # Initialize a NumPy array of the appropriate length
        cdef np.ndarray result = np.empty(list_length, dtype=np.intc)

        # Traverse the linked list and fill the NumPy array
        cdef ListNode * current_node = self.first[n]
        cdef int i = 0
        while current_node is not NULL:
            result[i] = current_node.value
            current_node = current_node.next
            i += 1

        return result


cdef class Trios:
    """
    Class to encapsulate trio-clique information.

    **Arguments:**

    - `n`: Number of nodes

    **Attributes:**

    - `parents`: parents[r,p] is parent p of trio r for p in [0, 1]
    - `child`: child[r] is the child of trio r
    - `weight`: weight[r] is the edge weight of parent[r,p] -> child[r] for p in [0, 1]
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
    cdef int[:,:] weight
    cdef int[:] clique
    cdef int[:, :] neighbors
    cdef int[:, :] edge_list
    cdef LinkedListArray clique_rows
    cdef ModHeap clique_size_heap
    cdef int num_cliques
    cdef int num_nodes
    cdef int num_edges
    cdef int num_trios
    # cdef int* clique_size
    # cdef object* clique_size

    def __init__(self, int n):
        self.n = n
        self.parents = -np.ones((n, 2), dtype=np.intc)
        self.child = -np.ones(n, dtype=np.intc)
        self.weight = np.zeros((n, 2), dtype=np.intc)
        self.clique = -np.ones(n, dtype=np.intc)
        self.neighbors = -np.ones((n, 2), dtype=np.intc)
        self.edge_list = -np.ones((n, 3), dtype=np.intc)
        self.clique_rows = LinkedListArray(n)
        self.clique_size_heap = None
        self.num_cliques = 0
        self.num_nodes = 0
        self.num_edges = 0
        self.num_trios = 0

    def get_num(self):
        return (self.n, self.num_cliques, self.num_nodes, self.num_trios, self.num_edges)
    # Methods
    cpdef int max_clique(self):
        cdef int max_clique = self.clique_size_heap.pop()
        # cdef int max_clique = np.argmax(self.clique_rows.length)
        if self.clique_size(max_clique) > 1:
            return max_clique
        else:
            return -1

    def extract_heap(self):
        return (self.clique_size_heap.act_heap, self.clique_size_heap.priority)

    cpdef convert_matrix_ranked(self, np.ndarray weights, np.ndarray indices, np.ndarray indptr, int num_nodes,
                         np.ndarray rank):

        cdef int parent, weight, previous_weight, previous_parent, previous_trio, jj
        cdef int current_trio = -1

        # Iterate over child nodes, creating trios involving adjacent pairs of parents for that child
        for child in range(num_nodes - 1):
            # stable ordering, s.t. consecutive parent indices with same rank are adjacent
            ordered_indptrs = indptr[child] + np.argsort(rank[indptr[child]: indptr[child + 1]], kind='stable')
            previous_parent = -1
            previous_trio = -1
            for jj in ordered_indptrs:
                parent = indices[jj]
                weight = int(weights[jj])
                if previous_parent == -1:
                    previous_parent = parent
                    previous_weight = weight
                    continue

                # Add a new row to the trio list
                current_trio += 1
                if current_trio >= self.n:
                    raise ValueError('Exceeded allowed space for trios')
                self.parents[current_trio, 0] = previous_parent
                self.parents[current_trio, 1] = parent
                self.child[current_trio] = child
                self.weight[current_trio, 0] = previous_weight
                self.weight[current_trio, 1] = weight
                previous_weight = weight
                previous_parent = parent

                # Link together neighboring rows
                if previous_trio == -1:
                    previous_trio = current_trio
                    continue
                self.neighbors[current_trio, 0] = previous_trio
                self.neighbors[previous_trio, 1] = current_trio
                previous_trio = current_trio

            # Add singleton edges (child node having in-degree 1) to edge_list
            if previous_trio == -1 and previous_parent >= 0:
                self.edge_list[self.num_edges, 0] = parent
                self.edge_list[self.num_edges, 1] = child
                self.edge_list[self.num_edges, 2] = weight
                self.num_edges += 1


        self.num_trios = current_trio + 1
        self.num_nodes = np.max(self.parents[:, 1]) + 1

        # Assign cliques based on parents and weights
        self.compute_cliques()

        # Collect rows belonging to each clique in self.clique_rows
        self.collect_cliques()

        return self.edge_list[:self.num_edges, :]

    cdef compute_cliques(self):
        # Array storing parents and coprime edge weights
        cdef int[:, :] trio_info = np.empty((self.num_trios, 4), dtype=np.intc)

        cdef int i, weight_gcd
        for i in range(self.num_trios):

            # Greatest common divisor of the edge weights
            weight_gcd = gcd(self.weight[i, 0], self.weight[i, 1])

            trio_info[i, 0] = self.parents[i, 0]
            trio_info[i, 1] = self.parents[i, 1]
            trio_info[i, 2] = self.weight[i, 0] // weight_gcd
            trio_info[i, 3] = self.weight[i, 1] // weight_gcd

        # Clique index of each trio
        _, unique_indices = np.unique(trio_info, axis=0, return_inverse=True)

        self.clique = unique_indices.astype(np.intc)


    cpdef convert_matrix(self, np.ndarray data, np.ndarray indices, np.ndarray indptr, int num_nodes):

            cdef int parent, weight, previous_parent, previous_row, count
            cdef int current_trio = -1
            # Iterate over child nodes, creating trios involving adjacent pairs of parents for that child
            for child in range(num_nodes - 1):
                last_row_dict = {}
                last_parent_dict = {}
                num_rows_dict = {}
                for jj in range(indptr[child], indptr[child + 1]):
                    parent = indices[jj]
                    weight = int(data[jj])
                    if weight not in last_parent_dict:
                        last_parent_dict[weight] = parent
                        num_rows_dict[weight] = 1
                        continue

                    num_rows_dict[weight] += 1

                    # Add a new row to the trio list
                    current_trio += 1
                    if current_trio >= self.n:
                        raise ValueError('Exceeded allowed space for trios')

                    previous_parent = last_parent_dict[weight]
                    last_parent_dict[weight] = parent
                    self.parents[current_trio, 0] =  previous_parent
                    self.parents[current_trio, 1] =  parent
                    self.child[current_trio] =  child
                    self.weight[current_trio, 0] = weight
                    self.weight[current_trio, 1] = weight

                    # Link together neighboring rows
                    if weight not in last_row_dict:
                        last_row_dict[weight] = current_trio
                        continue

                    previous_row = last_row_dict[weight]
                    last_row_dict[weight] = current_trio
                    self.neighbors[current_trio, 0] = previous_row
                    self.neighbors[previous_row, 1] = current_trio

                # Add singleton edges to edge_list
                for weight, count in num_rows_dict.items():
                    if count > 1:
                        continue
                    self.edge_list[self.num_edges,0] = last_parent_dict[weight]
                    self.edge_list[self.num_edges,1] = child
                    self.edge_list[self.num_edges,2] = weight
                    self.num_edges += 1

            # Assign cliques
            current_trio += 1
            # _, indices = np.unique(self.parents, axis=0, return_inverse=True)
            # self.clique = indices.astype(np.intc)

            self.num_trios = current_trio
            self.num_nodes = np.max(self.parents[:,1]) + 1

            # Assign cliques based on parents and weights
            self.compute_cliques()

            # Collect rows belonging to each clique in self.clique_rows
            self.collect_cliques()

            return self.edge_list[:self.num_edges,:]

    def add_trio(self, int row, int parent1, int parent2, int child, int weight, int clique, int left_neighbor,
                 int right_neighbor):
        # Assuming n is the current number of trios and is maintained correctly
        if row > self.n:
            raise MemoryError("Exceeded allocated space for trios.")

        # Update the class properties with the new trio information
        self.parents[row, 0] = parent1
        self.parents[row, 1] = parent2
        self.child[row] = child
        self.weight[row, 0] = weight
        self.weight[row, 1] = weight
        self.clique[row] = clique
        self.neighbors[row, 0] = left_neighbor
        self.neighbors[row, 1] = right_neighbor
        self.clique_rows.extend(clique, row)

    def add_trios(self, trios_data, index_offset):
        """
        trios_data: A list of tuples, each containing (parent1, parent2, child, weight, clique, left_neighbor, right_neighbor) for a trio.
        """
        # Ensure there is enough space for new trios
        required_space = len(trios_data)
        if required_space > self.n:
            raise MemoryError("Exceeded allocated space for trios.")

        cdef Py_ssize_t row
        # Update class properties with new trios information
        for row, trio in enumerate(trios_data):
            parent1, parent2, child, weight, clique, left_neighbor, right_neighbor = trio
            self.parents[row, 0] = parent1 - index_offset
            self.parents[row, 1] = parent2 - index_offset
            self.child[row] = child - index_offset
            self.weight[row, 0] = weight
            self.weight[row, 1] = weight
            self.clique[row] = clique - index_offset
            self.neighbors[row, 0] = left_neighbor - index_offset
            self.neighbors[row, 1] = right_neighbor - index_offset

        self.num_nodes = max(np.max(self.parents), np.max(self.child)) + 1

    def get_cliques(self):
        return np.asarray(self.clique)

    cpdef collect_cliques(self):
        self.num_cliques = np.max(self.clique) + 1
        cdef int[:] what = np.arange(self.num_trios, dtype=np.intc)
        cdef int[:] where = np.arange(self.num_cliques, dtype=np.intc)
        cdef int[:] which = self.clique[:self.num_trios]

        # Use 'assign' to populate 'cliqueRows'
        self.clique_rows.assign(what, where, which)

        # Populate 'clique_size'
        self.clique_size_heap = ModHeap(self.clique_rows.length)

    cpdef extract_trio(self, int trio_index):
        # Check if the index is within bounds
        if trio_index < 0 or trio_index >= self.n:
            raise ValueError("Trio index out of bounds")

        # Extract and return the trio data
        parent1 = self.parents[trio_index, 0]
        parent2 = self.parents[trio_index, 1]
        child = self.child[trio_index]
        weight1 = self.weight[trio_index, 0]
        weight2 = self.weight[trio_index, 1]
        clique = self.clique[trio_index]
        left_neighbor = self.neighbors[trio_index, 0]
        right_neighbor = self.neighbors[trio_index, 1]

        return parent1, parent2, child, weight1, weight2, clique, left_neighbor, right_neighbor

    cpdef find_recombinations(self):

        cdef int c = self.max_clique()
        while c >= 0:
            self.factor_clique(c)
            c = self.max_clique()

    cpdef factor_clique(self, int c):
        cdef int p, i
        cdef int nr = self.clique_rows.length[c]

        # Look up the trios belonging to the clique 'c'
        cdef int[:] clique_rows = self.clique_rows.extract(c).astype(np.intc)

        # Edge weights of the clique and of each row
        cdef int g = gcd(self.weight[clique_rows[0],0], self.weight[clique_rows[0],1])
        cdef int[:] clique_weight = np.empty(2, dtype=np.intc)
        for p in range(2):
            clique_weight[p] = self.weight[clique_rows[0],p] // g
        cdef int[:] row_weight = np.empty(nr, dtype=np.intc)
        for i in range(nr):
            row_weight[i] = self.weight[clique_rows[i],0] // clique_weight[0]
            assert(self.weight[clique_rows[i],1] == row_weight[i] * clique_weight[1])

        # Prepare storage for the boolean vectors returned by update_trios
        bool_array = np.zeros((nr, 2), dtype=np.intc)
        cdef bint[:, :] has_shared_duo_list = bool_array

        # Call update_trios for p=0 and p=1, store the boolean vectors
        for p in range(2):
            has_shared_duo = self.update_trios(p, clique_rows, self.num_nodes, clique_weight[p])
            has_shared_duo_list[:, p] = has_shared_duo

        # Update neighbor lists; add edges for child nodes with no neighboring trios
        for i in range(nr):
            row = clique_rows[i]
            if has_shared_duo_list[i, 0] and has_shared_duo_list[i, 1]:
                # Both parent-child duos have neighbors: bypass them in neighbor list
                self.bypass_trios(row)
            elif has_shared_duo_list[i, 0] or has_shared_duo_list[i, 1]:
                # One has a neighbor: truncate end of neighbor list
                self.remove_adjacent(row, 0 if has_shared_duo_list[i, 0] else 1)
            else:
                self.edge_list[self.num_edges,0] = self.num_nodes
                self.edge_list[self.num_edges,1] = self.child[row]
                self.edge_list[self.num_edges,2] = row_weight[i]
                self.num_edges += 1

        # Call collapse_clique on 'c', 'n', and 'rows'
        self.collapse_clique(c, self.num_nodes, clique_rows, clique_weight)
        self.num_nodes += 1

    cdef bint[:] update_trios(self, int p, int[:] rows, int new_node, int edge_weight_divisor):
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
                self.weight[neighbors[idx], 1-p] = self.weight[neighbors[idx], 1-p] // edge_weight_divisor
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
        # Assuming t, l, and r are 0-based indices
        cdef int left = self.neighbors[t, 0]
        cdef int right = self.neighbors[t, 1]

        if left >= 0 and right >= 0:  # Check if l and r are valid indices
            self.neighbors[left, 1] = right  # Update right_adjacent of l to r
            self.neighbors[right, 0] = left  # Update left_adjacent of r to l

        # Set the adjacency of t to -1, indicating no adjacent trios
        self.neighbors[t, 1] = -1
        self.neighbors[t, 0] = -1

    cpdef remove_adjacent(self, int t, int p):
        cdef int x
        x = self.neighbors[t, p]
        self.neighbors[x, 1-p] = -1

    cpdef collapse_clique(self, int c, int node, int[:] ts, int[:] clique_weight):

        # Update all trios in ts, except for the first one
        cdef int t
        for t in ts[1:]:
            self.parents[t, 0] = -1
            self.parents[t, 1] = -1
            self.child[t] = -1
            self.weight[t, 0] = 0
            self.weight[t, 1] = 0
            self.neighbors[t, 0] = -1
            self.neighbors[t, 1] = -1
            self.clique[t] = -1

        # Update the first trio
        cdef int t_first = ts[0]
        self.child[t_first] = node
        self.weight[t_first, :] = clique_weight
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

            # Add edge entry for parent[p] to child with the corresponding weight
            self.edge_list[self.num_edges, 0] = self.parents[trio_idx, p]
            self.edge_list[self.num_edges, 1] = self.child[trio_idx]
            self.edge_list[self.num_edges, 2] = self.weight[trio_idx, p]
            self.num_edges += 1

            # If neighbor[1-p] is -1, it means that this parent-child duo doesn't appear elsewhere and must be added now
            if self.neighbors[trio_idx, 1-p] < 0:
                self.edge_list[self.num_edges, 0] = self.parents[trio_idx, 1-p]
                self.edge_list[self.num_edges, 1] = self.child[trio_idx]
                self.edge_list[self.num_edges, 2] = self.weight[trio_idx, 1-p]
                self.num_edges += 1

        return self.edge_list[:self.num_edges, :]

    cpdef clique_size(self, n):
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
