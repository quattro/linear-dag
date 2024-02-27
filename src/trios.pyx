from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
import cython
cimport cython

cdef struct ListNode:
    ListNode* next
    int value

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
                raise ValueError(f"Length of linked list array exceeded.")
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
    cdef int n
    cdef int[:,:] parents # parents[r,p] is parent p of trio r for p in [0,1]
    cdef int[:] child # child[r] is the child of trio r
    cdef int[:] weight # weight[r] is the edge weight of parent[r,p] -> child[r] for p in [0,1]
    cdef int[:] clique # clique[r] is the clique to which trio r belongs
    cdef int[:,:] neighbors # neighbors[r,p] is the adjacent trio s s.t. child[r]==child[s] and parent[r,p]==parent[s,1-p], or -1 if none such exists
    cdef LinkedListArray cliqueRows # lists of rows r s.t. clique[r]==c
    cdef int nCliques # number of cliques so far (for creating new ones)
    cdef int nNodes # number of nodes so far (for creating new ones)
    cdef int[:,:] edgeList # edgeList[e,0]: parent node in the output graph; edgeList[e,1]: child node; edgeList[e,2]: weight
    cdef int nEdges # number of edges so far (for creating new ones)


    def __init__(self, int n):
        self.n = n
        self.parents = np.zeros((n, 2), dtype=np.intc) 
        self.child = np.zeros(n, dtype=np.intc) 
        self.weight = np.zeros(n, dtype=np.intc)  
        self.clique = np.zeros(n, dtype=np.intc)  
        self.neighbors = np.zeros((n, 2), dtype=np.intc)
        self.cliqueRows = LinkedListArray(n)
        self.nCliques = 0
        self.nNodes = 0
        self.edgeList = np.zeros((n, 3), dtype=np.intc)
        self.nEdges = 0

    # Methods
    cpdef maxClique(self):
        cdef int maxClique = np.argmax(self.cliqueRows.length)
        if self.cliqueRows.length[maxClique] > 1:
            return maxClique
        else:
            return -1

    def add_trio(self, int row, int parent1, int parent2, int child, int weight, int clique, int left_neighbor,
                 int right_neighbor):
        # Assuming n is the current number of trios and is maintained correctly
        if row > self.n:
            raise MemoryError("Exceeded allocated space for trios.")

        # Update the class properties with the new trio information
        self.parents[row, 0] = parent1
        self.parents[row, 1] = parent2
        self.child[row] = child
        self.weight[row] = weight
        self.clique[row] = clique
        self.neighbors[row, 0] = left_neighbor
        self.neighbors[row, 1] = right_neighbor
        self.cliqueRows.extend(clique, row)
    

    def add_trios(self, trios_data, index_offset):
        """
        trios_data: A list of tuples, each containing (parent1, parent2, child, weight, clique, left_neighbor, right_neighbor) for a trio.
        """
        # Ensure there is enough space for new trios
        required_space = len(trios_data)
        if required_space > self.n:
            raise MemoryError("Exceeded allocated space for trios.")

        # Update class properties with new trios information
        for row, trio in enumerate(trios_data):
            parent1, parent2, child, weight, clique, left_neighbor, right_neighbor = trio
            self.parents[row, 0] = parent1 - index_offset
            self.parents[row, 1] = parent2 - index_offset
            self.child[row] = child - index_offset
            self.weight[row] = weight
            self.clique[row] = clique - index_offset
            self.neighbors[row, 0] = left_neighbor - index_offset
            self.neighbors[row, 1] = right_neighbor - index_offset

        self.nNodes = max(np.max(self.parents), np.max(self.child))+1

    cpdef collect_cliques(self):
        self.nCliques = np.max(self.clique) + 1
        cdef int[:] what = np.arange(self.n, dtype=np.intc)
        cdef int[:] where = np.arange(self.nCliques, dtype=np.intc)
        cdef int[:] which = self.clique

        # Use 'assign' to populate 'cliqueRows'
        self.cliqueRows.assign(what, where, which)

        
    cpdef extract_trio(self, int trio_index):
        # Check if the index is within bounds
        if trio_index < 0 or trio_index >= self.n:
            raise ValueError("Trio index out of bounds")

        # Extract and return the trio data
        parent1 = self.parents[trio_index, 0]
        parent2 = self.parents[trio_index, 1]
        child = self.child[trio_index]
        weight = self.weight[trio_index]
        clique = self.clique[trio_index]
        left_neighbor = self.neighbors[trio_index, 0]
        right_neighbor = self.neighbors[trio_index, 1]

        return (parent1, parent2, child, weight, clique, left_neighbor, right_neighbor)

    def extract_edgelist(self):
        return np.asarray(self.edgeList[:self.nEdges, :])

    cpdef factor_clique(self, int c):
        cdef int nr = self.cliqueRows.length[c]
        # Look up the trios belonging to the clique 'c'
        cdef int[:] clique_rows = self.cliqueRows.extract(c).astype(np.intc)
        # Prepare storage for the boolean vectors returned by update_triolist
        bool_array = np.zeros((nr, 2), dtype=np.intc)
        cdef bint[:, :] has_shared_duo_list = bool_array

        # Call update_triolist for p=0 and p=1, store the boolean vectors
        for p in range(2):
            has_shared_duo = self.update_trios(p, clique_rows, self.nNodes)
            has_shared_duo_list[:,p] = has_shared_duo

        # Process rows based on the has_shared_duo vectors
        for i in range(nr):
            row = clique_rows[i]
            if not has_shared_duo_list[i,0] and not has_shared_duo_list[i,1]:
                # Add to the edgelist for rows with [false false]
                self.update_edgelist(self.nNodes, self.child[row], self.weight[row])
            elif has_shared_duo_list[i,0] and has_shared_duo_list[i,1]:
                # Call bypass_trios for rows with [true true]
                self.bypass_trios(row)
            else:
                # Call remove_adjacent for other rows
                self.remove_adjacent(row, 0 if has_shared_duo_list[i,0] else 1)

        # Call collapse_clique on 'c', 'n', and 'rows'
        self.collapse_clique(c, self.nNodes, clique_rows)

        self.nNodes += 1
    cdef bint[:] update_trios(self, int p, int[:] rows, int newNode):
        nn = len(rows)
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
                self.parents[neighbors[idx], 1-p] = newNode
                idx += 1

        # Re-assign clique[neighbors]
        affected_cliques, which_affected_cliques = np.unique(np.take(self.clique, neighbors), return_inverse=True)
        for i in range(idx):
            self.clique[neighbors[i]] = which_affected_cliques[i] + self.nCliques

        # Assign neighbors to cliqueRows[new cliques]
        self.cliqueRows.assign(neighbors, self.nCliques + np.arange(len(affected_cliques), dtype=np.intc), which_affected_cliques.astype(np.intc))

        # Remove them from cliqueRows[old cliques]
        for i in range(len(affected_cliques)):
            self.cliqueRows.remove_difference(self.nCliques + i, affected_cliques[i])

        self.nCliques += len(affected_cliques)

        return has_neighbor
        
    cpdef update_edgelist(self, int parent, int child, int weight):
        # Update the edgelist with the trio data
        self.edgeList[self.nEdges, 0] = parent
        self.edgeList[self.nEdges, 1] = child
        self.edgeList[self.nEdges, 2] = weight
        self.nEdges += 1

    cpdef bypass_trios(self, int t):
        # Assuming t, l, and r are 0-based indices
        cdef int l = self.neighbors[t,0]
        cdef int r = self.neighbors[t,1]

        if l >= 0 and r >= 0:  # Check if l and r are valid indices
            self.neighbors[l,1] = r  # Update right_adjacent of l to r
            self.neighbors[r,0] = l  # Update left_adjacent of r to l

        # Set the adjacency of t to -1, indicating no adjacent trios
        self.neighbors[t,1] = -1
        self.neighbors[t,0] = -1

    cpdef remove_adjacent(self, int t, int p):
        cdef int x
        x = self.neighbors[t,p]
        self.neighbors[x,1-p] = -1
    
    cpdef collapse_clique(self, int c, int node, int[:] ts):

        # Update all trios in ts, except for the first one
        cdef int t
        for t in ts[1:]:
            self.parents[t,0] = -1
            self.parents[t,1] = -1
            self.child[t] = -1
            self.weight[t] = -1
            self.neighbors[t,0] = -1
            self.neighbors[t,1] = -1
            self.clique[t] = -1

        # Update the first trio
        cdef int t_first = ts[0]
        self.child[t_first] = node
        self.weight[t_first] = 1
        self.clique[t_first] = -1
        self.neighbors[t_first,0] = -1
        self.neighbors[t_first,1] = -1
        # Note: parents for the first trio remain unchanged

        self.cliqueRows.clear_list(c) # eliminate the clique

    def cliqueSize(self):
        return self.cliqueRows.length

    def numNodes(self):
        return self.nNodes

    def check_properties(self, int nRowsToCheck):
        # Flag to indicate if the property holds for all elements
        property_holds = True

        # First check: clique membership
        for i in range(min(len(self.clique), nRowsToCheck)):
            # Get the clique ID for the current element
            clique_id = self.clique[i]
            if clique_id == -1:
                continue

            # Check if i is contained in self.cliqueRows[clique_id]
            try:
                if i not in self.cliqueRows.extract(clique_id):
                    property_holds = False
                    print(f"Property does not hold for element {i} in clique {clique_id}.")
                    break  # Exit early if a single violation is found
            except:
                print(f"Error when evaluating property for element {i} in clique {clique_id}")
                return

        if not property_holds:
            print("Property does not hold for at least one element in self.clique.")
            return  # Exit the function if the first property check failed

        # Second check: neighbors consistency
        cdef int r, p, s
        for r in range(min(self.n, nRowsToCheck)):
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
        
