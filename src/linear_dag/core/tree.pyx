

cdef struct tnode:
    uint parent
    uint child
    uint next_sib
    uint prev_sib

cdef class Tree:
    cdef tnode * nodes
    cdef uint maximum_number_of_nodes
    cdef uint available_node

    def __cinit__(self, uint maximum_number_of_nodes):
        self.maximum_number_of_nodes = maximum_number_of_nodes
        self.nodes = <tnode *>malloc(maximum_number_of_nodes * sizeof(tnode))
        if self.nodes == NULL:
            raise MemoryError("Failed to allocate memory for nodes")
        self.available_node = 0
        cdef int i
        for i in range(maximum_number_of_nodes):
            self.nodes[i].child = i + 1
            self.nodes[i].parent = i + 1 # node is available
    
    def __dealloc__(self):
        free(self.nodes)
    
    cdef uint add_node(self):
        if self.available_node == self.maximum_number_of_nodes:
            raise MemoryError("Maximum number of nodes reached")
        cdef uint new_node = self.available_node
        self.available_node = self.nodes[new_node].child
        self.nodes[new_node].parent = new_node
        self.nodes[new_node].child = new_node
        return new_node
    
    cdef bint is_node(self, uint node):
        return self.nodes[node].parent != self.nodes[node].child

    cdef void _patch_sibs(self, uint sib):
        self.nodes[self.nodes[sib].prev_sib].next_sib = self.nodes[sib].next_sib
        self.nodes[self.nodes[sib].next_sib].prev_sib = self.nodes[sib].prev_sib

    cdef void _remove_node(self, uint node):
        self.nodes[node].child = self.available_node
        self.nodes[node].parent = self.available_node
        self.available_node = node

    cdef void remove_leaf(self, uint node):
        if not self.is_node(node):
            raise ValueError("Node is not in the tree")
        if self.nodes[node].child != node:
            raise ValueError("Node is not a leaf")
        
        self._patch_sibs(node)
        self._remove_node(node)

    cdef void set_parent(self, uint child, uint parent):
        if not self.is_node(parent):
            raise ValueError("Parent node is not in the tree")
        if not self.is_node(child):
            raise ValueError("Child node is not in the tree")
        
        self._patch_sibs(child)
        self.nodes[child].parent = parent
        
        # get new sibs
        cdef uint old_child = self.nodes[parent].child
        if old_child != parent: # no children
            self.nodes[old_child].prev_sib = child
            self.nodes[child].next_sib = old_child
        else:
            self.nodes[child].next_sib = child
            self.nodes[child].prev_sib = child
        self.nodes[parent].child = child
    
    cdef void _merge_sibs(self, uint sib1, uint sib2):
        self.nodes[sib1].next_sib = sib2
         
        self.nodes[self.nodes[sib2].next_sib].prev_sib = sib1
    
    cdef void collapse_edge(self, uint child, uint parent):
        
        # Update parent of each grandchild
        cdef uint grandchild = self.nodes[child].child
        cdef uint next_grandchild
        while self.nodes[grandchild].parent == child:
            next_grandchild = self.nodes[grandchild].next_sib
            self.nodes[grandchild].parent = parent
            grandchild = next_grandchild
        
        #  Merge grandchildren into list of sibs, replacing child
        next_grandchild = self.nodes[grandchild].next_sib
        cdef uint prev_sib = self.nodes[child].prev_sib
        cdef uint next_sib = self.nodes[child].next_sib
        self.nodes[prev_sib].next_sib = next_grandchild
        self.nodes[next_grandchild].prev_sib = prev_sib
        self.nodes[next_sib].prev_sib = grandchild
        self.nodes[grandchild].next_sib = next_sib

        self._remove_node(child)
            
    