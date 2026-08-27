# pattern: Mixed (unavoidable)
# Reason: the input-linear hot path must update graph edges and its intrusive
# boundary index together; separating those mutations would require copying
# dynamic graph state or allocating operation objects inside the loop.

from libc.limits cimport INT_MAX
from libc.stdint cimport int32_t, int64_t, uint32_t, UINT32_MAX
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memset

import numpy as np

from .digraph cimport DiGraph, edge, node


cdef struct boundary_record:
    int32_t left_edge_index
    int32_t class_id
    int32_t occurrence_prev
    int32_t occurrence_next
    int32_t child_prev
    int32_t child_next


cdef struct pair_class:
    int32_t count
    int32_t occurrence_head
    int32_t bucket_prev
    int32_t bucket_next


cdef void* checked_malloc(size_t count, size_t item_size) except NULL:
    cdef void* result
    if item_size != 0 and count > (<size_t> -1) // item_size:
        raise OverflowError("MSBF allocation size overflow")
    if count == 0:
        count = 1
    result = malloc(count * item_size)
    if result is NULL:
        raise MemoryError("Could not allocate MSBF workspace")
    return result


cdef void* checked_realloc(void* current, size_t count, size_t item_size) except NULL:
    cdef void* result
    if item_size != 0 and count > (<size_t> -1) // item_size:
        raise OverflowError("MSBF allocation size overflow")
    if count == 0:
        count = 1
    result = realloc(current, count * item_size)
    if result is NULL:
        raise MemoryError("Could not grow MSBF workspace")
    return result


cdef class Recombination(DiGraph):
    """Find shared ordered parent blocks and replace them with factor nodes."""

    cdef boundary_record* boundaries
    cdef pair_class* classes
    cdef int32_t* left_class_map
    cdef int32_t* right_class_map
    cdef int32_t* bucket_head
    cdef int32_t* left_scratch
    cdef int32_t* right_scratch
    cdef int32_t* external_scratch
    cdef int32_t* touched_class_scratch
    cdef int32_t* bucket_event_scratch
    cdef int32_t* bucket_order_scratch
    cdef int32_t* parent_scratch
    cdef int32_t* block_class_scratch
    cdef int64_t boundary_count
    cdef int64_t boundary_capacity
    cdef int64_t class_count
    cdef int64_t class_capacity
    cdef int64_t bucket_capacity
    cdef int64_t node_scan_capacity
    cdef int64_t reserved_node_capacity
    cdef int64_t scratch_capacity
    cdef int64_t parent_scratch_capacity
    cdef int64_t largest_frequency
    cdef int64_t bucket_mutation_count
    cdef int64_t direct_external_factor_count
    cdef int64_t batched_external_factor_count

    def __cinit__(self, int num_nodes, int num_edges):
        self.boundaries = NULL
        self.classes = NULL
        self.left_class_map = NULL
        self.right_class_map = NULL
        self.bucket_head = NULL
        self.left_scratch = NULL
        self.right_scratch = NULL
        self.external_scratch = NULL
        self.touched_class_scratch = NULL
        self.bucket_event_scratch = NULL
        self.bucket_order_scratch = NULL
        self.parent_scratch = NULL
        self.block_class_scratch = NULL
        self.boundary_count = 0
        self.boundary_capacity = 0
        self.class_count = 0
        self.class_capacity = 0
        self.bucket_capacity = 0
        self.node_scan_capacity = 0
        self.reserved_node_capacity = 0
        self.scratch_capacity = 0
        self.parent_scratch_capacity = 0
        self.largest_frequency = 0
        self.bucket_mutation_count = 0
        self.direct_external_factor_count = 0
        self.batched_external_factor_count = 0

    def __init__(self, int num_nodes, int num_edges):
        DiGraph.__init__(self, num_nodes, num_edges)

    def __dealloc__(self):
        free(self.boundaries)
        free(self.classes)
        free(self.left_class_map)
        free(self.right_class_map)
        free(self.bucket_head)
        free(self.left_scratch)
        free(self.right_scratch)
        free(self.external_scratch)
        free(self.touched_class_scratch)
        free(self.bucket_event_scratch)
        free(self.bucket_order_scratch)
        free(self.parent_scratch)
        free(self.block_class_scratch)

    @staticmethod
    def from_graph(brick_graph: DiGraph) -> Recombination:
        """Copy a graph and initialize maximal-support boundary classes."""
        cdef DiGraph source = brick_graph
        cdef int64_t n = source.maximum_node_index() + 1
        cdef int64_t m = source.number_of_edges
        cdef int64_t initial_node_capacity
        cdef int64_t node_capacity
        cdef Recombination result

        if n > INT_MAX or m > INT_MAX - 2:
            raise OverflowError("Graph is too large for the Cython graph index types")
        if m > 2 * (<int64_t> INT_MAX - n - 1):
            raise OverflowError("MSBF factor-node capacity would overflow")

        # Every factor consumes at least two active boundaries. The extra slot
        # also keeps an empty graph constructible. Preserve the historical +2
        # edge reserve even though MSBF never needs transient edge growth.
        node_capacity = n + m // 2 + 1
        initial_node_capacity = n if n > 0 else 1
        result = Recombination(<int> initial_node_capacity, <int> (m + 2))
        result._copy_ordered_from(source, n, node_capacity)
        result._initialize_boundaries(n)
        return result

    cdef void _copy_ordered_from(
        self,
        DiGraph source,
        int64_t n,
        int64_t node_capacity,
    ) except *:
        cdef int64_t node_index
        cdef edge* current

        # Retain the legacy dense 0..maximum-index node domain so newly added
        # factor nodes never reuse a pre-existing index gap.
        for node_index in range(n):
            self.add_node(node_index)

        # No edge points into the node array yet, so it can move safely. Keep
        # the worst-case factor capacity as untouched address space and
        # materialize factor records sequentially only when factors are found.
        if n > 0:
            if self.number_of_available_nodes != 0:
                raise RuntimeError("Original node domain was not fully initialized")
            self.nodes = <node*> checked_realloc(
                self.nodes, node_capacity, sizeof(node)
            )
        self.reserved_node_capacity = node_capacity

        # add_edge inserts at the head. Walking each source parent list from
        # tail to head therefore preserves its existing order exactly.
        for node_index in range(n):
            if not source.is_node(node_index):
                continue
            current = source.nodes[node_index].first_in
            if current is NULL:
                continue
            while current.next_in is not NULL:
                current = current.next_in
            while current is not NULL:
                self.add_edge(current.u.index, node_index)
                current = current.prev_in

    cdef inline node* _add_factor_node(self) except NULL:
        cdef int64_t node_index = self.maximum_number_of_nodes
        cdef node* new_node
        if node_index >= self.reserved_node_capacity:
            raise RuntimeError("MSBF factor-node reserve was exhausted")
        new_node = &self.nodes[node_index]
        new_node.index = <int> node_index
        new_node.first_in = NULL
        new_node.first_out = NULL
        self.maximum_number_of_nodes += 1
        return new_node

    cdef inline edge* _boundary_left_edge(self, int64_t slot) noexcept:
        return &self.edge_arrays[0][self.boundaries[slot].left_edge_index]

    cdef void _initialize_boundaries(self, int64_t existing_node_count) except *:
        cdef int64_t edge_capacity = self.number_of_edges
        cdef int64_t node_capacity = existing_node_count
        cdef int32_t* head_by_left = NULL
        cdef uint32_t* pair_stamp = NULL
        cdef int32_t* pair_value = NULL
        cdef uint32_t epoch = 0
        cdef int64_t child
        cdef int64_t previous_slot
        cdef int64_t next_same
        cdef int64_t slot
        cdef int64_t class_id
        cdef int64_t left_parent
        cdef int64_t right_parent
        cdef int64_t frequency
        cdef edge* current

        self.boundary_capacity = edge_capacity if edge_capacity > 0 else 1
        self.boundaries = <boundary_record*> checked_malloc(
            self.boundary_capacity, sizeof(boundary_record)
        )
        # Classes are normally far fewer than boundaries. Grow these arrays
        # with the number of distinct pairs instead of reserving one record per
        # input edge up front.
        self.class_capacity = 0

        # A pair can occur at most once in each original child list. Factor-node
        # defining edges never enter the boundary index, so this bound remains
        # valid for the lifetime of the index.
        self.node_scan_capacity = node_capacity
        self.bucket_capacity = node_capacity if node_capacity > 0 else 1
        self.bucket_head = <int32_t*> checked_malloc(
            self.bucket_capacity + 1, sizeof(int32_t)
        )
        for slot in range(self.bucket_capacity + 1):
            self.bucket_head[slot] = -1

        try:
            head_by_left = <int32_t*> checked_malloc(node_capacity, sizeof(int32_t))
            pair_stamp = <uint32_t*> checked_malloc(node_capacity, sizeof(uint32_t))
            pair_value = <int32_t*> checked_malloc(node_capacity, sizeof(int32_t))
            for slot in range(node_capacity):
                head_by_left[slot] = -1
                pair_stamp[slot] = 0

            # Create one stable slot per active child-list adjacency and link
            # those slots in the same order as the in-edges.
            for child in range(node_capacity):
                if not self.is_node(child):
                    continue
                previous_slot = -1
                current = self.nodes[child].first_in
                while current is not NULL and current.next_in is not NULL:
                    if self.boundary_count >= self.boundary_capacity:
                        raise RuntimeError("Boundary count exceeded edge capacity")
                    slot = self.boundary_count
                    self.boundary_count += 1
                    self.boundaries[slot].left_edge_index = <int32_t> current.index
                    self.boundaries[slot].class_id = -1
                    self.boundaries[slot].occurrence_prev = -1
                    self.boundaries[slot].child_prev = previous_slot
                    self.boundaries[slot].child_next = -1
                    if previous_slot >= 0:
                        self.boundaries[previous_slot].child_next = slot
                    previous_slot = slot

                    left_parent = current.u.index
                    # Reuse occurrence_next as the temporary same-left chain.
                    # Class construction below overwrites it with the lasting
                    # occurrence-list link.
                    self.boundaries[slot].occurrence_next = head_by_left[left_parent]
                    head_by_left[left_parent] = slot
                    current = current.next_in

            # CountingArray-style grouping: clear the right-parent map by
            # advancing a timestamp once for each left parent.
            for left_parent in range(node_capacity):
                slot = head_by_left[left_parent]
                if slot < 0:
                    continue
                if epoch == UINT32_MAX:
                    memset(pair_stamp, 0, node_capacity * sizeof(uint32_t))
                    epoch = 0
                epoch += 1
                while slot >= 0:
                    next_same = self.boundaries[slot].occurrence_next
                    right_parent = self._boundary_left_edge(slot).next_in.u.index
                    if pair_stamp[right_parent] != epoch:
                        class_id = self._new_class()
                        pair_stamp[right_parent] = epoch
                        pair_value[right_parent] = class_id
                    else:
                        class_id = pair_value[right_parent]
                    self._link_occurrence(slot, class_id)
                    slot = next_same

            for class_id in range(self.class_count):
                frequency = self.classes[class_id].count
                if frequency > 0:
                    self._bucket_insert(class_id)
                    if frequency > self.largest_frequency:
                        self.largest_frequency = frequency
        finally:
            free(head_by_left)
            free(pair_stamp)
            free(pair_value)

    cdef void _ensure_class_capacity(self, int64_t needed) except *:
        cdef int64_t old_capacity
        cdef int64_t new_capacity

        if needed <= self.class_capacity:
            return
        old_capacity = self.class_capacity
        new_capacity = old_capacity if old_capacity > 0 else 8
        while new_capacity < needed:
            if new_capacity > (<int64_t> INT_MAX) // 2:
                raise OverflowError("MSBF class capacity overflow")
            new_capacity *= 2

        self.classes = <pair_class*> checked_realloc(
            self.classes, new_capacity, sizeof(pair_class)
        )
        self.left_class_map = <int32_t*> checked_realloc(
            self.left_class_map, new_capacity, sizeof(int32_t)
        )
        self.right_class_map = <int32_t*> checked_realloc(
            self.right_class_map, new_capacity, sizeof(int32_t)
        )
        memset(
            &self.left_class_map[old_capacity],
            0xFF,
            (new_capacity - old_capacity) * sizeof(int32_t),
        )
        memset(
            &self.right_class_map[old_capacity],
            0xFF,
            (new_capacity - old_capacity) * sizeof(int32_t),
        )
        self.class_capacity = new_capacity

    cdef int64_t _new_class(self) except -1:
        cdef int64_t class_id
        self._ensure_class_capacity(self.class_count + 1)
        class_id = self.class_count
        self.class_count += 1
        self.classes[class_id].count = 0
        self.classes[class_id].occurrence_head = -1
        self.classes[class_id].bucket_prev = -1
        self.classes[class_id].bucket_next = -1
        return class_id

    cdef inline void _link_occurrence(
        self,
        int64_t slot,
        int64_t class_id,
    ) noexcept:
        cdef int64_t old_head = self.classes[class_id].occurrence_head
        self.boundaries[slot].class_id = class_id
        self.boundaries[slot].occurrence_prev = -1
        self.boundaries[slot].occurrence_next = old_head
        if old_head >= 0:
            self.boundaries[old_head].occurrence_prev = slot
        self.classes[class_id].occurrence_head = slot
        self.classes[class_id].count += 1

    cdef void _bucket_insert(self, int64_t class_id) except *:
        cdef int64_t frequency = self.classes[class_id].count
        cdef int64_t old_head
        if frequency <= 0 or frequency > self.bucket_capacity:
            raise RuntimeError("Invalid MSBF class frequency")
        old_head = self.bucket_head[frequency]
        self.classes[class_id].bucket_prev = -1
        self.classes[class_id].bucket_next = old_head
        if old_head >= 0:
            self.classes[old_head].bucket_prev = class_id
        self.bucket_head[frequency] = class_id
        self.bucket_mutation_count += 1

    cdef void _bucket_remove(self, int64_t class_id) except *:
        cdef int64_t frequency = self.classes[class_id].count
        cdef int64_t previous = self.classes[class_id].bucket_prev
        cdef int64_t following = self.classes[class_id].bucket_next
        if frequency <= 0 or frequency > self.bucket_capacity:
            raise RuntimeError("Invalid bucket removal frequency")
        if previous >= 0:
            self.classes[previous].bucket_next = following
        else:
            if self.bucket_head[frequency] != class_id:
                raise RuntimeError("MSBF class is missing from its frequency bucket")
            self.bucket_head[frequency] = following
        if following >= 0:
            self.classes[following].bucket_prev = previous
        self.classes[class_id].bucket_prev = -1
        self.classes[class_id].bucket_next = -1
        self.bucket_mutation_count += 1

    cdef inline void _unlink_occurrence(self, int64_t slot) except *:
        cdef boundary_record* boundary = &self.boundaries[slot]
        cdef int64_t class_id = boundary.class_id
        cdef int64_t previous = boundary.occurrence_prev
        cdef int64_t following = boundary.occurrence_next

        if boundary.left_edge_index < 0 or class_id < 0:
            raise RuntimeError("Attempted to unlink a stale boundary occurrence")
        if previous >= 0:
            self.boundaries[previous].occurrence_next = following
        else:
            self.classes[class_id].occurrence_head = following
        if following >= 0:
            self.boundaries[following].occurrence_prev = previous
        self.classes[class_id].count -= 1
        boundary.class_id = -1
        boundary.occurrence_prev = -1
        boundary.occurrence_next = -1

    cdef void _ensure_occurrence_scratch(self, int64_t needed) except *:
        cdef int64_t new_capacity
        if needed <= self.scratch_capacity:
            return
        new_capacity = self.scratch_capacity if self.scratch_capacity > 0 else 8
        while new_capacity < needed:
            if new_capacity > (<int64_t> INT_MAX) // 2:
                raise OverflowError("MSBF occurrence scratch capacity overflow")
            new_capacity *= 2
        self.left_scratch = <int32_t*> checked_realloc(
            self.left_scratch, new_capacity, sizeof(int32_t)
        )
        self.right_scratch = <int32_t*> checked_realloc(
            self.right_scratch, new_capacity, sizeof(int32_t)
        )
        self.external_scratch = <int32_t*> checked_realloc(
            self.external_scratch, 2 * new_capacity, sizeof(int32_t)
        )
        self.touched_class_scratch = <int32_t*> checked_realloc(
            self.touched_class_scratch, 2 * new_capacity, sizeof(int32_t)
        )
        self.bucket_event_scratch = <int32_t*> checked_realloc(
            self.bucket_event_scratch, 4 * new_capacity, sizeof(int32_t)
        )
        self.bucket_order_scratch = <int32_t*> checked_realloc(
            self.bucket_order_scratch, 4 * new_capacity, sizeof(int32_t)
        )
        self.scratch_capacity = new_capacity

    cdef void _ensure_parent_scratch(self, int64_t needed) except *:
        cdef int64_t new_capacity
        if needed <= self.parent_scratch_capacity:
            return
        new_capacity = self.parent_scratch_capacity if self.parent_scratch_capacity > 0 else 8
        while new_capacity < needed:
            if new_capacity > (<int64_t> INT_MAX) // 2:
                raise OverflowError("MSBF parent scratch capacity overflow")
            new_capacity *= 2
        self.parent_scratch = <int32_t*> checked_realloc(
            self.parent_scratch, new_capacity, sizeof(int32_t)
        )
        self.block_class_scratch = <int32_t*> checked_realloc(
            self.block_class_scratch, new_capacity, sizeof(int32_t)
        )
        self.parent_scratch_capacity = new_capacity

    cdef void _validate_active_boundary(self, int64_t slot) except *:
        cdef boundary_record* boundary
        cdef edge* left_edge
        cdef edge* right_edge
        cdef int64_t class_id
        if slot < 0 or slot >= self.boundary_count:
            raise RuntimeError("Boundary slot is out of range")
        boundary = &self.boundaries[slot]
        if boundary.left_edge_index < 0 or boundary.class_id < 0:
            raise RuntimeError("Encountered a stale boundary slot")
        left_edge = self._boundary_left_edge(slot)
        if left_edge.u is NULL or left_edge.v is NULL:
            raise RuntimeError("Active boundary references a released left edge")
        right_edge = left_edge.next_in
        if right_edge is NULL:
            raise RuntimeError("Active boundary has no right edge")
        if right_edge.u is NULL or right_edge.v is NULL:
            raise RuntimeError("Active boundary references a released right edge")
        if right_edge.prev_in != left_edge:
            raise RuntimeError("Active boundary reverse link is inconsistent")
        if left_edge.v != right_edge.v:
            raise RuntimeError("Active boundary crosses child lists")
        class_id = boundary.class_id
        if class_id < 0 or class_id >= self.class_count:
            raise RuntimeError("Active boundary has an invalid class")

    cdef int64_t _mapped_external_class(
        self,
        int64_t old_class,
        int64_t new_class_floor,
        bint is_left,
    ) except -1:
        cdef int32_t mapped_class
        cdef int64_t new_class

        if old_class < 0 or old_class >= self.class_count:
            raise RuntimeError("External boundary has an invalid old class")
        if is_left:
            mapped_class = self.left_class_map[old_class]
        else:
            mapped_class = self.right_class_map[old_class]
        # Class identifiers increase monotonically. A prior factor's mapping is
        # therefore always below this factor's class floor and needs no epoch.
        if mapped_class >= new_class_floor and mapped_class < self.class_count:
            return mapped_class
        new_class = self._new_class()
        if is_left:
            self.left_class_map[old_class] = <int32_t> new_class
        else:
            self.right_class_map[old_class] = <int32_t> new_class
        return new_class

    cpdef void find_recombinations(self):
        cdef int64_t class_id
        cdef int64_t frequency
        cdef int64_t slot
        cdef int64_t previous_occurrence
        cdef int64_t candidate
        cdef int64_t candidate_class
        cdef int64_t i
        cdef int64_t j
        cdef int64_t parent_count
        cdef int64_t internal_class_count
        cdef int64_t start_slot
        cdef int64_t end_slot
        cdef int64_t left_external
        cdef int64_t right_external
        cdef int64_t old_left_class
        cdef int64_t old_right_class
        cdef int64_t new_class
        cdef int64_t new_class_floor
        cdef int64_t old_class
        cdef int64_t touched_old_class_count
        cdef int64_t external_count
        cdef int64_t bucket_event_count
        cdef int64_t bucket_order_count
        cdef int64_t next_boundary
        cdef int64_t edges_before
        cdef edge* representative
        cdef edge* last_edge
        cdef edge* stop_edge
        cdef edge* current_edge
        cdef edge* next_edge
        cdef node* new_node
        cdef bint can_extend
        cdef bint direct_external_updates

        while self.largest_frequency >= 2:
            while (
                self.largest_frequency >= 2
                and self.bucket_head[self.largest_frequency] < 0
            ):
                self.largest_frequency -= 1
            if self.largest_frequency < 2:
                break

            class_id = self.bucket_head[self.largest_frequency]
            frequency = self.classes[class_id].count
            if frequency != self.largest_frequency:
                raise RuntimeError("Maximum-frequency bucket is inconsistent")
            self._ensure_occurrence_scratch(frequency)

            slot = self.classes[class_id].occurrence_head
            previous_occurrence = -1
            for i in range(frequency):
                if slot < 0:
                    raise RuntimeError("Class occurrence list is shorter than its count")
                self._validate_active_boundary(slot)
                if self.boundaries[slot].class_id != class_id:
                    raise RuntimeError("Occurrence belongs to the wrong class")
                if self.boundaries[slot].occurrence_prev != previous_occurrence:
                    raise RuntimeError("Occurrence reverse link is inconsistent")
                self.left_scratch[i] = slot
                self.right_scratch[i] = slot
                previous_occurrence = slot
                slot = self.boundaries[slot].occurrence_next
            if slot >= 0:
                raise RuntimeError("Class occurrence list is longer than its count")

            # Extend left while all f occurrences expose the same complete
            # maximum-frequency support class.
            while True:
                candidate = self.boundaries[self.left_scratch[0]].child_prev
                if candidate < 0:
                    break
                self._validate_active_boundary(candidate)
                candidate_class = self.boundaries[candidate].class_id
                if self.classes[candidate_class].count != frequency:
                    break
                can_extend = 1
                for i in range(1, frequency):
                    candidate = self.boundaries[self.left_scratch[i]].child_prev
                    if candidate < 0:
                        can_extend = 0
                        break
                    self._validate_active_boundary(candidate)
                    if self.boundaries[candidate].class_id != candidate_class:
                        can_extend = 0
                        break
                if not can_extend:
                    break
                for i in range(frequency):
                    self.left_scratch[i] = self.boundaries[self.left_scratch[i]].child_prev

            # Right extension is analogous and does not change the support set.
            while True:
                candidate = self.boundaries[self.right_scratch[0]].child_next
                if candidate < 0:
                    break
                self._validate_active_boundary(candidate)
                candidate_class = self.boundaries[candidate].class_id
                if self.classes[candidate_class].count != frequency:
                    break
                can_extend = 1
                for i in range(1, frequency):
                    candidate = self.boundaries[self.right_scratch[i]].child_next
                    if candidate < 0:
                        can_extend = 0
                        break
                    self._validate_active_boundary(candidate)
                    if self.boundaries[candidate].class_id != candidate_class:
                        can_extend = 0
                        break
                if not can_extend:
                    break
                for i in range(frequency):
                    self.right_scratch[i] = self.boundaries[self.right_scratch[i]].child_next

            # Snapshot the maximal parent word and its internal classes from one
            # occurrence. All other occurrences are proven equal by extension.
            start_slot = self.left_scratch[0]
            end_slot = self.right_scratch[0]
            parent_count = 0
            current_edge = self._boundary_left_edge(start_slot)
            last_edge = self._boundary_left_edge(end_slot).next_in
            stop_edge = last_edge.next_in
            while current_edge != stop_edge:
                self._ensure_parent_scratch(parent_count + 1)
                self.parent_scratch[parent_count] = current_edge.u.index
                parent_count += 1
                current_edge = current_edge.next_in
            if parent_count < 2:
                raise RuntimeError("Selected MSBF block has fewer than two parents")

            internal_class_count = 0
            slot = start_slot
            while True:
                self.block_class_scratch[internal_class_count] = self.boundaries[slot].class_id
                internal_class_count += 1
                if slot == end_slot:
                    break
                slot = self.boundaries[slot].child_next
                if slot < 0:
                    raise RuntimeError("Selected block boundary range is disconnected")
            if internal_class_count != parent_count - 1:
                raise RuntimeError("Selected block edge and boundary counts disagree")

            # Every internal class has exactly this block's support and will be
            # exhausted. Remove each class from its bucket once instead of
            # rebucketing it after every occurrence deletion.
            for j in range(internal_class_count):
                class_id = self.block_class_scratch[j]
                if self.classes[class_id].count != frequency:
                    raise RuntimeError("Internal MSBF class has unexpected support")
                self._bucket_remove(class_id)

            edges_before = self.number_of_edges
            new_node = self._add_factor_node()
            new_class_floor = self.class_count
            touched_old_class_count = 0
            external_count = 0
            bucket_event_count = 0

            # External occurrences can share an old class, and occurrences from
            # one old class share a fresh class on each side of the factor. Take
            # each affected old class out of its bucket once. Repeated classes
            # use final-event replay; all-unique classes reinsert as they move.
            # A -2 predecessor marks a class already staged by this factor.
            for i in range(frequency):
                start_slot = self.left_scratch[i]
                end_slot = self.right_scratch[i]
                self._validate_active_boundary(start_slot)
                self._validate_active_boundary(end_slot)
                left_external = self.boundaries[start_slot].child_prev
                right_external = self.boundaries[end_slot].child_next
                self.external_scratch[2 * i] = <int32_t> left_external
                self.external_scratch[2 * i + 1] = <int32_t> right_external

                if left_external >= 0:
                    external_count += 1
                    old_class = self.boundaries[left_external].class_id
                    if self.classes[old_class].bucket_prev != -2:
                        self._bucket_remove(old_class)
                        self.classes[old_class].bucket_prev = -2
                        self.touched_class_scratch[touched_old_class_count] = (
                            <int32_t> old_class
                        )
                        touched_old_class_count += 1
                if right_external >= 0:
                    external_count += 1
                    old_class = self.boundaries[right_external].class_id
                    if self.classes[old_class].bucket_prev != -2:
                        self._bucket_remove(old_class)
                        self.classes[old_class].bucket_prev = -2
                        self.touched_class_scratch[touched_old_class_count] = (
                            <int32_t> old_class
                        )
                        touched_old_class_count += 1

            # If every exposed occurrence came from a different old class,
            # batching cannot save a bucket mutation. Reinsert each class as
            # its occurrence moves and avoid recording/deduplicating a replay.
            # All old classes were staged out first, so this produces the same
            # final head-insertion order as the sequential update sequence.
            direct_external_updates = (
                external_count > 0 and touched_old_class_count == external_count
            )
            if direct_external_updates:
                self.direct_external_factor_count += 1
            elif external_count > 0:
                self.batched_external_factor_count += 1

            for i in range(frequency):
                start_slot = self.left_scratch[i]
                end_slot = self.right_scratch[i]
                left_external = self.external_scratch[2 * i]
                right_external = self.external_scratch[2 * i + 1]
                old_left_class = (
                    self.boundaries[left_external].class_id if left_external >= 0 else -1
                )
                old_right_class = (
                    self.boundaries[right_external].class_id if right_external >= 0 else -1
                )
                # The endpoint arrays are no longer needed after this
                # occurrence is localized; retain the exposed slots for the
                # post-factor maximality check.
                self.left_scratch[i] = left_external
                self.right_scratch[i] = right_external

                representative = self._boundary_left_edge(start_slot)
                last_edge = self._boundary_left_edge(end_slot).next_in
                stop_edge = last_edge.next_in

                # Check the complete word before releasing any edge. This also
                # makes stale-boundary failures deterministic rather than
                # allowing an edge-pool reuse to mask them.
                current_edge = representative
                for j in range(parent_count):
                    if current_edge == stop_edge or current_edge is NULL:
                        raise RuntimeError("MSBF occurrence is shorter than its block")
                    if current_edge.u.index != self.parent_scratch[j]:
                        raise RuntimeError("MSBF occurrences disagree on parent order")
                    current_edge = current_edge.next_in
                if current_edge != stop_edge:
                    raise RuntimeError("MSBF occurrence is longer than its block")

                # Exhaust every internal boundary occurrence before any of its
                # edge pointers can enter the reusable graph-edge pool.
                slot = start_slot
                while True:
                    next_boundary = self.boundaries[slot].child_next
                    self._unlink_occurrence(slot)
                    self.boundaries[slot].child_prev = -1
                    self.boundaries[slot].child_next = -1
                    self.boundaries[slot].left_edge_index = -1
                    if slot == end_slot:
                        break
                    slot = next_boundary

                if left_external >= 0:
                    self.boundaries[left_external].child_next = right_external
                if right_external >= 0:
                    self.boundaries[right_external].child_prev = left_external
                    self.boundaries[right_external].left_edge_index = (
                        <int32_t> representative.index
                    )

                current_edge = representative.next_in
                while current_edge != stop_edge:
                    next_edge = current_edge.next_in
                    self.remove_edge(current_edge)
                    current_edge = next_edge
                self.set_edge_parent(representative, new_node)

                if left_external >= 0:
                    new_class = self._mapped_external_class(
                        old_left_class, new_class_floor, 1
                    )
                    self._unlink_occurrence(left_external)
                    self._link_occurrence(left_external, new_class)
                    if direct_external_updates:
                        self.classes[old_left_class].bucket_prev = -1
                        if self.classes[old_left_class].count > 0:
                            self._bucket_insert(old_left_class)
                        self._bucket_insert(new_class)
                    else:
                        if self.classes[old_left_class].count > 0:
                            self.bucket_event_scratch[bucket_event_count] = (
                                <int32_t> old_left_class
                            )
                            bucket_event_count += 1
                        self.bucket_event_scratch[bucket_event_count] = (
                            <int32_t> new_class
                        )
                        bucket_event_count += 1
                    self._validate_active_boundary(left_external)
                if right_external >= 0:
                    new_class = self._mapped_external_class(
                        old_right_class, new_class_floor, 0
                    )
                    self._unlink_occurrence(right_external)
                    self._link_occurrence(right_external, new_class)
                    if direct_external_updates:
                        self.classes[old_right_class].bucket_prev = -1
                        if self.classes[old_right_class].count > 0:
                            self._bucket_insert(old_right_class)
                        self._bucket_insert(new_class)
                    else:
                        if self.classes[old_right_class].count > 0:
                            self.bucket_event_scratch[bucket_event_count] = (
                                <int32_t> old_right_class
                            )
                            bucket_event_count += 1
                        self.bucket_event_scratch[bucket_event_count] = (
                            <int32_t> new_class
                        )
                        bucket_event_count += 1
                    self._validate_active_boundary(right_external)

            # Recreate the exact bucket ordering that sequential occurrence
            # moves would produce. Only a class's final insertion event affects
            # its lasting position; replay those final events chronologically.
            if not direct_external_updates:
                bucket_order_count = 0
                for j in range(bucket_event_count - 1, -1, -1):
                    class_id = self.bucket_event_scratch[j]
                    if (
                        self.classes[class_id].count > 0
                        and self.classes[class_id].bucket_prev != -3
                    ):
                        self.classes[class_id].bucket_prev = -3
                        self.bucket_order_scratch[bucket_order_count] = <int32_t> class_id
                        bucket_order_count += 1
                for i in range(touched_old_class_count):
                    old_class = self.touched_class_scratch[i]
                    if self.classes[old_class].count == 0:
                        if self.classes[old_class].bucket_prev != -2:
                            raise RuntimeError("Staged empty class lost its bucket marker")
                        self.classes[old_class].bucket_prev = -1
                for j in range(bucket_order_count - 1, -1, -1):
                    class_id = self.bucket_order_scratch[j]
                    if self.classes[class_id].bucket_prev != -3:
                        raise RuntimeError("Staged class lost its insertion marker")
                    self.classes[class_id].bucket_prev = -1
                    self._bucket_insert(class_id)

            # Released block edges are sufficient even in the k=f=2 equality
            # case, so defining edges never trigger edge-pool growth.
            if self.number_of_available_edges < parent_count:
                raise RuntimeError("MSBF did not release enough defining-edge capacity")
            for j in range(parent_count - 1, -1, -1):
                self.add_edge(self.parent_scratch[j], new_node.index)

            if self.number_of_edges > edges_before:
                raise RuntimeError("MSBF increased the graph edge count")
            for j in range(internal_class_count):
                if self.classes[self.block_class_scratch[j]].count != 0:
                    raise RuntimeError("A factored internal pair retained active support")
            for i in range(frequency):
                left_external = self.left_scratch[i]
                right_external = self.right_scratch[i]
                if (
                    left_external >= 0
                    and self.classes[self.boundaries[left_external].class_id].count >= frequency
                ):
                    raise RuntimeError("A newly exposed left class violates maximality")
                if (
                    right_external >= 0
                    and self.classes[self.boundaries[right_external].class_id].count >= frequency
                ):
                    raise RuntimeError("A newly exposed right class violates maximality")

    def _validate_boundary_index(self):
        """Validate every active boundary, occurrence list, and frequency bucket."""
        cdef int64_t slot
        cdef int64_t class_id
        cdef int64_t frequency
        cdef int64_t previous
        cdef int64_t visited
        cdef int64_t expected_left_parent
        cdef int64_t expected_right_parent
        cdef edge* left_edge
        cdef edge* right_edge
        cdef list occurrence_seen = [False] * self.boundary_count
        cdef list bucket_seen = [False] * self.class_count

        for class_id in range(self.class_count):
            slot = self.classes[class_id].occurrence_head
            previous = -1
            visited = 0
            expected_left_parent = -1
            expected_right_parent = -1
            while slot >= 0:
                if visited >= self.boundary_count:
                    raise RuntimeError("Boundary occurrence list contains a cycle")
                self._validate_active_boundary(slot)
                if self.boundaries[slot].class_id != class_id:
                    raise RuntimeError("Boundary occurrence is linked from the wrong class")
                if self.boundaries[slot].occurrence_prev != previous:
                    raise RuntimeError("Boundary occurrence reverse link is inconsistent")
                left_edge = self._boundary_left_edge(slot)
                right_edge = left_edge.next_in
                if expected_left_parent < 0:
                    expected_left_parent = left_edge.u.index
                    expected_right_parent = right_edge.u.index
                elif (
                    left_edge.u.index != expected_left_parent
                    or right_edge.u.index != expected_right_parent
                ):
                    raise RuntimeError("Boundary class mixes different parent pairs")
                if occurrence_seen[slot]:
                    raise RuntimeError("Boundary occurrence appears more than once")
                occurrence_seen[slot] = True
                previous = slot
                slot = self.boundaries[slot].occurrence_next
                visited += 1
            if visited != self.classes[class_id].count:
                raise RuntimeError("Boundary class count disagrees with its occurrence list")

        for slot in range(self.boundary_count):
            if self.boundaries[slot].class_id >= 0:
                self._validate_active_boundary(slot)
                if not occurrence_seen[slot]:
                    raise RuntimeError("Active boundary is absent from its occurrence class")
                previous = self.boundaries[slot].child_prev
                if previous >= 0:
                    if self.boundaries[previous].child_next != slot:
                        raise RuntimeError("Boundary child reverse link is inconsistent")
                    if (
                        self._boundary_left_edge(previous).next_in
                        != self._boundary_left_edge(slot)
                    ):
                        raise RuntimeError("Adjacent boundary slots do not share their edge")
                previous = self.boundaries[slot].child_next
                if previous >= 0:
                    if self.boundaries[previous].child_prev != slot:
                        raise RuntimeError("Boundary child forward link is inconsistent")
                    if (
                        self._boundary_left_edge(slot).next_in
                        != self._boundary_left_edge(previous)
                    ):
                        raise RuntimeError("Adjacent boundary slots do not share their edge")
            elif occurrence_seen[slot]:
                raise RuntimeError("Inactive boundary remains in an occurrence class")

        for frequency in range(1, self.bucket_capacity + 1):
            class_id = self.bucket_head[frequency]
            previous = -1
            visited = 0
            while class_id >= 0:
                if visited >= self.class_count:
                    raise RuntimeError("Frequency bucket contains a cycle")
                if bucket_seen[class_id]:
                    raise RuntimeError("Boundary class appears in multiple frequency buckets")
                if self.classes[class_id].count != frequency:
                    raise RuntimeError("Boundary class is in the wrong frequency bucket")
                if self.classes[class_id].bucket_prev != previous:
                    raise RuntimeError("Frequency bucket reverse link is inconsistent")
                bucket_seen[class_id] = True
                previous = class_id
                class_id = self.classes[class_id].bucket_next
                visited += 1
        for class_id in range(self.class_count):
            if (self.classes[class_id].count > 0) != bool(bucket_seen[class_id]):
                raise RuntimeError("Boundary class bucket membership is inconsistent")
        return True

    @property
    def get_clique_rows(self):
        """Return active class occurrences for legacy diagnostics."""
        cdef int64_t class_id
        cdef int64_t slot
        result = []
        for class_id in range(self.class_count):
            entries = []
            slot = self.classes[class_id].occurrence_head
            while slot >= 0:
                entries.append(self.boundaries[slot].left_edge_index)
                slot = self.boundaries[slot].occurrence_next
            result.append(np.asarray(entries, dtype=np.int64))
        return result

    @property
    def get_cliques(self):
        """Return the active left-edge to class mapping for diagnostics."""
        cdef int64_t slot
        result = np.full(self.maximum_number_of_edges, -1, dtype=np.int64)
        for slot in range(self.boundary_count):
            if self.boundaries[slot].class_id >= 0:
                result[self.boundaries[slot].left_edge_index] = self.boundaries[slot].class_id
        return result

    @property
    def workspace_statistics(self):
        """Return allocation counters for boundary-index diagnostics."""
        return {
            "node_scan_capacity": self.node_scan_capacity,
            "bucket_capacity": self.bucket_capacity,
            "reserved_node_capacity": self.reserved_node_capacity,
            "materialized_node_capacity": self.maximum_number_of_nodes,
            "boundary_count": self.boundary_count,
            "boundary_capacity": self.boundary_capacity,
            "class_count": self.class_count,
            "class_capacity": self.class_capacity,
            "boundary_record_bytes": sizeof(boundary_record),
            "class_record_bytes": sizeof(pair_class),
            "class_map_entry_bytes": sizeof(int32_t),
            "node_record_bytes": sizeof(node),
            "bucket_mutation_count": self.bucket_mutation_count,
            "direct_external_factor_count": self.direct_external_factor_count,
            "batched_external_factor_count": self.batched_external_factor_count,
        }

    @property
    def get_heap(self):
        """Return frequency state in the shape of the retired heap diagnostic."""
        cdef int64_t class_id
        priorities = np.zeros(self.class_count, dtype=np.int64)
        for class_id in range(self.class_count):
            priorities[class_id] = self.classes[class_id].count
        return [], priorities
