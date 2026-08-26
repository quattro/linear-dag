import numpy as np

from linear_dag.core.data_structures import ModHeap


def test_mod_heap_stores_only_live_priorities_and_breaks_ties_by_index():
    heap = ModHeap(np.array([0, 3, 3, 1], dtype=np.int64))

    assert len(heap.act_heap) == 3
    assert heap.live_count == 3
    assert [heap.pop(), heap.pop(), heap.pop(), heap.pop()] == [1, 2, 3, -1]
    assert heap.live_count == 0


def test_mod_heap_rebuild_bounds_stale_entries_and_preserves_updates():
    heap = ModHeap(np.array([5, 4, 3, 0], dtype=np.int64))

    for priority in range(6, 26):
        heap.push(0, priority)

    assert heap.rebuild_count > 0
    assert len(heap.act_heap) <= 2 * heap.live_count

    heap.push(3, 10)
    heap.push(1, 0)
    heap.push(2, 10)

    assert len(heap.act_heap) <= 2 * heap.live_count
    assert [heap.pop(), heap.pop(), heap.pop(), heap.pop()] == [0, 2, 3, -1]
    assert heap.live_count == 0
