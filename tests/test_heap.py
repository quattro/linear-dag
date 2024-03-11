import linear_dag as ld
import numpy as np


priority = np.array([1, 3, 6, 8, 4, 3, 10, 7, 1], dtype=np.intc)

h = ld.ModHeap(priority)
h.push(1, 5)
h.push(1, 5)
print(h.pop())  # 6
h.push(1, 11)
print(h.pop())  # 1
h.push(3, 0)
print(h.pop())  # 7
