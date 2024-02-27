from trios import LinkedListArray
import numpy as np

def test_linked_list_array():
    # Initialize the LinkedListArray with 3 linked lists for example
    n_lists = 5
    linked_list_array = LinkedListArray(n_lists)

    # Define the arrays
    what = np.array([10, 20, 30, 40, 50, 60, 70, 60, 70], dtype=np.intc)  # Values to add
    where = np.array([0, 1, 2, 3, 4], dtype=np.intc)  # Indices of linked lists to add values to
    which = np.array([0, 1, 2, 3, 4, 1, 1, 2, 2], dtype=np.intc)  # Indices in 'what' array
    order = np.random.permutation(len(what))
    # Assign values to linked lists
    linked_list_array.assign(what[order], where,which[order])

    linked_list_array.remove_difference(2, 1)

    arr = linked_list_array.extract(1)
    print(arr)

# Call the test function
test_linked_list_array()
