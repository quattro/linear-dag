from trios import Trios

# Create an instance of Trios with enough space for at least a few trios
n = 5  # Example size
trios_instance = Trios(n)

# Assign some data using add_trio
# Parameters: row, parent1, parent2, child, weight, clique, left_neighbor, right_neighbor
trios_instance.add_trio(0, 0, 1, 4, 1, 0, -1, 2)  # Example trio data
trios_instance.add_trio(1, 0, 1, 5, 1, 0, -1, -1)  # Example trio data
trios_instance.add_trio(2, 1, 2, 4, 1, 1, 0, 3)  # Another trio with the first trio as left neighbor
trios_instance.add_trio(3, 2, 3, 4, 1, 2, 2, -1)  # Another trio with the first trio as left neighbor

# Now call bypass_trios() on the second trio to bypass its connection to the first trio
# Assuming bypass_trios() takes the index of the trio to bypass
#trios_instance.bypass_trios(2)
# trios_instance.collapse_clique(0, 6, [0,1])
# trios_instance.remove_adjacent(0, 1)
# trios_instance.update_edgelist(0,0)
# edges = trios_instance.extract_edgelist()
# print(edges)
#

for i in range(4):
    tup = trios_instance.extract_trio(i)
    print(i, tup)