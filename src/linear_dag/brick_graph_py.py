from collections import defaultdict

import networkx as nx
import numpy as np

from scipy.sparse import csc_matrix


class BrickGraph(nx.DiGraph):
    num_samples: int
    tree: nx.DiGraph
    root: int
    subsequence: dict
    times_visited: dict

    def __init__(self, num_samples: int, num_variants: int):
        super().__init__()
        self.num_samples = num_samples
        self.tree = nx.DiGraph()
        self.cumulative_number_of_tree_nodes = 0
        self.subsequence = {}
        self.times_visited = {}
        self.root = 0
        self.initialize_tree()

    def initialize_tree(self):
        number_of_nodes = self.num_samples + 1
        self.root = self.num_samples
        self.tree = nx.DiGraph()
        self.tree.add_nodes_from(range(number_of_nodes))
        self.tree.add_edges_from([(self.root, i) for i in range(self.num_samples)])
        self.cumulative_number_of_tree_nodes = number_of_nodes
        self.subsequence = {i: [] for i in range(number_of_nodes)}

    @staticmethod
    def from_genotypes(sparse_matrix: csc_matrix):
        num_samples, num_variants = sparse_matrix.shape
        result = BrickGraph(num_samples, num_variants)

        # forward pass
        for i in range(num_variants):
            carriers = np.where(sparse_matrix[:, i].toarray())[0]
            result.intersect_clades(carriers, i)
            assert nx.is_tree(result.tree), f"{i}, {result.tree.edges}"

        # backward pass
        result.initialize_tree()
        for i in reversed(range(num_variants)):
            carriers = np.where(sparse_matrix[:, i].toarray())[0]
            result.intersect_clades(carriers, i)

        return result

    def to_csr(self):
        return nx.to_scipy_sparse_array(self, format="csc", dtype=np.intc).transpose()

    def intersect_clades(self, carriers: list, clade_index: int) -> None:
        traversal: list = self.tree_partial_traversal(carriers)

        lowest_common_ancestor = traversal[-1]
        self.add_edges_from_subsequence(lowest_common_ancestor, clade_index)
        self.add_edge(clade_index, clade_index)

        for v in traversal:
            visited_children, unvisited_children = self.get_visited_children(v)
            num_children_unvisited, num_children_visited = len(unvisited_children), len(visited_children)

            # No unvisited children: means intersect(v, new_clade) == v
            if num_children_unvisited == 0:
                self.subsequence[v].append(clade_index)
                continue

            # v must have a parent node
            if v == self.root:
                self.create_new_root()
            assert self.parent_in_tree(v) is not None

            # Exactly one visited and one unvisited child: delete v, as there are existing nodes
            # for both intersect(v, new_clade) and intersect(v, new_clade_complement)
            if num_children_visited == 1 and num_children_unvisited == 1:
                self.remove_tree_node(v)
                continue

            # Exactly one child w is visited: there is an existing node for intersect(v, new_clade);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade_complement)
            if num_children_visited == 1:
                visited_child = visited_children.pop()
                self.set_parent_in_tree(visited_child, self.parent_in_tree(v))
                self.times_visited[v] = 0
                continue

            # Exactly one child is w is unvisited: there is an existing node for intersect(v, new_clade_complement);
            # replace (v,w) with (parent(v), w), replacing v with intersect(v, new_clade)
            if num_children_unvisited == 1:
                unvisited_child = unvisited_children.pop()
                self.set_parent_in_tree(unvisited_child, self.parent_in_tree(v))
                self.subsequence[v].append(clade_index)  # v is now a subset of the new clade
                continue

            # Multiple visited and unvisited children: create new_node for intersect(v, new_clade_complement)
            # and replace v with intersect(v, new_clade)
            sibling_node = self.add_tree_node()
            self.subsequence[sibling_node] = self.subsequence[v].copy()
            self.subsequence[v].append(clade_index)
            self.set_parent_in_tree(sibling_node, self.parent_in_tree(v))
            for child in unvisited_children:
                self.set_parent_in_tree(child, sibling_node)

    def parent_in_tree(self, node: int) -> int:
        if self.tree.in_degree[node] == 0:
            return None
        return next(self.tree.predecessors(node))

    def tree_partial_traversal(self, leaves: np.ndarray) -> list:
        self.times_visited = defaultdict(int)
        num_leaves = len(leaves)
        active_nodes = list(leaves)

        # Bottom-up traversal tracking how many visited leaves are descended from each node
        while active_nodes:
            v = active_nodes.pop()
            self.times_visited[v] += 1
            v_is_lca = self.times_visited[v] == num_leaves
            if v_is_lca:
                break
            parent = self.parent_in_tree(v)
            if parent is not None:
                active_nodes.append(parent)
        lowest_common_ancestor = v

        # Top-down traversal putting children after parents
        queue = [lowest_common_ancestor]
        place_in_queue = 0
        while place_in_queue < len(queue):
            node = queue[place_in_queue]
            for child in self.tree.successors(node):
                if self.times_visited[child] > 0:
                    queue.append(child)
            place_in_queue += 1

        return queue[::-1]

    def add_edges_from_subsequence(self, add_from: int, add_to: int):
        self.add_edges_from([(node, add_to) for node in self.subsequence[add_from]])

    def add_tree_node(self) -> int:
        self.tree.add_node(self.cumulative_number_of_tree_nodes)
        self.subsequence[self.cumulative_number_of_tree_nodes] = []
        self.cumulative_number_of_tree_nodes += 1
        return self.cumulative_number_of_tree_nodes - 1

    def get_visited_children(self, node: int) -> tuple[list, list]:
        visited_children, unvisited_children = [], []
        for child in self.tree.successors(node):
            if self.times_visited[child] > 0:
                visited_children.append(child)
            else:
                unvisited_children.append(child)
        return visited_children, unvisited_children

    def set_parent_in_tree(self, node: int, parent: int):
        if self.parent_in_tree(node):
            self.tree.remove_edge(self.parent_in_tree(node), node)
        self.tree.add_edge(parent, node)

    def remove_tree_node(self, node: int):
        parent = self.parent_in_tree(node)
        if parent is None:
            raise ValueError
        for child in self.tree.successors(node):
            self.tree.add_edges_from([(parent, child) for child in self.tree.successors(node)])
        self.tree.remove_node(node)

        del self.subsequence[node]

    def create_new_root(self):
        v = self.root
        self.root = self.add_tree_node()
        self.tree.add_edge(self.root, v)
        self.times_visited[self.root] = self.times_visited[v]
        self.subsequence[self.root] = self.subsequence[v].copy()
