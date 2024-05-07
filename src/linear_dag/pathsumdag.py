from functools import partial
from itertools import groupby

import networkit as nk
import numpy as np

from networkit.algebraic import adjacencyMatrix
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from .lineararg import LinearARG


# noinspection PyCallingNonCallable
class PathSumDAG:
    """
    Implements a weighted directed acyclic graph with path sum-preserving operations.
    """

    g: nk.Graph
    last_predecessor: NDArray[np.int_]
    original_nodes: set[int]
    node_position: NDArray[np.int_]

    @property
    def nodes(self) -> set[int]:
        return set(self.g.iterNodes())

    @property
    def number_of_nodes(self) -> int:
        return self.g.numberOfNodes()

    @property
    def number_of_edges(self) -> int:
        return self.g.numberOfEdges()

    def __init__(self, directed_graph=None, last_predecessor=None, original_nodes=None, node_position=None):
        self.g = directed_graph if directed_graph else nk.Graph()
        self.last_predecessor = last_predecessor
        self.original_nodes = original_nodes
        self.node_position = node_position

    @classmethod
    def from_digraph(cls, directed_graph: "nk.Graph") -> "PathSumDAG":
        m = directed_graph.numberOfEdges()
        last_predecessor = -np.ones(m, dtype=np.intc)
        original_nodes = set(directed_graph.iterNodes())
        node_position = np.zeros(m)
        n = directed_graph.numberOfNodes()
        node_position[:n] = np.arange(n)

        return cls(
            directed_graph=directed_graph,
            last_predecessor=last_predecessor,
            original_nodes=original_nodes,
            node_position=node_position,
        )

    @classmethod
    def from_lineararg(cls, lineararg: "LinearARG") -> "PathSumDAG":
        D = nk.GraphFromCoo(coo_matrix(lineararg.A.T).astype(float), weighted=True, directed=True)
        return cls.from_digraph(D)

    def to_csr_matrix(self):
        A = adjacencyMatrix(self.g)
        return csr_matrix(A.T)

    def unweight_all(self) -> None:
        for node in list(self.g.iterNodes()):
            self.unweight_node(node)

    def recombine_all(self, threshold: int = 1) -> None:
        """
        Iterate over all nodes in the graph twice and recombine them.
        """

        if threshold < 1:
            raise ValueError("Recombination threshold parameter should be at least 1")

        self.last_predecessor[:] = -1
        order = np.argsort(self.node_position)
        for i in order:
            if not self.g.hasNode(i):
                continue
            if self.g.degreeOut(i) == 0:
                continue
            self.recombine_node(i, threshold=threshold)

    def recombine_node(self, u: int, threshold: int = 1) -> None:
        """
        Processes outgoing edges from node u in DiGraph G:
        - Finds unique last-visited predecessors of successors of u.
        - Factor bicliques sharing more than n children, where n is the threshold parameter.
        - Removes predecessors if they have an out-degree of 1 (and are not in the original set of nodes), and removes
        children if they have an in-degree of 1 (and are not in the original set of nodes)
        """

        successors = [v for v in self.g.iterNeighbors(u) if self.g.weight(u, v) == 1]

        # successors of v shared with each last predecessor
        def last_predecessor(v):
            return self.last_predecessor[v]

        sorted_successors = sorted(successors, key=last_predecessor)

        # Process each unique predecessor
        for p, shared_successors in groupby(sorted_successors, last_predecessor):
            shared_successors = list(shared_successors)
            if p == -1:
                self.last_predecessor[shared_successors] = u
                continue

            if len(shared_successors) <= threshold:
                self.last_predecessor[shared_successors] = u
                continue

            self.factor((p, u), shared_successors)

            # Check and potentially remove the predecessor
            if self.g.degreeOut(p) == 1 and p not in self.original_nodes:
                self.remove_node(p)

        for s in successors:
            if self.g.degreeIn(s) == 1 and s not in self.original_nodes:
                self.remove_node(s)

    def create_node(self, position: int) -> int:
        n: int = self.g.addNode()
        self.node_position[n] = position
        return n

    def factor(self, U: tuple[int, int], V: list[int]) -> None:
        """
        Factor a biclique (U,V), where (u,v) is in G for all u in U, v in V, creating a new node n with edges (u,n)
        and (n,v)
        """
        n = self.create_node(self.node_position[U[1]])

        for v in V:
            self.g.addEdge(n, v, w=self.g.weight(U[0], v))

        for u in U:
            for v in V:
                assert self.g.hasEdge(u, v), f"Problem with nodes {u} and {v}"
                self.g.removeEdge(u, v)

            self.g.addEdge(u, n, w=1)

        self.last_predecessor[n] = U[-1]
        self.last_predecessor[V] = n

    def unweight_node(self, u: int) -> None:
        """
        Takes outgoing edges (u,v) of u having weight w != 1 and replaces them with an edge (n_w,v) having weight 1 for
        a new node n_w, together with a single edge (u,n_w) having weight w.
        """
        # Outgoing edges from each node with given weights
        neighbor_to_weight = partial(self.g.weight, u)
        sorted_neighbors = sorted(self.g.iterNeighbors(u), key=neighbor_to_weight)
        for weight, neighbors in groupby(sorted_neighbors, neighbor_to_weight):
            if weight == 1:
                continue

            # TODO think about node position here
            n = self.create_node(self.node_position[u])

            for v in neighbors:
                self.g.addEdge(n, v, w=1)
                self.g.removeEdge(u, v)

            self.g.addEdge(u, n, w=weight)

    def remove_node(self, node: int) -> None:
        """
        Removes a node and patches paths passing through it such that path sums are preserved.
        Also updates self.last_predecessor.
        """

        # Bypass the node
        for p in self.g.iterInNeighbors(node):
            for s in self.g.iterNeighbors(node):
                edge_weight = self.g.weight(p, node) * self.g.weight(node, s)
                if not self.g.hasEdge(p, s):
                    self.g.addEdge(p, s, edge_weight)
                    continue

                edge_weight += self.g.weight(p, s)
                if edge_weight == 0:
                    self.g.removeEdge(p, s)
                else:
                    self.g.setWeight(p, s, edge_weight)

        # Update last_predecessor
        for s in self.g.iterNeighbors(node):
            if self.last_predecessor[s] == node:
                self.last_predecessor[s] = self.last_predecessor[node]
            if s not in self.original_nodes:
                self.node_position[s] = max(self.node_position[s], self.node_position[node])

        # Delete the node
        self.g.removeNode(node)
        self.last_predecessor[node] = -1

    def cleanup(self) -> None:
        """
        Remove all nodes having either in- or out-degree one
        """
        for node in self.g.iterNodes():
            if node in self.original_nodes:
                continue
            if self.g.degreeIn(node) == 1 or self.g.degreeOut(node) == 1:
                self.remove_node(node)
