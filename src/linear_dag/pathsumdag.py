import networkx as nx
import numpy as np

from numpy.typing import NDArray

from .lineararg import LinearARG


# noinspection PyCallingNonCallable
class PathSumDAG:
    """
    Implements a weighted directed acyclic graph with path sum-preserving operations.
    """

    g: nx.DiGraph
    last_predecessor: NDArray[np.int_]
    original_nodes: set[int]
    node_position: NDArray[np.int_]
    next_node_index: int

    @property
    def nodes(self) -> set[int]:
        return set(self.g.nodes())

    @property
    def number_of_nodes(self) -> int:
        return self.g.number_of_nodes()

    @property
    def number_of_edges(self) -> int:
        return self.g.number_of_edges()

    def __init__(self, directed_graph=None, last_predecessor=None, original_nodes=None, node_position=None):
        self.g = directed_graph
        self.last_predecessor = last_predecessor
        self.original_nodes = original_nodes
        self.node_position = node_position
        self.next_node_index = int(np.max(directed_graph.nodes()) + 1)

    @classmethod
    def from_digraph(cls, directed_graph: nx.DiGraph) -> "PathSumDAG":
        m = directed_graph.number_of_edges()
        last_predecessor = -np.ones(m, dtype=np.intc)
        original_nodes = set(directed_graph.nodes)
        node_position = np.zeros(m)
        node_position[: directed_graph.number_of_nodes()] = np.arange(directed_graph.number_of_nodes())

        return cls(
            directed_graph=directed_graph,
            last_predecessor=last_predecessor,
            original_nodes=original_nodes,
            node_position=node_position,
        )

    @classmethod
    def from_lineararg(cls, lineararg: "LinearARG") -> "PathSumDAG":
        D = nx.from_scipy_sparse_array(lineararg.A.T, create_using=nx.DiGraph, edge_attribute="weight")
        return cls.from_digraph(D)

    def to_csr_matrix(self):
        A = nx.to_scipy_sparse_array(self.g, format="csc", weight="weight")
        return A.T

    def iterate(self, threshold: int = 1) -> None:
        """
        Iterate over all nodes in the graph twice, first performing an unweighting operation and then processing them.
        """

        if threshold < 1:
            raise ValueError("Recombination threshold parameter should be at least 1")

        for node in list(self.g.nodes()):
            self.unweight(node)

        self.last_predecessor[:] = -1
        order = np.argsort(self.node_position)
        for i in order:
            if not self.g.has_node(i):
                continue
            self.process_node(i, threshold=threshold)

    def process_node(self, u: int, threshold: int = 1) -> None:
        """
        Processes outgoing edges from node u in DiGraph G:
        - Finds unique last-visited predecessors of successors of u.
        - Factor bicliques sharing more than n children, where n is the threshold parameter.
        - Removes predecessors if they have an out-degree of 1 (and are not in the original set of nodes), and removes
        children if they have an in-degree of 1 (and are not in the original set of nodes)
        """

        successors = [v for v in self.g.successors(u) if self.g.edges[u, v]["weight"] == 1]

        # successors of v shared with each last predecessor
        shared_successors_with = {}
        for v in successors:
            p = self.last_predecessor[v]
            if p not in shared_successors_with:
                shared_successors_with[p] = []
            shared_successors_with[p].append(v)

        # Process each unique predecessor
        for p, shared_successors in shared_successors_with.items():
            if p == -1:
                self.last_predecessor[shared_successors] = u
                continue

            if len(shared_successors) <= threshold:
                self.last_predecessor[shared_successors] = u
                continue

            self.factor((p, u), shared_successors)

            # Check and potentially remove the predecessor
            if self.g.out_degree(p) == 1 and p not in self.original_nodes:
                self.remove_node(p)

        for s in successors:
            if self.g.in_degree(s) == 1 and s not in self.original_nodes:
                self.remove_node(s)

    def create_node(self, position: int) -> int:
        n = self.next_node_index  # Ensuring the new node has a unique index
        self.next_node_index += 1
        self.g.add_node(n)
        self.node_position[n] = position
        return n

    def factor(self, U: tuple[int, int], V: list[int]) -> None:
        """
        Factor a biclique (U,V), where (u,v) is in G for all u in U, v in V, creating a new node n with edges (u,n)
        and (n,v)
        """
        n = self.create_node(self.node_position[U[-1]])

        for v in V:
            self.g.add_edge(n, v, weight=self.g.edges[U[0], v]["weight"])

        for u in U:
            self.g.add_edge(u, n, weight=1)
            for v in V:
                assert self.g.has_edge(u, v), f"Problem with nodes {u} and {v}"
                self.g.remove_edge(u, v)

        self.last_predecessor[n] = U[-1]

        # Update last_predecessor for each v in V to the new node n
        for v in V:
            self.last_predecessor[v] = n

    def unweight(self, u: int) -> None:
        """
        Takes outgoing edges (u,v) of u having weight w != 1 and replaces them with an edge (n_w,v) having weight 1 for
        a new node n_w, together with a single edge (u,n_w) having weight w.
        """
        # Outgoing edges from each node with given weights
        successors_with_weight = {}
        for v in self.g.successors(u):
            w = self.g.edges[u, v]["weight"]
            if w not in successors_with_weight:
                successors_with_weight[w] = []
            successors_with_weight[w].append(v)

        for w, successors in successors_with_weight.items():
            if w == 1 or len(successors) <= 1:
                continue

            n = self.create_node(self.node_position[u])

            for s in successors:
                self.g.add_edge(n, s, weight=1)
                self.g.remove_edge(u, s)
                if self.last_predecessor[s] == u:
                    self.last_predecessor[s] = n

            self.g.add_edge(u, n, weight=w)

    def remove_node(self, node: int) -> None:
        """
        Removes a node and patches paths passing through it such that path sums are preserved.
        Also updates self.last_predecessor.
        """

        predecessors = list(self.g.predecessors(node))
        successors = list(self.g.successors(node))

        # Bypass the node
        for p in predecessors:
            for s in successors:
                if not self.g.has_edge(p, s):
                    self.g.add_edge(p, s, weight=0)

                self.g.edges[p, s]["weight"] += self.g.edges[p, node]["weight"] * self.g.edges[node, s]["weight"]

                if self.g.edges[p, s]["weight"] == 0:
                    self.g.remove_edge(p, s)

        # Update last_predecessor
        for s in successors:
            if self.last_predecessor[s] == node:
                self.last_predecessor[s] = predecessors[0] if predecessors else -1

        # Delete the node
        self.g.remove_node(node)

    def cleanup(self) -> None:
        """
        Remove all nodes having either in- or out-degree one
        """
        for node in list(self.g.nodes):
            if node in self.original_nodes:
                continue
            if self.g.in_degree(node) == 1 or self.g.out_degree(node) == 1:
                self.remove_node(node)
