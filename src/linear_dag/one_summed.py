import heapq

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np


@dataclass(order=True)
class SetNode:
    weight: int
    node: Any = field(compare=False)


class ActiveSet:
    act_heap: list[SetNode]
    act_set: set[int]

    def __init__(self, i: int, weight: int):
        node = SetNode(weight, i)
        self.act_heap = [node]
        self.act_set = {i}
        heapq.heapify(self.act_heap)

    def __contains__(self, i: int) -> bool:
        return i in self.act_set

    def __str__(self) -> str:
        return str(self.act_set)

    def __bool__(self) -> bool:
        return bool(self.act_set)

    def push(self, i: int, weight: int):
        node = SetNode(weight, i)
        heapq.heappush(self.act_heap, node)
        self.act_set.add(i)

    def pop(self) -> int:
        node = heapq.heappop(self.act_heap)
        self.act_set.remove(node.node)
        return node.node


def resolve_node(G: nx.DiGraph, node):
    # For each parent and child, update/create an edge with the appropriate weight
    for parent in G.predecessors(node):
        for child in G.successors(node):
            # Calculate the new weight
            new_weight = G[parent][node]["weight"] * G[node][child]["weight"]

            # If the edge already exists, add the new weight to the existing weight
            if G.has_edge(parent, child):
                G[parent][child]["weight"] += new_weight
            # Otherwise, create a new edge with the calculated weight
            else:
                G.add_edge(parent, child, weight=new_weight)

    # Finally, remove the node from the graph
    G.remove_node(node)

    return


def compute_path_sums(G: nx.DiGraph):
    nodes = list(G.nodes())
    n = len(nodes)
    path_sum = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for path in nx.all_simple_edge_paths(G, nodes[i], nodes[j]):
                path_prod = 1
                for edge in path:
                    path_prod *= G[edge[0]][edge[1]]["weight"]
                path_sum[i][j] += path_prod

    return path_sum


def construct_1_summed_DAG_slow(G: nx.DiGraph) -> nx.DiGraph:
    G_prime = G.copy()

    # Perform a topological sort of G and reverse the order
    sorted_nodes = reversed(list(nx.topological_sort(G)))
    topological_ordering = {node: index for index, node in enumerate(sorted_nodes)}

    for i in topological_ordering:
        L = ActiveSet(i, -topological_ordering[i])
        path_sum = {i: 1}

        while L:
            j = L.pop()
            for k in G_prime.successors(j):
                if k in L:
                    path_sum[k] += G_prime[j][k]["weight"]
                else:
                    L.push(k, -topological_ordering[k])
                    path_sum[k] = G_prime[j][k]["weight"]
            if path_sum[j] != 1:
                if not G_prime.has_edge(i, j):
                    G_prime.add_edge(i, j, weight=0)
                if not G_prime.has_edge(i, j):
                    G_prime.add_edge(i, j, weight=0)

                G_prime[i][j]["weight"] += 1 - path_sum[j]

    return G_prime


def construct_1_summed_DAG_fast(G: nx.DiGraph) -> nx.DiGraph:
    G_prime = G.copy()

    # Perform a topological sort of G and reverse the order
    sorted_nodes = reversed(list(nx.topological_sort(G)))
    topological_ordering = {node: index for index, node in enumerate(sorted_nodes)}

    for i in topological_ordering:
        L = ActiveSet(i, -topological_ordering[i])
        path_sum = {i: 1}

        # entry nodes either have an incoming path from outside the descendents of i D(i),
        # or they have incoming paths within D(i) from >1 unique entry nodes such that
        # those paths do not pass through a different entry node
        entry_node = {i: None}

        while L:
            j = L.pop()
            nodes = [entry_node[k] if k in entry_node else k for k in G.predecessors(j)]
            unique_nodes = set([node for node in nodes if node is not None])
            if len(unique_nodes) > 1:
                entry_node[j] = j
            elif len(unique_nodes) == 1:
                entry_node[j] = unique_nodes.pop()
            else:
                entry_node[j] = None

            for k in G.successors(j):
                if k in L:
                    path_sum[k] += G[j][k]["weight"]
                else:
                    L.push(k, -topological_ordering[k])
                    path_sum[k] = G[j][k]["weight"]
            if path_sum[j] != 1:
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=0)
                if not G_prime.has_edge(i, j):
                    G_prime.add_edge(i, j, weight=0)

                G[i][j]["weight"] += 1 - path_sum[j]
                G_prime[i][j]["weight"] += 1 - path_sum[j]

            if entry_node[j] is None:
                if j != i:
                    resolve_node(G, j)
            elif entry_node[j] != j and -topological_ordering[entry_node[j]] > -topological_ordering[i] and j != i:
                resolve_node(G, j)

        if entry_node[i] is None:
            resolve_node(G, i)

    return G_prime
