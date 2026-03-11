import itertools
import random

from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from scipy.sparse import coo_matrix, eye

random.seed(0)


@dataclass
class Intervals:
    """Collection of interval endpoints with associated candidate identifiers.

    This helper stores interval metadata used during candidate-brick search for
    sample insertion routines.

    !!! Example

        ```python
        intervals = Intervals(
            left_bound=np.array([0, 2]),
            right_bound=np.array([2, 4]),
            identifiers=np.array([10, 11]),
        )
        cover = intervals.minimal_disjoint_cover()
        ```
    """

    left_bound: np.ndarray
    right_bound: np.ndarray
    identifiers: np.ndarray

    def sort_intervals_by_right_bound(self):
        """Sort all stored intervals in place by ascending right endpoint.

        **Returns:**

        - `None`.
        """
        order = np.argsort(self.right_bound)
        self.right_bound = self.right_bound[order]
        self.left_bound = self.left_bound[order]
        self.identifiers = self.identifiers[order]

    def minimal_disjoint_cover(self):
        """Return one minimal disjoint cover across the represented span.

        **Returns:**

        - List of interval identifiers forming a minimal cover.
        """
        self.sort_intervals_by_right_bound()
        interval_start = np.min(self.left_bound)
        min_n_intervals = defaultdict(lambda: np.inf)
        min_n_intervals[interval_start] = 0
        mdc = defaultdict(lambda: [])  # keep track of all minimal disjoint covers
        for lb, rb, i in zip(self.left_bound, self.right_bound, self.identifiers):
            if min_n_intervals[lb] + 1 < min_n_intervals[rb]:  # add interval i to solution
                min_n_intervals[rb] = min_n_intervals[lb] + 1
                mdc[rb] = mdc[lb].copy()
                mdc[rb].append(i)
        return mdc[rb]


class linarg_add_sample:
    """Helper object for attaching a new haplotype to a LinearARG.

    The object builds NetworkX views of reduced and full ancestor graphs, then
    exposes search utilities for finding candidate parental bricks and
    reconstructing the corresponding path-sum vector.

    !!! Example

        ```python
        helper = make_linarg_add_sample(linarg, linarg_recom)
        parent_mutations, candidates, cover, intervals, coeffs = helper.add_sample(haplotype)
        ```
    """

    def __init__(self, A_reduced, A, variant_indices_reduced, variant_indices, A_lin):
        self.A_reduced = A_reduced
        self.G_reduced = nx.from_numpy_array(A_reduced, create_using=nx.DiGraph)
        self.A = A
        self.G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        self.variant_indices_reduced = variant_indices_reduced
        self.variant_indices = variant_indices
        self.A_lin = A_lin  # full adjacency matrix with -1 edges

    def get_successors(self, mut):
        """Return all descendants of a reduced-graph mutation node.

        **Arguments:**

        - `mut`: Mutation-node index in `self.G_reduced`.

        **Returns:**

        - Set of descendant node indices reachable from `mut`.
        """
        return {item for sublist in list(nx.dfs_successors(self.G_reduced, mut).values()) for item in sublist}

    def get_parent_mutations(self, haplotype):
        """Partition haplotype mutations into parent and non-parent sets.

        **Arguments:**

        - `haplotype`: Iterable of mutation indices relative to the reduced graph.

        **Returns:**

        - Tuple `(parent_mutations, nonparent_mutations)`.
        """
        haplotype_mapped = set([self.variant_indices_reduced[m] for m in haplotype])  # variants indices wrt A_reduced
        parent_mutations = [
            x
            for x in haplotype
            if len(self.get_successors(self.variant_indices_reduced[x]).intersection(haplotype_mapped)) == 0
        ]
        nonparent_mutations = [x for x in haplotype if x not in parent_mutations]
        return parent_mutations, nonparent_mutations

    def get_interval(self, parent_intervals):
        """Merge parent intervals when they define a contiguous span.

        **Arguments:**

        - `parent_intervals`: Sequence of interval tuples, empty tuples, or `None`.

        **Returns:**

        - Interval tuple `(left, right)` for a contiguous merge.
        - Empty tuple `()` when all intervals are empty.
        - `None` when the merge is invalid or unresolved.
        """
        if len(parent_intervals) == 1:
            return parent_intervals[0]
        if None in parent_intervals:  # some parent of the child either has not been visited or is not a candidate
            return None
        parent_intervals = [i for i in parent_intervals if i != ()]
        if len(parent_intervals) == 0:  # all intervals were empty
            return ()
        sorted_intervals = [parent_intervals[i] for i in np.argsort([interval[0] for interval in parent_intervals])]
        for i in range(len(sorted_intervals) - 1):
            current_int = sorted_intervals[i]
            next_int = sorted_intervals[i + 1]
            if current_int[1] < next_int[0]:  # non-contiguous
                return None
        child_interval = (sorted_intervals[0][0], np.max([interval[1] for interval in parent_intervals]))
        return child_interval

    def find_candidates(self, haplotype):
        """Find candidate ancestor nodes covering a haplotype's mutations.

        **Arguments:**

        - `haplotype`: Iterable of mutation indices in reduced-graph coordinates.

        **Returns:**

        - Tuple `(candidate_intervals, parent_mutations, interval_map)` where
          `candidate_intervals` is an [`linear_dag.core.add_sample.Intervals`][]
          instance.
        """
        haplotype_mapped = [self.variant_indices[m] for m in sorted(haplotype)]  # variants indices wrt A
        parent_mutations, nonparent_mutations = self.get_parent_mutations(haplotype)  # order by genomic location
        parent_mutations_mapped = [
            self.variant_indices[m] for m in sorted(parent_mutations)
        ]  # parent mutation indices wrt A
        nonparent_mutations_mapped = [self.variant_indices[m] for m in sorted(nonparent_mutations)]
        nodes_to_search = haplotype_mapped.copy()
        interval = defaultdict(lambda: None)
        interval.update({parent_mutations_mapped[i]: (i, i + 1) for i in range(len(parent_mutations_mapped))})
        interval.update({npm: () for npm in nonparent_mutations_mapped})
        while nodes_to_search:
            node = nodes_to_search.pop()
            for child in self.G.successors(node):
                if interval[child] is not None:  # child has already been added to candidates
                    continue
                if (child in self.variant_indices) and (
                    child not in haplotype_mapped
                ):  # contains mutation not in h, consider asserting the second statement
                    continue
                child_interval = self.get_interval(
                    [interval[parent] for parent in self.G.predecessors(child)]
                )  # returns none if non-contiguous
                if child_interval is not None:
                    nodes_to_search.append(child)
                    interval[child] = child_interval
        candidates = [node for node in interval.keys() if (interval[node] is not None) and (interval[node] != ())]
        left_bound = np.array([interval[c][0] for c in candidates])
        right_bound = np.array([interval[c][1] for c in candidates])
        candidate_intervals = Intervals(
            left_bound=left_bound, right_bound=right_bound, identifiers=np.array(candidates)
        )
        # return candidate_intervals
        return candidate_intervals, parent_mutations, interval

    def minimal_disjoint_cover(self, candidate_intervals):
        """Return a minimal disjoint cover for candidate intervals.

        **Arguments:**

        - `candidate_intervals`: Candidate interval collection.

        **Returns:**

        - List of candidate identifiers selected for the cover.
        """
        mdc = candidate_intervals.minimal_disjoint_cover()
        return mdc

    def get_predecessors(self, node, visited=None):
        """Return recursive predecessors of `node`.

        **Arguments:**

        - `node`: Node whose ancestors should be collected.
        - `visited`: Optional accumulator set.

        **Returns:**

        - Set of predecessor node indices.
        """
        if visited is None:
            visited = set()
        predecessors = set(self.predecessors(node))
        for predecessor in predecessors:
            if predecessor not in visited:
                visited.add(predecessor)
                visited.update(get_all_predecessors(self, predecessor, visited))
        return visited

    def correct_path_sum(self, mdc):
        """Compute the corrected path-sum vector for a chosen cover.

        **Arguments:**

        - `mdc`: Minimal disjoint cover of candidate brick identifiers.

        **Returns:**

        - CSC column vector of path-sum coefficients.
        """
        predecessors = set()
        for b in mdc:
            b_predecessors = get_all_predecessors(self.G, b)
            b_predecessors.add(b)  # include brick itself
            predecessors.update(b_predecessors)
        h_tilde = coo_matrix(
            ([1] * len(predecessors), (list(predecessors), [0] * len(predecessors))), shape=(self.A.shape[0], 1)
        )
        h_tilde = h_tilde.tocsc()
        X_inv = eye(self.A_lin.shape[0]) - self.A_lin
        return X_inv @ h_tilde

    def add_sample(self, haplotype):
        """Run candidate search and path-sum reconstruction for one haplotype.

        **Arguments:**

        - `haplotype`: Iterable of mutation indices in reduced-graph coordinates.

        **Returns:**

        - Tuple `(parent_mutations, candidate_ids, cover, interval_map, path_sum)`.
        """
        candidate_intervals, parent_mutations, interval = self.find_candidates(haplotype)
        mdc = self.minimal_disjoint_cover(candidate_intervals)
        a = self.correct_path_sum(mdc)
        return parent_mutations, list(candidate_intervals.identifiers), mdc, interval, a
        # return parent_mutations, list(candidate_intervals.identifiers), mdc, interval


def get_ancestral_adjacency_matrix(linarg, mask_negs=True):
    """Extract the ancestor-only adjacency matrix from a LinearARG-like object.

    **Arguments:**

    - `linarg`: Object exposing `.A`, `.sample_indices`, and `.variant_indices`.
    - `mask_negs`: Whether to zero out negative edges in the returned matrix.

    **Returns:**

    - Tuple `(A_anc, variant_indices)` where `A_anc` is a CSR ancestor graph.
    """
    A_anc = linarg.A.copy()
    A_anc = A_anc.T
    A_anc = A_anc.tocoo()
    mask = np.isin(A_anc.col, linarg.sample_indices)
    A_anc.data[mask] = 0  # zero out sample columns i.e. parents of each sample
    A_anc.eliminate_zeros()
    if mask_negs:
        A_anc.data[A_anc.data < 0] = 0
    A_anc = A_anc.tocsr()
    return A_anc, linarg.variant_indices


def make_linarg_add_sample(linarg, linarg_recom):
    """Construct a [`linear_dag.core.add_sample.linarg_add_sample`][] helper.

    **Arguments:**

    - `linarg`: Reduced LinearARG used to identify parent mutations.
    - `linarg_recom`: Recombination-expanded LinearARG.

    **Returns:**

    - Initialized helper object for sample insertion workflows.
    """
    A_reduced, variant_indices_reduced = get_ancestral_adjacency_matrix(linarg)
    A, variant_indices = get_ancestral_adjacency_matrix(linarg_recom)
    A_lin, _ = get_ancestral_adjacency_matrix(linarg_recom, mask_negs=False)
    linarg_sample = linarg_add_sample(A_reduced, A, variant_indices_reduced, variant_indices, A_lin)
    return linarg_sample


def get_all_predecessors(graph, node, visited=None):
    """Return all recursive predecessors of `node` in a directed graph.

    **Arguments:**

    - `graph`: Graph object exposing `.predecessors(node)`.
    - `node`: Node whose ancestors should be collected.
    - `visited`: Optional accumulator set.

    **Returns:**

    - Set of predecessor nodes reachable by reverse traversal.
    """
    if visited is None:
        visited = set()
    predecessors = set(graph.predecessors(node))
    for predecessor in predecessors:
        if predecessor not in visited:
            visited.add(predecessor)
            visited.update(get_all_predecessors(graph, predecessor, visited))
    return visited


def plot_mutations(true_a, h, linarg_sample):
    """Plot haplotype mutations explained by inferred parental bricks.

    **Arguments:**

    - `true_a`: Iterable of selected brick or node identifiers.
    - `h`: Haplotype mutation identifiers ordered by genomic position.
    - `linarg_sample`: Sample-insertion helper object.

    **Returns:**

    - Dictionary mapping brick identifiers to covered mutation positions.
    """
    ind_to_var = {
        linarg_sample.variant_indices[i]: i for i in range(len(linarg_sample.variant_indices))
    }  # map index on adjacency matrix to variant index
    var_to_haplo = {sorted(h)[i]: i for i in range(len(h))}  # map variant index to ith mutation on haplotype
    preds = {}  # store bricks and their mutations
    for i in range(len(true_a)):
        predecessors = get_all_predecessors(linarg_sample.G, true_a[i])
        if true_a[i] in linarg_sample.variant_indices:  # include node itself if it has a variant
            predecessors.add(true_a[i])
        var = [ind_to_var[x] for x in predecessors if x in ind_to_var.keys()]
        muts = [var_to_haplo[x] for x in var if x in var_to_haplo.keys()]
        if len(muts) == 0:
            continue
        preds[true_a[i]] = muts

    pred_sorted = dict(sorted(preds.items(), key=lambda item: min(item[1])))
    b = list(
        itertools.chain(*[[i] * len(pred_sorted[list(pred_sorted.keys())[i]]) for i in range(len(pred_sorted.keys()))])
    )
    b_mutations = list(itertools.chain(*[pred_sorted[b] for b in pred_sorted.keys()]))

    plt.figure(figsize=(6, 8))
    plt.scatter(b_mutations, b, s=1, c="blue")
    plt.xlabel("mutations on h sorted by genomic location")
    plt.ylabel("parental bricks of h")
    ax = plt.gca()
    ax.set_yticks(np.arange(len(list(pred_sorted.keys()))))
    ax.set_yticklabels(list(pred_sorted.keys()))
    plt.show()
    return pred_sorted


def is_contiguous(muts):
    """Return whether mutation indices form a contiguous integer run.

    **Arguments:**

    - `muts`: Iterable of integer mutation indices.

    **Returns:**

    - `True` when every integer between `min(muts)` and `max(muts)` is present.
    """
    if len(muts) == 0:
        return True
    min_val = min(muts)
    max_val = max(muts)
    expected_length = max_val - min_val + 1
    return len(muts) == expected_length
