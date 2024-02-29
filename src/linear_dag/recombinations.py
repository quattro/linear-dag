from collections import defaultdict

import numpy as np
import scipy.sparse as sparse


def _index_array(data):
    # maps each element in data to a set, containing all indices where it shows up
    cells = defaultdict(set)
    for idx, val in enumerate(data):
        cells[val].add(idx)

    return cells


def find_recombinations(A):
    n, n = A.shape

    trioList = np.zeros(shape=(A.nnz, 5))
    edgeList = np.zeros(shape=(A.nnz, 3))

    nTrio = 0
    nEdge = 0
    B = A.T

    for child in range(n):
        rowParents, _, rowWeights = sparse.find(B[:, [child]])
        if len(rowWeights) > 0:
            weight_sets = _index_array(rowWeights)
            for value, idxs in weight_sets.items():
                rows = list(idxs)
                these_parents = rowParents[rows]
                if len(these_parents) > 1:
                    for pp in range(1, len(these_parents)):
                        trioList[nTrio, 0:4] = np.asarray(
                            [these_parents[pp - 1], these_parents[pp], child, rowWeights[rows[pp]]]
                        )
                        nTrio += 1
                elif len(these_parents) == 1:
                    edgeList[nEdge, :] = np.asarray([these_parents.squeeze(), child, rowWeights[rows].squeeze()])
                    nEdge += 1

    # drop non-obs
    trioList = trioList[0:nTrio, :]
    edgeList = edgeList[0:nEdge, :]

    # not sure if needed atm...
    # trioList = sortrows(trioList);

    # [cliques, ~, which_clique] = unique(trioList(:, 1: 2), 'rows');
    # map parent-pairs to row indices where they show up
    cliqueRows = _index_array(trioList[:, 0:2])
    cliqueSize = dict()

    # add lookup back in trioList to parent-pair ID (here, jdx)
    for jdx, (parent_pair, idxs) in enumerate(cliqueRows.items()):
        idxs = list(idxs)
        trioList[idxs, 4] = jdx
        cliqueSize[parent_pair] = len(idxs)

    # trioList(:, 5) = which_clique;
    """
    nClique = len(cliqueRows)

    # Lookup tables for  rows of trio list
    parentRows = [_index_array(trioList[:, 0]), _index_array(trioList[:, 1])]
    childRows = _index_array(trioList[:, 2])

    newNode = n
    loopcounter = 0
    """
    return
