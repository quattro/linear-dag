# pattern: Functional Core

import importlib
import sys
import types

import pytest


class _PredecessorGraph:
    def __init__(self) -> None:
        self._predecessors = {
            0: (),
            1: (0,),
            2: (1, 4),
            3: (0,),
            4: (3,),
        }

    def predecessors(self, node: int) -> tuple[int, ...]:
        return self._predecessors[node]


def test_get_predecessors_recurses_through_graph_predecessors(monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    networkx = types.ModuleType("networkx")
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "networkx", networkx)
    add_sample = importlib.import_module("linear_dag.core.add_sample")
    helper = object.__new__(add_sample.linarg_add_sample)
    helper.G = _PredecessorGraph()

    predecessors = helper.get_predecessors(2)

    assert predecessors == {0, 1, 3, 4}
