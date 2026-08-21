# pattern: Functional Core

import importlib
import sys
import types

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest

_ADD_SAMPLE_MODULE_NAME = "linear_dag.core.add_sample"
_MISSING = object()


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


@contextmanager
def _preserve_add_sample_import_state() -> Iterator[None]:
    core_package = importlib.import_module("linear_dag.core")
    cached_module = sys.modules.pop(_ADD_SAMPLE_MODULE_NAME, _MISSING)
    package_attribute = vars(core_package).pop("add_sample", _MISSING)
    try:
        yield
    finally:
        sys.modules.pop(_ADD_SAMPLE_MODULE_NAME, None)
        vars(core_package).pop("add_sample", None)
        if cached_module is not _MISSING:
            sys.modules[_ADD_SAMPLE_MODULE_NAME] = cast(types.ModuleType, cached_module)
        if package_attribute is not _MISSING:
            setattr(core_package, "add_sample", package_attribute)


@contextmanager
def _import_add_sample_with_optional_dependency_stubs() -> Iterator[types.ModuleType]:
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    networkx = types.ModuleType("networkx")
    setattr(matplotlib, "pyplot", pyplot)
    with _preserve_add_sample_import_state():
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
            monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
            monkeypatch.setitem(sys.modules, "networkx", networkx)
            yield importlib.import_module(_ADD_SAMPLE_MODULE_NAME)


@pytest.fixture(autouse=True)
def _restore_add_sample_import_state() -> Iterator[None]:
    with _preserve_add_sample_import_state():
        yield


@pytest.fixture
def add_sample_module() -> Iterator[types.ModuleType]:
    with _import_add_sample_with_optional_dependency_stubs() as add_sample:
        yield add_sample


def test_get_predecessors_recurses_through_graph_predecessors(add_sample_module: types.ModuleType) -> None:
    add_sample = add_sample_module
    helper = object.__new__(add_sample.linarg_add_sample)
    helper.G = _PredecessorGraph()

    predecessors = helper.get_predecessors(2)

    assert predecessors == {0, 1, 3, 4}


def test_optional_dependency_stubs_do_not_contaminate_subsequent_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored_matplotlib = types.ModuleType("matplotlib")
    restored_pyplot = types.ModuleType("matplotlib.pyplot")
    restored_networkx = types.ModuleType("networkx")
    setattr(restored_matplotlib, "pyplot", restored_pyplot)
    monkeypatch.setitem(sys.modules, "matplotlib", restored_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", restored_pyplot)
    monkeypatch.setitem(sys.modules, "networkx", restored_networkx)

    with _import_add_sample_with_optional_dependency_stubs() as stubbed_add_sample:
        assert stubbed_add_sample.plt is not restored_pyplot
        assert stubbed_add_sample.nx is not restored_networkx

    subsequently_imported = importlib.import_module(_ADD_SAMPLE_MODULE_NAME)

    assert subsequently_imported.plt is restored_pyplot
    assert subsequently_imported.nx is restored_networkx
