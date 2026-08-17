# pattern: Functional Core

from __future__ import annotations

import inspect

from typing import TypedDict

import jax
import jax.tree_util as jtu
import numpy as np
import pytest

from jax.sharding import Mesh

import linear_dag
import linear_dag.core as core_module
import linear_dag.core.jaxlinarg as jaxlinarg_module
import linear_dag.core.jaxlinarg.ingress as ingress_module

from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxLinearARG, JaxParallelOperator
from linear_dag.core.jaxlinarg.ingress import _PackedJaxLinearARG
from linear_dag.core.lineararg import LinearARG


class _PromotionChecklist(TypedDict):
    constructors: tuple[tuple[str, tuple[str, ...]], ...]
    methods: tuple[str, ...]
    metadata: tuple[str, ...]
    remaining_changes: tuple[str, ...]


_PROMOTION_CHECKLIST: _PromotionChecklist = {
    "constructors": (
        (
            "from_lineararg_arrays",
            (
                "indptr",
                "indices",
                "data",
                "variant_indices",
                "flip",
                "sample_indices",
                "nonunique_indices",
                "n_variants",
                "n_samples",
                "allele_counts",
                "iids",
                "mesh",
                "backend",
                "dtype",
                "max_padding_ratio",
            ),
        ),
        ("from_lineararg", ("linarg", "mesh", "backend", "dtype", "max_padding_ratio")),
        ("from_linearargs", ("lineargs", "mesh", "backend", "dtype", "max_padding_ratio")),
        (
            "from_hdf5_block",
            ("path", "block", "mesh", "backend", "load_metadata", "dtype", "max_padding_ratio"),
        ),
        ("from_hdf5", ("path", "mesh", "block_metadata", "backend", "dtype", "max_padding_ratio")),
    ),
    "methods": ("matmat", "rmatmat", "compile_matmat", "compile_rmatmat"),
    "metadata": ("shape", "dtype", "iids", "backend"),
    "remaining_changes": (
        "rename the public exact single-block implementation to a private compatibility name",
        "rename _PackedJaxLinearARG to JaxLinearARG and update package exports",
        "update public annotations and JaxParallelOperator fallback routing",
        "reroute --jax-backend only in a separately approved migration",
    ),
}


def _graph_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:1]), ("graph",))


def _blocks_mesh() -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:1]), ("blocks",))


def test_private_candidate_matches_promotion_surface_without_changing_class_identity(
    linarg_h5_path,
    first_block_name,
) -> None:
    packed = _PackedJaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name)
    exact = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.PURE_JAX)

    for name, expected_parameters in _PROMOTION_CHECKLIST["constructors"]:
        assert tuple(inspect.signature(getattr(_PackedJaxLinearARG, name)).parameters) == expected_parameters
    assert _PROMOTION_CHECKLIST["methods"] == tuple(
        name for name in _PROMOTION_CHECKLIST["methods"] if callable(getattr(packed, name))
    )
    assert all(hasattr(packed, name) for name in _PROMOTION_CHECKLIST["metadata"])
    assert _PROMOTION_CHECKLIST["remaining_changes"]
    assert type(packed) is _PackedJaxLinearARG
    assert type(exact) is JaxLinearARG
    assert packed.__class__.__name__ == "_PackedJaxLinearARG"
    assert exact.__class__.__name__ == "JaxLinearARG"


def test_public_modules_annotations_and_pytrees_do_not_expose_private_packed_or_hijax_names(
    linarg_h5_path,
    first_block_name,
) -> None:
    exact = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.PURE_JAX)
    public_modules = (linear_dag, core_module, jaxlinarg_module)
    public_objects = (JaxLinearARG, JaxParallelOperator, JaxGRMOperator)

    for module in public_modules:
        public_names = tuple(name for name in vars(module) if not name.startswith("_"))
        explicit_exports = tuple(getattr(module, "__all__", ()))
        assert not any("Packed" in name or "HiJAX" in name or "hijax" in name for name in public_names)
        assert not any("Packed" in name or "HiJAX" in name or "hijax" in name for name in explicit_exports)
    for public_object in public_objects:
        annotations = inspect.get_annotations(public_object, eval_str=False)
        signature = inspect.signature(public_object)
        inventory = f"{annotations} {signature} {inspect.getdoc(public_object) or ''}"
        assert "_PackedJaxLinearARG" not in inventory
        assert "_hijax" not in inventory
        assert "HiJAX" not in inventory

    assert all(isinstance(leaf, jax.Array) for leaf in jtu.tree_leaves(exact))


def test_packed_and_exact_ragged_products_and_grm_coexist(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    packed = _PackedJaxLinearARG.from_hdf5(
        linarg_h5_path,
        mesh=_graph_mesh(),
        block_metadata=linarg_block_metadata,
    )
    exact = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_blocks_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    rng = np.random.default_rng(20260817)
    weights = rng.normal(size=(packed.shape[1], 2)).astype(np.float32)
    samples = rng.normal(size=(packed.shape[0], 2)).astype(np.float32)

    np.testing.assert_allclose(
        np.asarray(packed.matmat(weights)),
        np.asarray(exact.matmat(weights)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(packed.rmatmat(samples)),
        np.asarray(exact.rmatmat(samples)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(JaxGRMOperator(packed).matmat(samples)),
        np.asarray(JaxGRMOperator(exact).matmat(samples)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_private_padding_failure_propagates_without_exact_ragged_fallback(
    linarg_h5_path,
    first_block_name,
    monkeypatch,
) -> None:
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    expected = ValueError("whole-block packing exceeds max_padding_ratio")

    def fail_packed(*args, **kwargs):
        raise expected

    def fail_exact(*args, **kwargs):
        raise AssertionError("private packed construction must not fall back to exact-ragged")

    monkeypatch.setattr(ingress_module, "_packed_from_block_arrays", fail_packed)
    monkeypatch.setattr(JaxParallelOperator, "from_linearargs", fail_exact)

    with pytest.raises(ValueError, match="whole-block packing exceeds") as exc_info:
        _PackedJaxLinearARG.from_linearargs((linarg,), mesh=_graph_mesh())

    assert exc_info.value is expected


def test_promotion_checklist_names_remaining_public_migration_work() -> None:
    assert set(_PROMOTION_CHECKLIST) == {"constructors", "methods", "metadata", "remaining_changes"}
    assert "JaxLinearARG" in " ".join(_PROMOTION_CHECKLIST["remaining_changes"])
    assert "exports" in " ".join(_PROMOTION_CHECKLIST["remaining_changes"])
    assert "--jax-backend" in " ".join(_PROMOTION_CHECKLIST["remaining_changes"])
