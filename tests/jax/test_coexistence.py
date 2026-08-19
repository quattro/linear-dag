# pattern: Functional Core

from __future__ import annotations

import inspect

from typing import Any, TypedDict

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
from linear_dag.core.jaxlinarg.packing import BLOCK_DESCRIPTOR_FIELDS
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
                "n_nonunique_indices",
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
    "methods": (
        "matmat",
        "rmatmat",
        "matvec",
        "rmatvec",
        "transpose_view",
        "__matmul__",
        "compile_matmat",
        "compile_rmatmat",
    ),
    "metadata": ("shape", "dtype", "iids", "backend", "T"),
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


def test_private_candidate_matches_established_parameter_kinds_and_defaults() -> None:
    packed_signature = inspect.signature(_PackedJaxLinearARG.from_lineararg_arrays)
    exact_signature = inspect.signature(JaxLinearARG.from_lineararg_arrays)

    for name, exact_parameter in exact_signature.parameters.items():
        packed_parameter = packed_signature.parameters[name]
        assert packed_parameter.kind is exact_parameter.kind
        assert packed_parameter.default == exact_parameter.default

    for name in ("matvec", "rmatvec", "transpose_view", "__matmul__"):
        packed_method = inspect.signature(getattr(_PackedJaxLinearARG, name))
        exact_method = inspect.signature(getattr(JaxLinearARG, name))
        assert tuple(packed_method.parameters) == tuple(exact_method.parameters)
        assert tuple(parameter.kind for parameter in packed_method.parameters.values()) == tuple(
            parameter.kind for parameter in exact_method.parameters.values()
        )


def test_private_candidate_matches_exact_vector_transpose_and_compressed_extent_behavior(
    oracle_case,
) -> None:
    arrays = ingress_module._lineararg_block_arrays(oracle_case.linarg)
    nonunique_indices = np.asarray(arrays.nonunique_indices)
    inferred_extent = int(nonunique_indices.max()) + 1 if nonunique_indices.size else 0
    explicit_extent = inferred_extent + 1
    kwargs: dict[str, Any] = {
        "indptr": arrays.indptr,
        "indices": arrays.indices,
        "data": arrays.data,
        "variant_indices": arrays.variant_indices,
        "flip": arrays.flip,
        "sample_indices": arrays.sample_indices,
        "nonunique_indices": nonunique_indices,
        "n_variants": arrays.n_variants,
        "n_samples": arrays.n_samples,
        "n_nonunique_indices": explicit_extent,
        "allele_counts": arrays.allele_counts,
        "backend": Backend.PURE_JAX,
        "dtype": np.float32,
    }
    exact = JaxLinearARG.from_lineararg_arrays(**kwargs)
    packed = _PackedJaxLinearARG.from_lineararg_arrays(
        **kwargs,
        max_padding_ratio=None,
    )
    variant_values = np.asarray(oracle_case.w, dtype=np.float32)
    sample_values = np.asarray(oracle_case.y, dtype=np.float32)
    if variant_values.ndim == 2:
        variant_values = variant_values[:, 0]
    if sample_values.ndim == 2:
        sample_values = sample_values[:, 0]

    compressed_length_column = BLOCK_DESCRIPTOR_FIELDS.index("compressed_length")
    assert int(np.asarray(packed.block_descriptors)[0, 0, compressed_length_column]) == explicit_extent
    np.testing.assert_allclose(
        np.asarray(packed.matvec(variant_values)),
        np.asarray(exact.matvec(variant_values)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(packed @ variant_values),
        np.asarray(exact @ variant_values),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(packed.rmatvec(sample_values)),
        np.asarray(exact.rmatvec(sample_values)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(packed.T @ sample_values),
        np.asarray(exact.T @ sample_values),
        rtol=1e-5,
        atol=1e-5,
    )
    assert packed.T.shape == exact.T.shape
    assert packed.transpose_view().T is packed

    invalid_extent = inferred_extent - 1
    invalid_kwargs = kwargs.copy()
    invalid_kwargs["n_nonunique_indices"] = invalid_extent
    with pytest.raises(ValueError, match="cannot be smaller than the maximum nonunique index"):
        JaxLinearARG.from_lineararg_arrays(**invalid_kwargs)
    with pytest.raises(ValueError, match="cannot be smaller than the maximum nonunique index"):
        _PackedJaxLinearARG.from_lineararg_arrays(
            **invalid_kwargs,
            max_padding_ratio=None,
        )


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

    monkeypatch.setattr(ingress_module, "_packed_from_linearargs", fail_packed)
    monkeypatch.setattr(JaxParallelOperator, "from_linearargs", fail_exact)

    with pytest.raises(ValueError, match="whole-block packing exceeds") as exc_info:
        _PackedJaxLinearARG.from_linearargs((linarg,), mesh=_graph_mesh())

    assert exc_info.value is expected


@pytest.mark.skipif(len(jax.devices("cpu")) < 2, reason="requires two forced CPU devices")
def test_private_constructor_padding_failure_reports_ratio_and_configured_limit(oracle_case) -> None:
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("graph",))

    with pytest.raises(ValueError, match="whole-block packing exceeds") as exc_info:
        _PackedJaxLinearARG.from_linearargs(
            (oracle_case.linarg,),
            mesh=mesh,
            max_padding_ratio=1.25,
        )

    message = str(exc_info.value)
    assert "padding ratio=2.000000" in message
    assert "configured max_padding_ratio=1.250000" in message


def test_promotion_checklist_names_remaining_public_migration_work() -> None:
    assert set(_PROMOTION_CHECKLIST) == {"constructors", "methods", "metadata", "remaining_changes"}
    assert "JaxLinearARG" in " ".join(_PROMOTION_CHECKLIST["remaining_changes"])
    assert "exports" in " ".join(_PROMOTION_CHECKLIST["remaining_changes"])
    assert "--jax-backend" in " ".join(_PROMOTION_CHECKLIST["remaining_changes"])
