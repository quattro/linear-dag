# pattern: Mixed (unavoidable)
# Reason: These tests construct real HDF5-backed operators, then compare the
# JAX RHE numerical path against the existing Cython-backed RHE implementation.

from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from jax.extend import core as jax_core
from jax.sharding import Mesh
from jax.tree_util import tree_leaves

from linear_dag.association._heritability_jax import (
    _build_residualized_operator,
    _prepare_jax_rhe_inputs,
    _ResidualizedJaxGRM,
    _should_use_blockwise_grm,
    randomized_haseman_elston as randomized_haseman_elston_jax,
)
from linear_dag.association.heritability import randomized_haseman_elston
from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxParallelOperator
from linear_dag.core.jaxlinarg._hijax import _PackedGraphType
from linear_dag.core.jaxlinarg.ingress import _packed_from_hdf5
from linear_dag.core.lineararg import list_blocks, list_iids
from linear_dag.core.parallel_processing import GRMOperator

PHENO_COLS = ["height", "bmi"]
COVAR_COLS = ["sex"]


def _jax_grm_from_hdf5(path: Path) -> JaxGRMOperator:
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))
    block_metadata = list_blocks(path)
    assert block_metadata is not None
    operator = JaxParallelOperator.from_hdf5(
        str(path),
        mesh=mesh,
        block_metadata=block_metadata,
        backend=Backend.PURE_JAX,
    )
    return JaxGRMOperator(operator, alpha=-1.0, iids=list_iids(path))


def _packed_jax_grm_from_hdf5(path: Path) -> JaxGRMOperator:
    devices = jax.devices("cpu")
    num_devices = min(2, len(devices))
    mesh = Mesh(np.asarray(devices[:num_devices]), ("graph",))
    block_metadata = list_blocks(path)
    assert block_metadata is not None
    block_names = tuple(block_metadata.get_column("block_name").to_list())
    operator = _packed_from_hdf5(
        path,
        block_names,
        mesh=mesh,
        allow_excess_padding=True,
    ).operator
    return JaxGRMOperator(operator, alpha=-1.0, iids=list_iids(path))


def _phenotypes() -> pl.DataFrame:
    frame = pl.read_csv(Path("tests/testdata/phenotypes_50.tsv"), separator="\t")
    return frame.select(["iid", *PHENO_COLS, *COVAR_COLS]).with_columns(pl.lit(1.0).alias("intercept"))


class _RecordingGRM:
    dtype = jnp.float32

    def __init__(self, iids: list[str]):
        self.iids = pl.Series("iids", iids).cast(pl.Utf8)
        self.shape = (len(iids), len(iids))
        self.default_calls = 0
        self.blockwise_calls = 0

    def matmat(self, values):
        self.default_calls += 1
        return values

    def matmat_blockwise(self, values):
        self.blockwise_calls += 1
        return values


def _recursive_array_constant_bytes(closed_jaxpr: jax_core.ClosedJaxpr) -> int:
    total = 0

    def add_constant(constant: Any) -> None:
        nonlocal total
        if isinstance(constant, (jax.Array, np.ndarray)):
            total += int(constant.size * constant.dtype.itemsize)
            return
        lower_val = getattr(jax.typeof(constant), "lower_val", None)
        if lower_val is not None:
            for lowered in lower_val(constant):
                add_constant(lowered)

    def visit(value: Any) -> None:
        if isinstance(value, jax_core.Jaxpr):
            for constant in getattr(value, "consts", ()):
                add_constant(constant)
            for equation in value.eqns:
                visit(equation.params)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                visit(nested)

    visit(closed_jaxpr)
    return total


def test_jax_randomized_haseman_elston_matches_cython_hutchinson(linarg_h5_path: Path):
    data = _phenotypes()
    covar_cols = ["intercept", *COVAR_COLS]

    with GRMOperator.from_hdf5(str(linarg_h5_path), num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est="hutchinson",
            sampler="normal",
            seed=20260522,
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est="hutchinson",
        sampler="normal",
        seed=20260522,
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize(
    ("num_matvecs", "reordered"),
    ((4, False), (20, True)),
    ids=("k4", "k20-reordered"),
)
def test_packed_jax_rhe_matches_cython_fixed_probes(
    linarg_h5_path: Path,
    num_matvecs: int,
    reordered: bool,
) -> None:
    data = _phenotypes()
    if reordered:
        data = data.sort("iid", descending=True)
    covar_cols = ["intercept", *COVAR_COLS]
    seed = 20260814 + num_matvecs

    with GRMOperator.from_hdf5(str(linarg_h5_path), num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=num_matvecs,
            trace_est="hutchinson",
            sampler="rademacher",
            seed=seed,
        )

    observed = randomized_haseman_elston_jax(
        _packed_jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=num_matvecs,
        trace_est="hutchinson",
        sampler="rademacher",
        seed=seed,
    )

    assert observed.columns == expected.columns
    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_randomized_haseman_elston_reorders_phenotype_rows(linarg_h5_path: Path):
    data = _phenotypes().sort("iid", descending=True)
    covar_cols = ["intercept", *COVAR_COLS]

    with GRMOperator.from_hdf5(str(linarg_h5_path), num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est="hutchinson",
            sampler="rademacher",
            seed=20260523,
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est="hutchinson",
        sampler="rademacher",
        seed=20260523,
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_packed_residualized_operator_is_explicit_with_zero_graph_constants(linarg_h5_path: Path) -> None:
    grm = _packed_jax_grm_from_hdf5(linarg_h5_path)
    data = _phenotypes()
    prepared = _prepare_jax_rhe_inputs(
        grm,
        data.lazy(),
        PHENO_COLS,
        ["intercept", *COVAR_COLS],
    )
    operator = _build_residualized_operator(grm, prepared.alignment, prepared.covariates)
    values = jnp.ones((operator.shape[0], 2), dtype=prepared.yresid.dtype)

    assert isinstance(operator, _ResidualizedJaxGRM)
    assert isinstance(operator, eqx.Module)
    assert isinstance(jax.typeof(operator.grm.operator.graph), _PackedGraphType)
    assert operator.residual_rank == operator.shape[0] - operator.basis.shape[1]
    dynamic_leaves = tree_leaves(operator)
    assert any(leaf is operator.left_indices for leaf in dynamic_leaves)
    assert any(leaf is operator.right_indices for leaf in dynamic_leaves)
    assert any(leaf is operator.basis for leaf in dynamic_leaves)

    closed_jaxpr = jax.make_jaxpr(lambda explicit_operator, dense: explicit_operator.matmat(dense))(
        operator,
        values,
    )
    assert _recursive_array_constant_bytes(closed_jaxpr) == 0
    result = operator.matmat(values)
    assert result.shape == values.shape
    assert bool(jnp.all(jnp.isfinite(result)))


def test_blockwise_grm_selection_is_exact_ragged_only(linarg_h5_path: Path, monkeypatch) -> None:
    exact = _jax_grm_from_hdf5(linarg_h5_path)
    packed = _packed_jax_grm_from_hdf5(linarg_h5_path)

    assert _should_use_blockwise_grm(exact)
    assert not _should_use_blockwise_grm(packed)

    monkeypatch.setenv("LINEAR_DAG_JAX_RHE_BLOCKWISE_GRM", "off")
    assert not _should_use_blockwise_grm(exact)


def test_jax_randomized_haseman_elston_rejects_missing_iids(linarg_h5_path: Path):
    grm = JaxGRMOperator(
        JaxParallelOperator.from_hdf5(
            str(linarg_h5_path),
            mesh=Mesh(np.asarray(jax.devices()[:1]), ("blocks",)),
            block_metadata=list_blocks(linarg_h5_path),
            backend=Backend.PURE_JAX,
        ),
        alpha=-1.0,
    )

    try:
        randomized_haseman_elston_jax(
            grm,
            _phenotypes().lazy(),
            PHENO_COLS,
            ["intercept", *COVAR_COLS],
            num_matvecs=4,
        )
    except ValueError as error:
        assert "iids" in str(error)
    else:
        raise AssertionError("missing JAX GRM iids should fail")


def test_jax_randomized_haseman_elston_uses_jax_arrays_in_matmat():
    class RecordingGRM:
        shape = (4, 4)
        dtype = jnp.float32
        iids = pl.Series("iids", ["a", "a", "b", "b"]).cast(pl.Utf8)

        def __init__(self):
            self.observed_types = []

        def matmat(self, values):
            self.observed_types.append(type(values))
            return values

    grm = RecordingGRM()
    data = pl.DataFrame(
        {
            "iid": ["a", "b"],
            "trait": [0.25, -0.5],
            "intercept": [1.0, 1.0],
        }
    )

    randomized_haseman_elston_jax(
        cast(JaxGRMOperator, grm),
        data.lazy(),
        ["trait"],
        ["intercept"],
        num_matvecs=2,
        trace_est="hutchinson",
        sampler="normal",
        seed=1,
    )

    assert grm.observed_types
    assert all(not issubclass(observed, np.ndarray) for observed in grm.observed_types)


def test_jax_randomized_haseman_elston_does_not_select_blockwise_by_duck_typing():
    grm = _RecordingGRM(["a", "a", "b", "b"])
    data = pl.DataFrame(
        {
            "iid": ["a", "b"],
            "trait": [0.25, -0.5],
            "intercept": [1.0, 1.0],
        }
    )

    randomized_haseman_elston_jax(
        cast(JaxGRMOperator, grm),
        data.lazy(),
        ["trait"],
        ["intercept"],
        num_matvecs=2,
        trace_est="hutchinson",
        sampler="normal",
        seed=1,
    )

    assert grm.default_calls > 0
    assert grm.blockwise_calls == 0


def test_jax_randomized_haseman_elston_allows_default_grm_matmat_opt_out(monkeypatch):
    class RecordingGRM:
        shape = (4, 4)
        dtype = jnp.float32
        iids = pl.Series("iids", ["a", "a", "b", "b"]).cast(pl.Utf8)

        def __init__(self):
            self.default_calls = 0
            self.blockwise_calls = 0

        def matmat(self, values):
            self.default_calls += 1
            return values

        def matmat_blockwise(self, values):
            self.blockwise_calls += 1
            return values

    monkeypatch.setenv("LINEAR_DAG_JAX_RHE_BLOCKWISE_GRM", "0")
    grm = RecordingGRM()
    data = pl.DataFrame(
        {
            "iid": ["a", "b"],
            "trait": [0.25, -0.5],
            "intercept": [1.0, 1.0],
        }
    )

    randomized_haseman_elston_jax(
        cast(JaxGRMOperator, grm),
        data.lazy(),
        ["trait"],
        ["intercept"],
        num_matvecs=2,
        trace_est="hutchinson",
        sampler="normal",
        seed=1,
    )

    assert grm.default_calls > 0
    assert grm.blockwise_calls == 0


@pytest.mark.parametrize(
    ("data", "grm_iids", "pheno_cols", "covar_cols", "message"),
    (
        (
            pl.DataFrame({"trait": [0.25, -0.5], "intercept": [1.0, 1.0]}),
            ["a", "a", "b", "b"],
            ["trait"],
            ["intercept"],
            "missing required column.*iid",
        ),
        (
            pl.DataFrame({"iid": ["a", "b"], "intercept": [1.0, 1.0]}),
            ["a", "a", "b", "b"],
            ["trait"],
            ["intercept"],
            "missing required column.*trait",
        ),
        (
            pl.DataFrame({"iid": ["a", "b"], "trait": [0.25, -0.5]}),
            ["a", "a", "b", "b"],
            ["trait"],
            ["intercept"],
            "missing required column.*intercept",
        ),
        (
            pl.DataFrame({"iid": ["x", "y"], "trait": [0.25, -0.5], "intercept": [1.0, 0.0]}),
            ["a", "a", "b", "b"],
            ["trait"],
            ["intercept"],
            "no overlapping IIDs",
        ),
        (
            pl.DataFrame({"iid": ["a", "b"], "trait": [0.25, -0.5], "intercept": [1.0, 1.0]}),
            ["a", "b"],
            ["trait"],
            ["intercept"],
            "zero or two",
        ),
        (
            pl.DataFrame({"iid": ["a", "a"], "trait": [0.25, -0.5], "intercept": [1.0, 1.0]}),
            ["a", "a"],
            ["trait"],
            ["intercept"],
            "at most one",
        ),
        (
            pl.DataFrame({"iid": ["a", "b"], "trait": [0.25, -0.5], "intercept": [1.0, 0.0]}),
            ["a", "a", "b", "b"],
            ["trait"],
            ["intercept"],
            "First column",
        ),
        (
            pl.DataFrame({"iid": ["a", "b"], "trait": [np.nan, np.nan], "intercept": [1.0, 1.0]}),
            ["a", "a", "b", "b"],
            ["trait"],
            ["intercept"],
            "at least one non-missing",
        ),
    ),
    ids=(
        "missing-iid-column",
        "missing-phenotype-column",
        "missing-covariate-column",
        "zero-overlap",
        "haploid-multiplicity",
        "duplicate-phenotype-iid",
        "invalid-intercept",
        "all-missing-phenotype",
    ),
)
def test_jax_rhe_rejects_invalid_alignment_before_graph_products(
    data: pl.DataFrame,
    grm_iids: list[str],
    pheno_cols: list[str],
    covar_cols: list[str],
    message: str,
) -> None:
    grm = _RecordingGRM(grm_iids)

    with pytest.raises(ValueError, match=message):
        randomized_haseman_elston_jax(
            cast(JaxGRMOperator, grm),
            data.lazy(),
            pheno_cols,
            covar_cols,
            num_matvecs=1,
            trace_est="hutchinson",
            sampler="normal",
            seed=1,
        )

    assert grm.default_calls == 0
    assert grm.blockwise_calls == 0
