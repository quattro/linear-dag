# pattern: Mixed (unavoidable)
# Reason: These tests construct real HDF5-backed operators, then compare the
# JAX RHE numerical path against the existing Cython-backed RHE implementation.

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from jax.sharding import Mesh
from scipy.sparse.linalg import LinearOperator

from linear_dag.association._heritability_jax import randomized_haseman_elston as randomized_haseman_elston_jax
from linear_dag.association.heritability import randomized_haseman_elston
from linear_dag.core.jaxlinarg import Backend, JaxGRMOperator, JaxParallelOperator
from linear_dag.core.lineararg import list_blocks, list_iids
from linear_dag.core.parallel_processing import GRMOperator

PHENO_COLS = ["height", "bmi"]
COVAR_COLS = ["sex"]


def _jax_grm_from_hdf5(path: Path) -> JaxGRMOperator:
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("blocks",))
    operator = JaxParallelOperator.from_hdf5(
        str(path),
        mesh=mesh,
        block_metadata=list_blocks(path),
        backend=Backend.PURE_JAX,
    )
    return JaxGRMOperator(operator, alpha=-1.0, iids=list_iids(path))


def _phenotypes() -> pl.DataFrame:
    frame = pl.read_csv(Path("tests/testdata/phenotypes_50.tsv"), separator="\t")
    return frame.select(["iid", *PHENO_COLS, *COVAR_COLS]).with_columns(pl.lit(1.0).alias("intercept"))


@pytest.mark.parametrize("trace_est", ["hutchinson", "hutch++", "xnystrace"])
@pytest.mark.parametrize("sampler", ["normal", "sphere", "rademacher"])
def test_jax_randomized_haseman_elston_matches_cython_estimators_and_samplers(
    linarg_h5_path: Path,
    trace_est: str,
    sampler: str,
):
    data = _phenotypes()
    covar_cols = ["intercept", *COVAR_COLS]

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est=trace_est,
            sampler=sampler,
            seed=20260522,
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est=trace_est,
        sampler=sampler,
        seed=20260522,
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_randomized_haseman_elston_matches_cython_with_near_collinear_covariates(
    linarg_h5_path: Path,
):
    data = _phenotypes().with_columns(
        (
            pl.col("sex").cast(pl.Float64) + pl.Series("perturbation", np.linspace(-1e-7, 1e-7, _phenotypes().height))
        ).alias("near_sex")
    )
    covar_cols = ["intercept", "sex", "near_sex"]

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est="hutchinson",
            sampler="normal",
            seed=20260827,
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est="hutchinson",
        sampler="normal",
        seed=20260827,
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_randomized_haseman_elston_matches_cython_with_missing_covariates(
    linarg_h5_path: Path,
):
    data = _phenotypes().with_columns(
        pl.when(pl.int_range(pl.len()) == 0).then(float("nan")).otherwise(pl.col("sex").cast(pl.Float64)).alias("sex")
    )
    covar_cols = ["intercept", *COVAR_COLS]

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est="hutchinson",
            sampler="sphere",
            seed=20260828,
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est="hutchinson",
        sampler="sphere",
        seed=20260828,
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_randomized_haseman_elston_matches_cython_with_generator_seed(linarg_h5_path: Path):
    data = _phenotypes()
    covar_cols = ["intercept", *COVAR_COLS]

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est="hutchinson",
            sampler="normal",
            seed=np.random.default_rng(20260829),
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est="hutchinson",
        sampler="normal",
        seed=np.random.default_rng(20260829),
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_randomized_haseman_elston_matches_cython_singular_failure():
    class ZeroCanonicalGRM(LinearOperator):
        iids = pl.Series("iids", ["a", "a", "b", "b"]).cast(pl.Utf8)

        def __init__(self):
            super().__init__(dtype=np.dtype(np.float64), shape=(4, 4))

        def _matvec(self, values):
            return np.zeros_like(values)

        def _matmat(self, values):
            return np.zeros_like(values)

    class ZeroJaxGRM:
        shape = (4, 4)
        dtype = jnp.float32
        iids = pl.Series("iids", ["a", "a", "b", "b"]).cast(pl.Utf8)

        @staticmethod
        def matmat(values):
            return jnp.zeros_like(values)

    data = pl.DataFrame(
        {
            "iid": ["a", "b"],
            "trait": [0.25, -0.5],
            "intercept": [1.0, 1.0],
        }
    ).lazy()

    with pytest.raises(np.linalg.LinAlgError):
        randomized_haseman_elston(
            ZeroCanonicalGRM(),
            data,
            ["trait"],
            ["intercept"],
            num_matvecs=2,
            seed=1,
        )

    with pytest.raises(np.linalg.LinAlgError):
        randomized_haseman_elston_jax(
            ZeroJaxGRM(),
            data,
            ["trait"],
            ["intercept"],
            num_matvecs=2,
            seed=1,
        )


def test_jax_randomized_haseman_elston_matches_cython_with_rank_deficient_covariates(
    linarg_h5_path: Path,
):
    data = _phenotypes().with_columns(pl.lit(0.0).alias("zero"))
    covar_cols = ["intercept", "zero", *COVAR_COLS]

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=-1.0) as cython_grm:
        expected = randomized_haseman_elston(
            cython_grm,
            data.lazy(),
            PHENO_COLS,
            covar_cols,
            num_matvecs=4,
            trace_est="hutchinson",
            sampler="normal",
            seed=20260826,
        )

    observed = randomized_haseman_elston_jax(
        _jax_grm_from_hdf5(linarg_h5_path),
        data.lazy(),
        PHENO_COLS,
        covar_cols,
        num_matvecs=4,
        trace_est="hutchinson",
        sampler="normal",
        seed=20260826,
    )

    np.testing.assert_allclose(
        observed.select(["s2g", "s2e", "h2g"]).to_numpy(),
        expected.select(["s2g", "s2e", "h2g"]).to_numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_randomized_haseman_elston_reorders_phenotype_rows(linarg_h5_path: Path):
    data = _phenotypes().sort("iid", descending=True)
    covar_cols = ["intercept", *COVAR_COLS]

    with GRMOperator.from_hdf5(linarg_h5_path, num_processes=1, alpha=-1.0) as cython_grm:
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
        grm,
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


def test_jax_randomized_haseman_elston_prefers_blockwise_grm_matmat():
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

    grm = RecordingGRM()
    data = pl.DataFrame(
        {
            "iid": ["a", "b"],
            "trait": [0.25, -0.5],
            "intercept": [1.0, 1.0],
        }
    )

    randomized_haseman_elston_jax(
        grm,
        data.lazy(),
        ["trait"],
        ["intercept"],
        num_matvecs=2,
        trace_est="hutchinson",
        sampler="normal",
        seed=1,
    )

    assert grm.blockwise_calls > 0
    assert grm.default_calls == 0


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
        grm,
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
