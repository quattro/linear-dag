from pathlib import Path

import numpy as np
import polars as pl
import pytest

from linear_dag.association.heritability import randomized_haseman_elston
from linear_dag.association.simulation import simulate_phenotype
from linear_dag.core.operators import get_diploid_operator
from linear_dag.core.parallel_processing import GRMOperator, ParallelOperator

TEST_DATA_DIR = Path(__file__).parent / "testdata"
BASE_SEED = 0
NUM_DATA_REPEATS = 2
NUM_ESTIMATOR_REPEATS = 4
PHENO_COLS = ["phenotype1", "phenotype2"]
COVAR_COLS = ["intercept", "covar1"]


def _seed_from_sequence(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def _impute_missing_with_mean(covariates: np.ndarray) -> np.ndarray:
    covariates = covariates.copy()
    for col_idx in range(covariates.shape[1]):
        is_missing = np.isnan(covariates[:, col_idx])
        if not np.any(is_missing):
            continue
        col_mean = np.mean(covariates[~is_missing, col_idx])
        covariates[is_missing, col_idx] = col_mean
    return covariates


def _exact_haseman_elston(grm: np.ndarray, phenotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    phenotypes = phenotypes.copy()
    covariates = _impute_missing_with_mean(covariates)
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    if np.any(num_nonmissing == 0):
        raise ValueError("Each phenotype must have at least one non-missing value")

    phenotypes[is_missing] = 0.0
    beta = np.linalg.lstsq(covariates, phenotypes, rcond=None)[0]
    yresid = phenotypes - covariates @ beta
    yresid[is_missing] = 0.0
    yresid /= np.sqrt(np.sum(yresid**2, axis=0) / num_nonmissing)

    C = np.sum(grm @ yresid * yresid, axis=0)
    E = np.sum(yresid * yresid, axis=0)
    N = grm.shape[0]
    grm_trace = np.trace(grm)
    grm_sq_trace = np.trace(grm @ grm)

    lhs = np.array([[grm_sq_trace, grm_trace], [grm_trace, N]], dtype=np.float64)
    rhs = np.vstack([C, E])
    solution = np.linalg.solve(lhs, rhs)

    s2g = solution[0, :] * grm_trace
    s2e = solution[1, :] * N
    return s2g / (s2g + s2e)


def _build_rhe_fixture(
    hdf5_path: Path,
    heritability: float,
    simulation_seed: int,
    covariate_seed: int,
) -> tuple[pl.DataFrame, np.ndarray]:
    simulation_rng = np.random.default_rng(simulation_seed)
    covariate_rng = np.random.default_rng(covariate_seed)

    with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as genotypes:
        n_haplotypes, _ = genotypes.shape
        dgenotypes = get_diploid_operator(genotypes.normalized) / np.sqrt(2.0)
        phenotypes, _ = simulate_phenotype(
            dgenotypes,
            heritability=heritability,
            fraction_causal=1.0,
            return_beta=True,
            num_traits=2,
            seed=simulation_rng,
        )

        covariates = np.column_stack(
            [
                np.ones(n_haplotypes // 2, dtype=np.float64),
                covariate_rng.random(size=(n_haplotypes // 2,)),
            ]
        )
        iids = genotypes.iids.unique(maintain_order=True).to_list()

        df_pheno = pl.DataFrame(
            {
                "iid": iids,
                "phenotype1": phenotypes[:, 0].astype(np.float64),
                "phenotype2": phenotypes[:, 1].astype(np.float64),
                "intercept": covariates[:, 0],
                "covar1": covariates[:, 1],
            }
        )

        G = dgenotypes @ np.eye(dgenotypes.shape[1], dtype=np.float32)
        grm_direct = G @ G.T / G.shape[1]

    expected = _exact_haseman_elston(grm_direct, phenotypes.astype(np.float64), covariates)
    return df_pheno, expected


def _run_randomized_replicates(
    hdf5_path: Path,
    df_pheno: pl.DataFrame,
    estimator: str,
    sampler: str,
    num_matvecs: int,
    estimator_seeds: list[int],
) -> np.ndarray:
    draws: list[np.ndarray] = []
    with GRMOperator.from_hdf5(hdf5_path, num_processes=2, alpha=-1.0) as grm:
        for seed in estimator_seeds:
            observed = randomized_haseman_elston(
                grm,
                df_pheno.lazy(),
                PHENO_COLS,
                COVAR_COLS,
                num_matvecs,
                estimator,
                sampler,
                seed=seed,
            )
            draws.append(observed.get_column("h2g").to_numpy())
    return np.vstack(draws)


@pytest.mark.parametrize("estimator", ("hutchinson", "hutch++", "xnystrace"))
def test_rhe_mean_tracks_exact_he_baseline(estimator: str):
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    error_values: list[float] = []

    root_ss = np.random.SeedSequence(BASE_SEED)
    for data_ss in root_ss.spawn(NUM_DATA_REPEATS):
        simulation_ss, covariate_ss, estimator_ss = data_ss.spawn(3)
        estimator_seeds = [_seed_from_sequence(ss) for ss in estimator_ss.spawn(NUM_ESTIMATOR_REPEATS)]
        df_pheno, expected = _build_rhe_fixture(
            hdf5_path,
            heritability=0.5,
            simulation_seed=_seed_from_sequence(simulation_ss),
            covariate_seed=_seed_from_sequence(covariate_ss),
        )
        observed_draws = _run_randomized_replicates(
            hdf5_path,
            df_pheno,
            estimator=estimator,
            sampler="normal",
            num_matvecs=10,
            estimator_seeds=estimator_seeds,
        )
        error_values.extend(np.abs(observed_draws.mean(axis=0) - expected).tolist())

    errors = np.asarray(error_values, dtype=np.float64)
    assert float(np.mean(errors)) < 0.30
    assert float(np.max(errors)) < 0.60


@pytest.mark.parametrize("estimator", ("hutchinson", "hutch++", "xnystrace"))
@pytest.mark.parametrize("sampler", ("normal", "sphere", "rademacher"))
def test_hutchinson_error_reduces_with_more_matvecs(estimator: str, sampler: str):
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    small_k_errors: list[float] = []
    large_k_errors: list[float] = []

    root_ss = np.random.SeedSequence(BASE_SEED)
    for data_ss in root_ss.spawn(NUM_DATA_REPEATS):
        simulation_ss, covariate_ss, estimator_ss = data_ss.spawn(3)
        estimator_seeds = [_seed_from_sequence(ss) for ss in estimator_ss.spawn(NUM_ESTIMATOR_REPEATS)]
        df_pheno, expected = _build_rhe_fixture(
            hdf5_path,
            heritability=0.5,
            simulation_seed=_seed_from_sequence(simulation_ss),
            covariate_seed=_seed_from_sequence(covariate_ss),
        )

        observed_small_k = _run_randomized_replicates(
            hdf5_path,
            df_pheno,
            estimator=estimator,
            sampler=sampler,
            num_matvecs=5,
            estimator_seeds=estimator_seeds,
        )
        observed_large_k = _run_randomized_replicates(
            hdf5_path,
            df_pheno,
            estimator=estimator,
            sampler=sampler,
            num_matvecs=25,
            estimator_seeds=estimator_seeds,
        )

        small_k_errors.extend(np.mean(np.abs(observed_small_k - expected[None, :]), axis=1).tolist())
        large_k_errors.extend(np.mean(np.abs(observed_large_k - expected[None, :]), axis=1).tolist())

    assert float(np.mean(large_k_errors)) <= float(np.mean(small_k_errors)) + 0.05


def test_rhe_seed_reproducible_across_operator_instances():
    hdf5_path = TEST_DATA_DIR / "test_chr21_50.h5"
    df_pheno, _ = _build_rhe_fixture(
        hdf5_path,
        heritability=0.5,
        simulation_seed=BASE_SEED,
        covariate_seed=BASE_SEED + 1,
    )
    seed = BASE_SEED + 2

    with GRMOperator.from_hdf5(hdf5_path, num_processes=2, alpha=-1.0) as grm:
        observed_first = (
            randomized_haseman_elston(
                grm,
                df_pheno.lazy(),
                PHENO_COLS,
                COVAR_COLS,
                num_matvecs=10,
                trace_est="hutchinson",
                sampler="normal",
                seed=seed,
            )
            .get_column("h2g")
            .to_numpy()
        )

    with GRMOperator.from_hdf5(hdf5_path, num_processes=2, alpha=-1.0) as grm:
        observed_second = (
            randomized_haseman_elston(
                grm,
                df_pheno.lazy(),
                PHENO_COLS,
                COVAR_COLS,
                num_matvecs=10,
                trace_est="hutchinson",
                sampler="normal",
                seed=seed,
            )
            .get_column("h2g")
            .to_numpy()
        )

    np.testing.assert_allclose(observed_first, observed_second, rtol=0.0, atol=1e-8)
