# pattern: Imperative Shell

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.helpers.linarg_fixtures import get_first_block_name, load_block_metadata


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runbench",
        action="store_true",
        default=False,
        help="run opt-in benchmark tests",
    )
    parser.addoption(
        "--linarg-h5-path",
        type=Path,
        default=None,
        help="override the LinearARG HDF5 fixture path used by tests and benchmarks",
    )
    parser.addoption(
        "--linarg-block-limit",
        type=int,
        default=None,
        help="limit LinearARG HDF5 block metadata to the first N blocks for tests and benchmarks",
    )
    parser.addoption(
        "--linarg-benchmark-k",
        nargs="+",
        type=int,
        default=(1, 8, 64),
        help="matrix widths to use for LinearARG benchmark inputs",
    )
    parser.addoption(
        "--linarg-parallel-processes",
        type=int,
        default=2,
        help="number of worker processes to use for ParallelOperator benchmarks",
    )
    parser.addoption(
        "--rhe-benchmark-num-matvecs",
        nargs="+",
        type=int,
        default=(4, 20),
        help="probe-vector counts to use for RHE benchmarks",
    )
    parser.addoption(
        "--jax-promotion-output",
        type=Path,
        default=None,
        help="write phase-level benchmark evidence rows to this JSON path",
    )
    parser.addoption(
        "--jax-enforce-promotion-gates",
        action="store_true",
        default=False,
        help="fail run if promotion gates evaluate as non-promotable",
    )
    parser.addoption(
        "--jax-validation-evidence-id",
        type=str,
        default=None,
        help="runner-owned checksum identifier proving validation suites completed",
    )
    parser.addoption(
        "--cache-policy",
        choices=["fresh", "reused"],
        default="fresh",
        help="internal cache-policy label for promotion evidence collection",
    )
    parser.addoption(
        "--platform-label",
        type=str,
        default="local",
        help="platform label recorded in promotion evidence",
    )


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).parent / "testdata"


@pytest.fixture(scope="session")
def linarg_h5_path(request: pytest.FixtureRequest, test_data_dir: Path) -> Path:
    override = request.config.getoption("--linarg-h5-path")
    if override is not None:
        return override
    return test_data_dir / "test_chr21_50.h5"


@pytest.fixture(scope="session")
def phenotypes_tsv_path(test_data_dir: Path) -> Path:
    return test_data_dir / "phenotypes_50.tsv"


@pytest.fixture(scope="session")
def linarg_block_metadata(request: pytest.FixtureRequest, linarg_h5_path: Path) -> pl.DataFrame:
    metadata = load_block_metadata(linarg_h5_path)
    block_limit = request.config.getoption("--linarg-block-limit")
    if block_limit is None:
        return metadata
    if block_limit < 1:
        raise ValueError("--linarg-block-limit must be at least 1")
    return metadata.head(block_limit)


@pytest.fixture(scope="session")
def first_block_name(linarg_h5_path: Path) -> str:
    return get_first_block_name(linarg_h5_path)


@pytest.fixture(scope="session")
def linarg_benchmark_k_values(request: pytest.FixtureRequest) -> tuple[int, ...]:
    k_values = tuple(request.config.getoption("--linarg-benchmark-k"))
    if not k_values:
        raise ValueError("--linarg-benchmark-k must contain at least one value")
    if any(k < 1 for k in k_values):
        raise ValueError("--linarg-benchmark-k values must be positive")
    return k_values


@pytest.fixture(scope="session")
def linarg_parallel_processes(request: pytest.FixtureRequest) -> int:
    num_processes = request.config.getoption("--linarg-parallel-processes")
    if num_processes < 1:
        raise ValueError("--linarg-parallel-processes must be at least 1")
    return num_processes


@pytest.fixture(scope="session")
def rhe_benchmark_num_matvecs(request: pytest.FixtureRequest) -> tuple[int, ...]:
    values = tuple(request.config.getoption("--rhe-benchmark-num-matvecs"))
    if not values:
        raise ValueError("--rhe-benchmark-num-matvecs must contain at least one value")
    if any(value < 1 for value in values):
        raise ValueError("--rhe-benchmark-num-matvecs values must be positive")
    return values
