from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.helpers.linarg_fixtures import get_first_block_name, load_block_metadata


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).parent / "testdata"


@pytest.fixture(scope="session")
def linarg_h5_path(test_data_dir: Path) -> Path:
    return test_data_dir / "test_chr21_50.h5"


@pytest.fixture(scope="session")
def phenotypes_tsv_path(test_data_dir: Path) -> Path:
    return test_data_dir / "phenotypes_50.tsv"


@pytest.fixture(scope="session")
def linarg_block_metadata(linarg_h5_path: Path) -> pl.DataFrame:
    return load_block_metadata(linarg_h5_path)


@pytest.fixture(scope="session")
def first_block_name(linarg_h5_path: Path) -> str:
    return get_first_block_name(linarg_h5_path)
