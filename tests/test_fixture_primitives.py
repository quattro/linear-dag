from __future__ import annotations

from pathlib import Path

import polars as pl

from tests.helpers.linarg_fixtures import get_first_block_name, load_lineararg_block


def test_fixture_paths_are_resolved(test_data_dir: Path, linarg_h5_path: Path, phenotypes_tsv_path: Path):
    assert test_data_dir.exists()
    assert linarg_h5_path.exists()
    assert linarg_h5_path.name == "test_chr21_50.h5"
    assert phenotypes_tsv_path.exists()
    assert phenotypes_tsv_path.name == "phenotypes_50.tsv"


def test_block_metadata_and_first_block_are_consistent(linarg_h5_path: Path, linarg_block_metadata: pl.DataFrame):
    assert isinstance(linarg_block_metadata, pl.DataFrame)
    assert not linarg_block_metadata.is_empty()
    block_names = linarg_block_metadata.get_column("block_name").to_list()
    first_block = get_first_block_name(linarg_h5_path)
    assert first_block in block_names


def test_helper_loads_lineararg_from_first_block(linarg_h5_path: Path, first_block_name: str):
    linarg = load_lineararg_block(linarg_h5_path, block_name=first_block_name)
    assert linarg.shape[0] > 0
    assert linarg.shape[1] > 0
