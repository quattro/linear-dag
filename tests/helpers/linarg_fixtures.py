from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from linear_dag.core.lineararg import LinearARG, list_blocks
from linear_dag.core.parallel_processing import ParallelOperator


def load_block_metadata(linarg_h5_path: Path) -> pl.DataFrame:
    """Load block metadata from a linear ARG HDF5 file."""
    return list_blocks(linarg_h5_path)


def get_first_block_name(linarg_h5_path: Path) -> str:
    """Return the first block name from block metadata."""
    metadata = load_block_metadata(linarg_h5_path)
    return str(metadata.get_column("block_name")[0])


def load_lineararg_block(
    linarg_h5_path: Path,
    *,
    block_name: str,
    load_metadata: bool = False,
) -> LinearARG:
    """Load one block-scoped LinearARG from HDF5."""
    return LinearARG.read(linarg_h5_path, block=block_name, load_metadata=load_metadata)


@contextmanager
def open_parallel_operator(
    linarg_h5_path: Path,
    *,
    num_processes: int = 2,
    **kwargs,
) -> Iterator[ParallelOperator]:
    """Open a ParallelOperator with explicit context-manager ownership."""
    with ParallelOperator.from_hdf5(linarg_h5_path, num_processes=num_processes, **kwargs) as operator:
        yield operator
