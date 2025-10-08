from pathlib import Path

import numpy as np
import polars as pl

from linear_dag import cli
from linear_dag.core.lineararg import list_blocks
from linear_dag.core.parallel_processing import ParallelOperator

TEST_DATA_DIR = Path(__file__).parent / "testdata"


def test_cli_assoc_smoke(tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "assoc_out"

    # Run CLI 'assoc'
    args = [
        "-v",
        "assoc",
        str(linarg_path),
        str(pheno_path),
        "--pheno-name",
        "iid, height,bmi",
        "--covar-name",
        "iid, sex,age",
        "--out",
        str(out_prefix),
        "--num-processes",
        "2",
    ]
    rc = cli._main(args)
    assert rc == 0 or rc is None

    # Output parquet is written as "<out_prefix>.parquet"
    out_parquet = Path(f"{out_prefix}.parquet")
    assert out_parquet.exists()

    df = pl.read_parquet(out_parquet)
    # Expect BETA/SE columns for each phenotype
    expected_cols = {"height_BETA", "height_SE", "bmi_BETA", "bmi_SE"}
    assert expected_cols.issubset(set(df.columns))


def test_cli_score_smoke(tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    out_prefix = tmp_path / "score_out"

    # Determine number of variants (rows needed in beta parquet)
    with ParallelOperator.from_hdf5(str(linarg_path), num_processes=1) as op:
        m = op.shape[1]

    beta_parquet = tmp_path / "betas.parquet"
    # Create a simple single-score column of ones (float32) with m rows
    tbl = pl.DataFrame({"score": np.ones(m, dtype=np.float32)})
    tbl.write_parquet(beta_parquet)

    # Run CLI 'score'
    args = [
        "-v",
        "score",
        "--linarg-path",
        str(linarg_path),
        "--beta-path",
        str(beta_parquet),
        "--score-cols",
        "score",
        "--out",
        str(out_prefix),
        "--num-processes",
        "2",
    ]
    rc = cli._main(args)
    assert rc == 0 or rc is None

    # Output TSV is written as "<out_prefix>.tsv"
    out_tsv = Path(f"{out_prefix}.tsv")
    assert out_tsv.exists()

    df = pl.read_csv(out_tsv, separator="\t")
    # Expect at least iid and score columns
    assert {"iid", "score"}.issubset(set(df.columns))
    # Row count equals number of unique IIDs in the HDF5
    with ParallelOperator.from_hdf5(str(linarg_path), num_processes=1) as op:
        n_unique_iids = op.iids.unique(maintain_order=True).len()
    assert df.shape[0] == n_unique_iids
