from pathlib import Path

import numpy as np
import polars as pl

from linear_dag import cli
from linear_dag.core.lineararg import list_blocks, load_variant_info
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
    # Create beta parquet with:
    # - a subset of linarg IDs (to test intersection)
    # - additional IDs not present in the linear ARG (to test they are ignored)
    linarg_ids = load_variant_info(str(linarg_path), columns="id_only").collect().get_column("ID").cast(pl.String)
    assert linarg_ids.len() == m
    subset_idx = np.arange(0, m, 2)  # take every other variant
    subset_ids = pl.Series("ID", linarg_ids.to_numpy()[subset_idx]).cast(pl.String)
    n_extra = min(10, max(1, m // 10))
    extra_ids = pl.Series("ID", [f"EXTRA_{i}" for i in range(n_extra)]).cast(pl.String)
    combined_ids = pl.concat([subset_ids, extra_ids])
    # Shuffle order
    rng = np.random.default_rng(42)
    perm = rng.permutation(combined_ids.len())
    combined_ids = pl.Series("ID", combined_ids.to_numpy()[perm]).cast(pl.String)
    # Scores aligned to combined_ids
    scores = np.ones(combined_ids.len(), dtype=np.float32)
    tbl = pl.DataFrame({"ID": combined_ids, "score": scores})
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

    # Build a second betas parquet that covers ALL ARG IDs but uses zeros for missing ones
    beta_parquet_full = tmp_path / "betas_full.parquet"
    score_full = np.zeros(m, dtype=np.float32)
    score_full[subset_idx] = 1.0  # ones on the intersecting subset, zeros elsewhere
    tbl_full = pl.DataFrame({"ID": linarg_ids, "score": score_full})
    tbl_full.write_parquet(beta_parquet_full)

    # Run CLI 'score' again with full IDs (zeros for non-overlap)
    out_prefix2 = tmp_path / "score_out_full"
    args2 = [
        "-v",
        "score",
        "--linarg-path",
        str(linarg_path),
        "--beta-path",
        str(beta_parquet_full),
        "--score-cols",
        "score",
        "--out",
        str(out_prefix2),
        "--num-processes",
        "2",
    ]
    rc2 = cli._main(args2)
    assert rc2 == 0 or rc2 is None

    out_tsv2 = Path(f"{out_prefix2}.tsv")
    assert out_tsv2.exists()
    df2 = pl.read_csv(out_tsv2, separator="\t")

    # The two outputs should be identical (extras ignored vs zeros for non-overlap)
    # Ensure same ordering by joining on iid
    merged = df.join(df2, on="iid", how="inner", suffix="_2")
    assert merged.shape[0] == df.shape[0] == df2.shape[0]
    # Compare score columns
    assert np.allclose(merged["score"].to_numpy(), merged["score_2"].to_numpy())


def test_cli_assoc_repeat_covar_smoke(tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "assoc_repeat_covar_out"

    # Run CLI 'assoc' with --repeat-covar (loop mode)
    args = [
        "-v",
        "assoc",
        str(linarg_path),
        str(pheno_path),
        "--pheno-name",
        "iid,height,bmi",
        "--covar-name",
        "iid,sex,age",
        "--repeat-covar",
        "--out",
        str(out_prefix),
        "--num-processes",
        "2",
    ]
    rc = cli._main(args)
    assert rc == 0 or rc is None

    out_parquet = Path(f"{out_prefix}.parquet")
    assert out_parquet.exists()

    df = pl.read_parquet(out_parquet)
    expected_cols = {"height_BETA", "height_SE", "bmi_BETA", "bmi_SE"}
    assert expected_cols.issubset(set(df.columns))
