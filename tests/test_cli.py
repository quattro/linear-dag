import argparse
import logging
import shlex

from argparse import Namespace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from linear_dag import cli
from linear_dag.core.lineararg import load_variant_info
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


def test_cli_version_fallback(monkeypatch):
    def _raise_missing(_dist_name: str):
        raise cli.metadata.PackageNotFoundError

    monkeypatch.setattr(cli.metadata, "version", _raise_missing)
    assert cli._resolve_cli_version() == "vunknown"


def test_prep_data_requires_block_metadata():
    linarg_path = TEST_DATA_DIR / "tiny.ma.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match="No block metadata found"):
        cli._prep_data(str(linarg_path), str(pheno_path))


def test_closest_matches_returns_candidate_for_typo():
    matches = cli._closest_matches("heigt", ["height", "weight", "bmi"])
    assert matches
    assert "height" in matches


def test_closest_matches_returns_empty_for_unrelated_value():
    matches = cli._closest_matches("zzzzz", ["height", "weight", "bmi"])
    assert matches == []


def test_closest_matches_respects_limit():
    matches = cli._closest_matches("traita", ["trait1", "trait2", "trait3", "trait4"], limit=2)
    assert len(matches) <= 2


def test_format_suggestion_fragment_has_expected_forms():
    with_hint = cli._format_suggestion_fragment("heigt", ["height", "weight", "bmi"])
    without_hint = cli._format_suggestion_fragment("zzzzz", ["height", "weight", "bmi"])

    assert "Did you mean:" in with_hint
    assert "height" in with_hint
    assert without_hint == ""


def test_filter_blocks_rejects_block_names_and_chromosomes_together():
    block_metadata = cli.list_blocks(str(TEST_DATA_DIR / "test_chr21_50.h5"))
    with pytest.raises(ValueError, match="Specify either block_names or chromosomes"):
        cli._filter_blocks(block_metadata, chromosomes=["21"], block_names=["21_10000001.0_10200000.0"])


def test_load_required_block_metadata_invalid_block_name_has_actionable_error():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    with pytest.raises(ValueError, match=r"Unknown block name\(s\): missing_block"):
        cli._load_required_block_metadata(
            str(linarg_path),
            chromosomes=None,
            block_names=["missing_block"],
            command_name="assoc/rhe",
            logger=logger,
        )


def test_load_required_block_metadata_invalid_block_name_includes_suggestion():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    with pytest.raises(ValueError, match=r"Unknown block name\(s\): 21_10000001\.0_10200000\.x") as excinfo:
        cli._load_required_block_metadata(
            str(linarg_path),
            chromosomes=None,
            block_names=["21_10000001.0_10200000.x"],
            command_name="assoc/rhe",
            logger=logger,
        )
    msg = str(excinfo.value)
    assert "Did you mean:" in msg
    assert "21_10000001.0_10200000.0" in msg


def test_load_required_block_metadata_invalid_chromosome_has_actionable_error():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    with pytest.raises(ValueError, match=r"Unknown chromosome selection\(s\): not_a_chrom"):
        cli._load_required_block_metadata(
            str(linarg_path),
            chromosomes=["not_a_chrom"],
            block_names=None,
            command_name="assoc/rhe",
            logger=logger,
        )


def test_load_required_block_metadata_invalid_chromosome_includes_suggestion():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    with pytest.raises(ValueError, match=r"Unknown chromosome selection\(s\): 2_1") as excinfo:
        cli._load_required_block_metadata(
            str(linarg_path),
            chromosomes=["2_1"],
            block_names=None,
            command_name="assoc/rhe",
            logger=logger,
        )
    msg = str(excinfo.value)
    assert "Did you mean:" in msg
    assert "21" in msg


def test_load_required_block_metadata_accepts_chr_prefix_for_numeric_blocks():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    block_metadata = cli._load_required_block_metadata(
        str(linarg_path),
        chromosomes=["chr21"],
        block_names=None,
        command_name="assoc/rhe",
        logger=logger,
    )
    assert block_metadata.height == 2
    assert set(block_metadata.get_column("chrom").to_list()) == {21}


def test_read_pheno_or_covar_missing_named_column_has_actionable_error():
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"Requested column name\(s\) not found.*not_a_col"):
        cli._read_pheno_or_covar(str(pheno_path), columns=["iid", "not_a_col"])


def test_read_pheno_or_covar_missing_named_column_includes_suggestion():
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"Requested column name\(s\) not found.*heigt") as excinfo:
        cli._read_pheno_or_covar(str(pheno_path), columns=["iid", "heigt"])
    msg = str(excinfo.value)
    assert "Did you mean:" in msg
    assert "height" in msg


def test_prep_data_missing_covar_name_has_actionable_error():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"Requested column name\(s\) not found.*not_a_col"):
        cli._prep_data(
            str(linarg_path),
            str(pheno_path),
            pheno_names=["iid", "height"],
            covar=str(pheno_path),
            covar_names=["iid", "not_a_col"],
        )


def test_prep_data_missing_covar_name_includes_suggestion():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"Requested column name\(s\) not found.*heigt") as excinfo:
        cli._prep_data(
            str(linarg_path),
            str(pheno_path),
            pheno_names=["iid", "height"],
            covar=str(pheno_path),
            covar_names=["iid", "heigt"],
        )
    msg = str(excinfo.value)
    assert "Did you mean:" in msg
    assert "height" in msg


def test_load_required_block_metadata_invalid_block_name_without_suggestion_lists_available():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    with pytest.raises(ValueError, match=r"Unknown block name\(s\): missing_block") as excinfo:
        cli._load_required_block_metadata(
            str(linarg_path),
            chromosomes=None,
            block_names=["missing_block"],
            command_name="assoc/rhe",
            logger=logger,
        )
    msg = str(excinfo.value)
    assert "Available block names include:" in msg
    assert "Did you mean:" not in msg


def test_load_required_block_metadata_invalid_chromosome_without_suggestion_lists_available():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    logger = cli.MemoryLogger(__name__)
    with pytest.raises(ValueError, match=r"Unknown chromosome selection\(s\): not_a_chrom") as excinfo:
        cli._load_required_block_metadata(
            str(linarg_path),
            chromosomes=["not_a_chrom"],
            block_names=None,
            command_name="assoc/rhe",
            logger=logger,
        )
    msg = str(excinfo.value)
    assert "Available chromosomes:" in msg
    assert "Did you mean:" not in msg


def test_read_pheno_or_covar_missing_named_column_without_suggestion_lists_available():
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"Requested column name\(s\) not found.*totally_unknown_column") as excinfo:
        cli._read_pheno_or_covar(str(pheno_path), columns=["iid", "totally_unknown_column"])
    msg = str(excinfo.value)
    assert "Available columns:" in msg
    assert "Did you mean:" not in msg


def test_read_pheno_or_covar_out_of_bounds_col_index_reports_bounds():
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"out of bounds.*999") as excinfo:
        cli._read_pheno_or_covar(str(pheno_path), columns=[0, 999])
    msg = str(excinfo.value)
    assert "Valid range: 0..5" in msg
    assert "Total columns: 6" in msg


def test_read_pheno_or_covar_negative_col_index_rejected():
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match="Must supply valid column indices to read_pheno/read_covar"):
        cli._read_pheno_or_covar(str(pheno_path), columns=[0, -1])


def test_prep_data_out_of_bounds_pheno_col_nums_reports_bounds():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"out of bounds.*999") as excinfo:
        cli._prep_data(
            str(linarg_path),
            str(pheno_path),
            pheno_col_nums=[0, 999],
        )
    msg = str(excinfo.value)
    assert "Valid range: 0..5" in msg
    assert "Total columns: 6" in msg


def test_prep_data_out_of_bounds_covar_col_nums_reports_bounds():
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    with pytest.raises(ValueError, match=r"out of bounds.*999") as excinfo:
        cli._prep_data(
            str(linarg_path),
            str(pheno_path),
            pheno_col_nums=[0, 1],
            covar=str(pheno_path),
            covar_col_nums=[0, 999],
        )
    msg = str(excinfo.value)
    assert "Valid range: 0..5" in msg
    assert "Total columns: 6" in msg


class _DummyContext:
    def __init__(self, value):
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_validate_num_processes_rejects_non_positive():
    with pytest.raises(ValueError, match="num_processes must be greater than zero"):
        cli._validate_num_processes(0)
    assert cli._validate_num_processes(None) is None
    assert cli._validate_num_processes(2) == 2


def test_warn_if_num_processes_exceeds_available_emits_warning(caplog):
    logger = logging.getLogger("linear_dag.cli.test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        cli._warn_if_num_processes_exceeds_available(8, logger, available_cpus=4)
    assert any("exceeds available CPUs (4)" in rec.message for rec in caplog.records)


def test_warn_if_num_processes_exceeds_available_no_warning_when_within_limit(caplog):
    logger = logging.getLogger("linear_dag.cli.test")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        cli._warn_if_num_processes_exceeds_available(4, logger, available_cpus=4)
    assert not caplog.records


def test_assoc_parallel_operator_kwargs_consistent_across_modes(tmp_path: Path, monkeypatch):
    block_metadata = pl.DataFrame(
        {
            "block_name": ["1:0-100"],
            "chrom": ["1"],
            "n_samples": [2],
            "n_variants": [4],
            "n_entries": [8],
        }
    )
    phenotypes = pl.DataFrame(
        {
            "iid": ["id1", "id2"],
            "trait1": [0.1, 0.2],
            "trait2": [0.3, 0.4],
            "i0": [1.0, 1.0],
        }
    )

    def _fake_prep_data(*_args, **_kwargs):
        return block_metadata, ["i0"], ["trait1", "trait2"], phenotypes

    captured_kwargs = []

    def _fake_from_hdf5(_cls, _linarg_path, **kwargs):
        captured_kwargs.append(kwargs)
        return _DummyContext(object())

    def _fake_run_gwas(_genotypes, _data, pheno_cols, **_kwargs):
        columns = {}
        for pheno_name in pheno_cols:
            columns[f"{pheno_name}_BETA"] = [0.1]
            columns[f"{pheno_name}_SE"] = [0.2]
        return pl.DataFrame(columns).lazy()

    monkeypatch.setattr(cli, "_prep_data", _fake_prep_data)
    monkeypatch.setattr(cli.ParallelOperator, "from_hdf5", classmethod(_fake_from_hdf5))
    monkeypatch.setattr(cli, "run_gwas", _fake_run_gwas)

    args = Namespace(
        linarg_path="dummy.h5",
        pheno="dummy.tsv",
        pheno_name=None,
        pheno_col_nums=None,
        covar=None,
        covar_name=None,
        covar_col_nums=None,
        chromosomes=None,
        block_names=None,
        num_processes=3,
        no_variant_info=True,
        all_variant_info=False,
        no_hwe=False,
        repeat_covar=True,
        recompute_ac=False,
        maf_log10_threshold=-2,
        bed="regions.bed",
        bed_maf_log10_threshold=-4,
        out=str(tmp_path / "assoc_repeat"),
    )
    cli._assoc_scan(args)

    args.repeat_covar = False
    args.out = str(tmp_path / "assoc_default")
    cli._assoc_scan(args)

    assert len(captured_kwargs) == 2
    repeat_kwargs, default_kwargs = captured_kwargs

    assert repeat_kwargs["num_processes"] == default_kwargs["num_processes"] == 3
    assert repeat_kwargs["maf_log10_threshold"] == default_kwargs["maf_log10_threshold"] == -2
    assert repeat_kwargs["bed_file"] == default_kwargs["bed_file"] == "regions.bed"
    assert repeat_kwargs["bed_maf_log10_threshold"] == default_kwargs["bed_maf_log10_threshold"] == -4
    assert repeat_kwargs["block_metadata"].to_dicts() == default_kwargs["block_metadata"].to_dicts()
    assert repeat_kwargs["max_num_traits"] == 2
    assert default_kwargs["max_num_traits"] == 3


def test_estimate_h2g_passes_filtered_block_metadata_to_grm_operator(tmp_path: Path, monkeypatch):
    block_metadata = pl.DataFrame(
        {
            "block_name": ["1:0-100"],
            "chrom": ["1"],
            "n_samples": [2],
            "n_variants": [4],
            "n_entries": [8],
        }
    )
    phenotypes = pl.DataFrame(
        {
            "iid": ["id1", "id2"],
            "trait1": [0.1, 0.2],
            "i0": [1.0, 1.0],
        }
    )

    def _fake_prep_data(*_args, **_kwargs):
        return block_metadata, ["i0"], ["trait1"], phenotypes

    captured = {}

    def _fake_grm_from_hdf5(_cls, hdf5_file, **kwargs):
        captured["hdf5_file"] = hdf5_file
        captured.update(kwargs)
        return _DummyContext(object())

    def _fake_randomized_he(*_args, **_kwargs):
        return pl.DataFrame({"trait": ["trait1"], "h2g": [0.2]})

    monkeypatch.setattr(cli, "_prep_data", _fake_prep_data)
    monkeypatch.setattr(cli.GRMOperator, "from_hdf5", classmethod(_fake_grm_from_hdf5))
    monkeypatch.setattr(cli, "randomized_haseman_elston", _fake_randomized_he)

    args = Namespace(
        linarg_path="dummy.h5",
        pheno="dummy.tsv",
        pheno_name=None,
        pheno_col_nums=None,
        covar=None,
        covar_name=None,
        covar_col_nums=None,
        chromosomes=["1"],
        block_names=None,
        num_processes=4,
        num_matvecs=10,
        estimator="xnystrace",
        sampler="normal",
        seed=0,
        out=str(tmp_path / "rhe"),
    )
    cli._estimate_h2g(args)

    assert captured["hdf5_file"] == "dummy.h5"
    assert captured["num_processes"] == 4
    assert captured["alpha"] == -1.0
    assert captured["block_metadata"].to_dicts() == block_metadata.to_dicts()


def test_cli_help_includes_argument_groups(capsys):
    with pytest.raises(SystemExit):
        cli._main(["assoc", "--help"])
    assoc_help = capsys.readouterr().out
    assert "Association Model:" in assoc_help
    assert "Variant Output and Filtering:" in assoc_help
    assert "Phenotype and Covariate Columns:" in assoc_help

    with pytest.raises(SystemExit):
        cli._main(["score", "--help"])
    score_help = capsys.readouterr().out
    assert "Input:" in score_help
    assert "Block Selection:" in score_help
    assert "Execution and Output:" in score_help


def test_attach_variant_info_joins_with_explicit_row_alignment():
    association_results = pl.DataFrame(
        {
            "height_BETA": [0.1, 0.2],
            "height_SE": [0.01, 0.02],
        }
    ).lazy()
    variant_info = pl.DataFrame({"ID": [b"v1", b"v2"]}).lazy()

    out = cli._attach_variant_info(association_results, variant_info).collect()

    assert out.columns == ["ID", "height_BETA", "height_SE"]
    assert out["ID"].to_list() == [b"v1", b"v2"]
    assert out["height_BETA"].to_list() == [0.1, 0.2]


def test_attach_variant_info_rejects_row_count_mismatch():
    association_results = pl.DataFrame({"height_BETA": [0.1, 0.2]}).lazy()
    variant_info = pl.DataFrame({"ID": [b"v1"]}).lazy()

    with pytest.raises(ValueError, match="Variant metadata alignment failed"):
        cli._attach_variant_info(association_results, variant_info).collect()


def test_construct_cmd_string_is_copy_paste_executable():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    msc = subparsers.add_parser("multi-step-compress")
    msc_subparsers = msc.add_subparsers(dest="subcmd", required=True)
    step1 = msc_subparsers.add_parser("step1")
    step1.add_argument("--job-metadata", required=True)
    step1.add_argument("--small-job-id", required=True, type=int)

    argv = [
        "-v",
        "multi-step-compress",
        "step1",
        "--job-metadata",
        "job metadata.tsv",
        "--small-job-id",
        "3",
    ]
    parsed = parser.parse_args(argv)
    cmd_str = cli._construct_cmd_string(argv, parser, parsed)

    lines = cmd_str.splitlines()
    assert lines[0] == "kodama -v multi-step-compress step1 \\"
    assert lines[1].startswith("\t\t--job-metadata")
    assert lines[1].endswith(" \\")
    assert lines[2].startswith("\t\t--small-job-id 3")

    executable = cmd_str.replace("\\\n", " ")
    reconstructed_args = shlex.split(executable)[1:]
    assert parser.parse_args(reconstructed_args) == parsed


def test_run_cli_maps_system_exit_to_explicit_code(monkeypatch):
    def _fake_main(_args):
        raise SystemExit(2)

    monkeypatch.setattr(cli, "_main", _fake_main)
    monkeypatch.setattr(cli.sys, "argv", ["kodama", "assoc"])
    assert cli.run_cli() == 2


def test_infer_primary_subcommand_skips_global_flags():
    assert cli._infer_primary_subcommand(["-q", "--verbose", "assoc"]) == "assoc"


def test_infer_primary_subcommand_returns_none_without_known_subcommand():
    assert cli._infer_primary_subcommand(["-q", "--out", "result-prefix"]) is None


def test_run_cli_runtime_error_returns_one_and_stderr(monkeypatch, capsys):
    def _fake_main(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_main", _fake_main)
    monkeypatch.setattr(cli.sys, "argv", ["kodama", "assoc"])
    assert cli.run_cli() == 1
    assert "error: assoc: boom" in capsys.readouterr().err


def test_run_cli_runtime_error_without_subcommand_falls_back_to_plain_error(monkeypatch, capsys):
    def _fake_main(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_main", _fake_main)
    monkeypatch.setattr(cli.sys, "argv", ["kodama", "-q"])
    assert cli.run_cli() == 1
    assert "error: boom" in capsys.readouterr().err


def test_run_cli_invalid_block_name_returns_one_and_actionable_stderr(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_block"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-name",
            "iid,height",
            "--block-names",
            "missing_block",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    assert "Unknown block name(s): missing_block" in capsys.readouterr().err


def test_run_cli_invalid_block_name_includes_suggestion_in_stderr(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_block_typo"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-name",
            "iid,height",
            "--block-names",
            "21_10000001.0_10200000.x",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    stderr = capsys.readouterr().err
    assert "error: assoc:" in stderr
    assert "Unknown block name(s): 21_10000001.0_10200000.x" in stderr
    assert "Did you mean:" in stderr
    assert "21_10000001.0_10200000.0" in stderr


def test_run_cli_invalid_chromosome_returns_one_and_actionable_stderr(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_chrom"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-name",
            "iid,height",
            "--chromosomes",
            "not_a_chrom",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    assert "Unknown chromosome selection(s): not_a_chrom" in capsys.readouterr().err


def test_run_cli_invalid_chromosome_includes_suggestion_in_stderr(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_chrom_typo"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-name",
            "iid,height",
            "--chromosomes",
            "2_1",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    stderr = capsys.readouterr().err
    assert "error: assoc:" in stderr
    assert "Unknown chromosome selection(s): 2_1" in stderr
    assert "Did you mean:" in stderr
    assert "21" in stderr


def test_run_cli_missing_covar_column_returns_one_and_actionable_stderr(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_covar"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-name",
            "iid,height",
            "--covar",
            str(pheno_path),
            "--covar-name",
            "iid,not_a_col",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    assert "Requested column name(s) not found" in capsys.readouterr().err


def test_run_cli_missing_covar_column_includes_suggestion_in_stderr(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_covar_typo"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-name",
            "iid,height",
            "--covar",
            str(pheno_path),
            "--covar-name",
            "iid,heigt",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    stderr = capsys.readouterr().err
    assert "error: assoc:" in stderr
    assert "Requested column name(s) not found" in stderr
    assert "Did you mean:" in stderr
    assert "height" in stderr


def test_run_cli_out_of_bounds_pheno_col_nums_returns_bounds_error(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_pheno_col_num"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-col-nums",
            "0,999",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    stderr = capsys.readouterr().err
    assert "Requested column index value(s) out of bounds" in stderr
    assert "Valid range: 0..5" in stderr
    assert "Total columns: 6" in stderr


def test_run_cli_out_of_bounds_covar_col_nums_returns_bounds_error(monkeypatch, capsys, tmp_path: Path):
    linarg_path = TEST_DATA_DIR / "test_chr21_50.h5"
    pheno_path = TEST_DATA_DIR / "phenotypes_50.tsv"
    out_prefix = tmp_path / "invalid_covar_col_num"

    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "kodama",
            "-q",
            "assoc",
            str(linarg_path),
            str(pheno_path),
            "--pheno-col-nums",
            "0,1",
            "--covar",
            str(pheno_path),
            "--covar-col-nums",
            "0,999",
            "--out",
            str(out_prefix),
        ],
    )

    assert cli.run_cli() == 1
    stderr = capsys.readouterr().err
    assert "Requested column index value(s) out of bounds" in stderr
    assert "Valid range: 0..5" in stderr
    assert "Total columns: 6" in stderr
