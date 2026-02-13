import logging

from pathlib import Path

import polars as pl

from linear_dag import pipeline


def test_msc_step1_skip_paths_log_via_logger_not_stdout(tmp_path: Path, caplog, capsys):
    out_dir = tmp_path / "kodama"
    region = "chr1:1-2"
    small_job_id = 0

    (out_dir / "genotype_matrices").mkdir(parents=True, exist_ok=True)
    (out_dir / "forward_backward_graphs").mkdir(parents=True, exist_ok=True)
    (out_dir / "genotype_matrices" / f"{small_job_id}_{region}.h5").write_text("")
    (out_dir / "forward_backward_graphs" / f"{small_job_id}_{region}_forward_graph.h5").write_text("")

    jobs_metadata = tmp_path / "job_metadata.parquet"
    pl.DataFrame(
        {
            "small_job_id": [small_job_id],
            "large_job_id": [0],
            "small_region": [region],
            "large_region": [region],
            "vcf_path": ["dummy.vcf.gz"],
        }
    ).write_parquet(
        jobs_metadata,
        metadata={
            "flip_minor_alleles": "False",
            "keep": "None",
            "maf": "None",
            "remove_indels": "False",
            "remove_multiallelics": "False",
            "sex_path": "None",
            "mount_point": "",
            "out": str(out_dir),
            "large_partition_size": "1000",
            "n_small_blocks": "1",
        },
    )

    logger = logging.getLogger("linear_dag.tests.pipeline")
    with caplog.at_level(logging.INFO, logger=logger.name):
        pipeline.msc_step1(str(jobs_metadata), small_job_id, logger=logger)

    assert capsys.readouterr().out == ""
    assert any("Genotype matrix for 0_chr1:1-2 already exists. Skipping." in rec.message for rec in caplog.records)
    assert any(
        "Forward backward graph for 0_chr1:1-2 already exists. Skipping." in rec.message for rec in caplog.records
    )
