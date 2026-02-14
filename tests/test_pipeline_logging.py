import io
import logging

from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl

from linear_dag import genotype, pipeline
from linear_dag.core import linear_arg_inference as lai
from scipy.sparse import csc_matrix


def test_msc_step1_skip_paths_log_via_logger_not_stdout(tmp_path: Path, caplog):
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
    stdout = io.StringIO()
    with caplog.at_level(logging.INFO, logger=logger.name):
        with redirect_stdout(stdout):
            pipeline.msc_step1(str(jobs_metadata), small_job_id, logger=logger)

    assert stdout.getvalue() == ""
    assert any("Genotype matrix for 0_chr1:1-2 already exists. Skipping." in rec.message for rec in caplog.records)
    assert any(
        "Forward backward graph for 0_chr1:1-2 already exists. Skipping." in rec.message for rec in caplog.records
    )


def test_load_genotypes_does_not_print_progress(tmp_path: Path):
    prefix = tmp_path / "geno"
    np.savetxt(prefix.with_suffix(".txt"), np.array([[0, 1], [1, 0]]), fmt="%d")

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        genotype.load_genotypes(str(prefix))
    assert stdout.getvalue() == ""


def test_linear_arg_inference_logs_progress_when_logger_provided(monkeypatch, caplog):
    class _FakeBrickGraph:
        @staticmethod
        def from_genotypes(_genotypes):
            return "graph", np.array([0]), np.array([0])

    class _FakeRecombination:
        @staticmethod
        def from_graph(_graph):
            return _FakeRecombination()

        def find_recombinations(self):
            return None

    monkeypatch.setattr(lai, "BrickGraph", _FakeBrickGraph)
    monkeypatch.setattr(lai, "Recombination", _FakeRecombination)
    monkeypatch.setattr(lai, "linearize_brick_graph", lambda _recom: np.eye(2))

    genotypes = csc_matrix(np.array([[1], [0]]))
    flip = np.array([False])
    logger = logging.getLogger("linear_dag.tests.linear_arg_inference")
    stdout = io.StringIO()
    with caplog.at_level(logging.INFO, logger=logger.name):
        with redirect_stdout(stdout):
            lai.linear_arg_from_genotypes(
                genotypes,
                flip,
                variant_info=None,
                find_recombinations=False,
                logger=logger,
            )

    assert stdout.getvalue() == ""
    assert any("Inferring brick graph" in rec.message for rec in caplog.records)
    assert any("Finding recombinations" in rec.message for rec in caplog.records)
    assert any("Linearizing brick graph" in rec.message for rec in caplog.records)


def test_compress_vcf_passes_logger_to_lineararg_from_vcf(monkeypatch, tmp_path: Path):
    captured = {}

    class _FakeLinarg:
        shape = (4, 3)
        nnz = 5
        A = np.zeros((7, 7))
        num_nonunique_indices = 2
        allele_frequencies = np.array([0.25, 0.1, 0.4])

        def calculate_nonunique_indices(self):
            captured["calculate_nonunique_indices.called"] = True

        def write(self, output_h5, block_info=None):
            captured["write.output_h5"] = output_h5
            captured["write.block_info"] = block_info

    def _fake_from_vcf(**kwargs):
        captured["from_vcf.kwargs"] = kwargs
        return _FakeLinarg()

    monkeypatch.setattr(pipeline.LinearARG, "from_vcf", staticmethod(_fake_from_vcf))

    injected = logging.getLogger("linear_dag.tests.pipeline.compress")
    out_path = tmp_path / "out.h5"
    pipeline.compress_vcf(
        input_vcf="input.vcf.gz",
        output_h5=str(out_path),
        logger=injected,
    )

    assert captured["from_vcf.kwargs"]["logger"] is injected
    assert captured["calculate_nonunique_indices.called"]
    assert captured["write.output_h5"] == str(out_path)


def test_compress_vcf_creates_fallback_logger_when_logger_not_provided(monkeypatch, tmp_path: Path):
    captured = {}

    class _FakeLinarg:
        shape = (2, 2)
        nnz = 2
        A = np.zeros((4, 4))
        num_nonunique_indices = 1
        allele_frequencies = np.array([0.1, 0.2])

        def calculate_nonunique_indices(self):
            captured["calculate_nonunique_indices.called"] = True

        def write(self, output_h5, block_info=None):
            captured["write.output_h5"] = output_h5

    def _fake_from_vcf(**kwargs):
        captured["from_vcf.kwargs"] = kwargs
        return _FakeLinarg()

    monkeypatch.setattr(pipeline.LinearARG, "from_vcf", staticmethod(_fake_from_vcf))
    monkeypatch.chdir(tmp_path)

    out_path = tmp_path / "out.h5"
    pipeline.compress_vcf(
        input_vcf="input.vcf.gz",
        output_h5=str(out_path),
    )

    assert isinstance(captured["from_vcf.kwargs"]["logger"], logging.Logger)
    assert captured["calculate_nonunique_indices.called"]
    assert captured["write.output_h5"] == str(out_path)
