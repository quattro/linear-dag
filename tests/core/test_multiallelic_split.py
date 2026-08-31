import logging
import shutil
import subprocess

import h5py
import numpy as np
import pytest

from scipy.sparse import csc_matrix

from linear_dag.core.lineararg import LinearARG
from linear_dag.genotype import read_vcf, write_vcf_to_hdf5

EXPECTED_PHASED = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    ],
    dtype=np.int8,
)

EXPECTED_VARIANTS = [
    (100, "biallelic", "G"),
    (110, "triallelic", "C"),
    (110, "triallelic", "G"),
    (120, "duplicate-first", "C"),
    (120, "duplicate-exact", "C"),
    (120, "duplicate-later", "C"),
    (120, "same-position-distinct", "G"),
    (130, "mixed-types", "C"),
    (130, "mixed-types", "AT"),
    (150, "high-frequency", "C"),
    (160, "zero-carrier-alt", "C"),
]


def test_native_phased_multiallelic_split_retains_duplicates_and_audits(test_data_dir, caplog):
    path = test_data_dir / "multiallelic_split.vcf"
    logger = logging.getLogger("linear_dag.tests.multiallelic")

    with caplog.at_level(logging.INFO, logger=logger.name):
        genotypes, flip, variants, iids = read_vcf(path, split_multiallelics=True, logger=logger)

    np.testing.assert_array_equal(genotypes.toarray(), EXPECTED_PHASED)
    expected_sparse = csc_matrix(EXPECTED_PHASED)
    np.testing.assert_array_equal(genotypes.indptr, expected_sparse.indptr)
    np.testing.assert_array_equal(genotypes.indices, expected_sparse.indices)
    np.testing.assert_array_equal(genotypes.data, expected_sparse.data)
    np.testing.assert_array_equal(np.asarray(genotypes.sum(axis=0)).ravel(), EXPECTED_PHASED.sum(axis=0))
    assert flip.tolist() == [False] * 11
    assert list(zip(variants["POS"], variants["ID"], variants["ALT"])) == EXPECTED_VARIANTS
    assert iids == ["S1", "S2", "S3", "S4"]
    audit = next(record.message for record in caplog.records if record.message.startswith("VCF read audit:"))
    assert "'missing_call_records_removed': 1" in audit
    assert "'zero_carrier_alts_removed': 1" in audit
    assert "'emitted_columns': 11" in audit
    assert "'biallelic_columns_emitted': 6" in audit
    assert "'multiallelic_alt_columns_emitted': 5" in audit
    duplicate_audit = next(
        record.message for record in caplog.records if record.message.startswith("Duplicate variant audit:")
    )
    assert "duplicate_keys=1" in duplicate_audit
    assert "extra_columns=2" in duplicate_audit
    assert "IDs=duplicate-first,duplicate-exact,duplicate-later" in duplicate_audit


def test_vcf_progress_and_phase_logging(test_data_dir, caplog, monkeypatch):
    logger = logging.getLogger("linear_dag.tests.multiallelic.progress")
    monkeypatch.setenv("KODAMA_VCF_PROGRESS_SECONDS", "1e-12")

    with caplog.at_level(logging.INFO, logger=logger.name):
        read_vcf(
            test_data_dir / "multiallelic_split.vcf",
            split_multiallelics=True,
            logger=logger,
        )

    messages = [record.message for record in caplog.records]
    assert any(message.startswith("Starting VCF iteration:") for message in messages)
    progress = next(message for message in messages if message.startswith("VCF progress:"))
    assert "records=" in progress
    assert "emitted_columns=" in progress
    assert "buffered_nonzeros=" in progress
    assert any(message.startswith("Finished VCF iteration:") for message in messages)
    assert any(message.startswith("Starting array concatenation:") for message in messages)
    assert any(message.startswith("Finished array concatenation:") for message in messages)
    assert any(message.startswith("Starting CSC construction:") for message in messages)
    assert any(message.startswith("Finished CSC construction:") for message in messages)


def test_duplicate_audit_also_runs_when_multiallelic_records_are_removed(test_data_dir, caplog):
    logger = logging.getLogger("linear_dag.tests.multiallelic.remove")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        genotypes, _, _, _ = read_vcf(
            test_data_dir / "multiallelic_split.vcf",
            remove_multiallelics=True,
            logger=logger,
        )

    assert genotypes.shape[1] == 6
    duplicate_audit = next(
        record.message for record in caplog.records if record.message.startswith("Duplicate variant audit:")
    )
    assert "duplicate_keys=1" in duplicate_audit
    assert "extra_columns=2" in duplicate_audit


def test_native_unphased_split_is_alt_specific_dosage(test_data_dir):
    genotypes, _, variants, _ = read_vcf(
        test_data_dir / "multiallelic_split.vcf",
        phased=False,
        split_multiallelics=True,
    )

    np.testing.assert_array_equal(genotypes.toarray(), EXPECTED_PHASED.reshape(4, 2, 11).sum(axis=1))
    assert list(zip(variants["POS"], variants["ID"], variants["ALT"])) == EXPECTED_VARIANTS


def test_streamed_multiallelic_split_matches_materialized_output(test_data_dir, tmp_path):
    path = test_data_dir / "multiallelic_split.vcf"
    output_path = tmp_path / "split-genotypes.h5"
    genotypes, flip, variants, iids = read_vcf(path, split_multiallelics=True)

    streamed_variants = write_vcf_to_hdf5(
        path,
        output_path,
        split_multiallelics=True,
        batch_nnz=7,
        batch_columns=2,
    )

    with h5py.File(output_path, "r") as output:
        streamed = csc_matrix(
            (output["data"][:], output["indices"][:], output["indptr"][:]),
            shape=output["shape"][:],
        )
        np.testing.assert_array_equal(output["flip"][:], flip)
        assert [iid.decode() for iid in output["iids"][:]] == [iid for iid in iids for _ in range(2)]
        assert not output.attrs["is_empty"]

    np.testing.assert_array_equal(streamed.toarray(), genotypes.toarray())
    assert streamed_variants.equals(variants)


def test_filters_and_flips_are_applied_per_alt(test_data_dir):
    path = test_data_dir / "multiallelic_split.vcf"
    genotypes, flip, variants, _ = read_vcf(
        path,
        split_multiallelics=True,
        remove_indels=True,
        flip_minor_alleles=True,
    )

    assert "AT" not in variants["ALT"].to_list()
    high_frequency_index = variants["ID"].to_list().index("high-frequency")
    duplicate_later_index = variants["ID"].to_list().index("duplicate-later")
    assert flip.sum() == 2
    assert flip[high_frequency_index]
    assert flip[duplicate_later_index]
    np.testing.assert_array_equal(genotypes[:, high_frequency_index].toarray().ravel(), [0, 0, 0, 0, 0, 0, 1, 0])


def test_maf_filter_is_applied_per_alt(test_data_dir, caplog):
    logger = logging.getLogger("linear_dag.tests.multiallelic.maf")
    with caplog.at_level(logging.INFO, logger=logger.name):
        _, _, variants, _ = read_vcf(
            test_data_dir / "multiallelic_split.vcf",
            split_multiallelics=True,
            maf_filter=0.2,
            logger=logger,
        )

    assert "high-frequency" not in variants["ID"].to_list()
    audit = next(record.message for record in caplog.records if record.message.startswith("VCF read audit:"))
    assert "'maf_alts_removed': 2" in audit


def test_split_and_remove_are_mutually_exclusive(test_data_dir):
    with pytest.raises(ValueError, match="mutually exclusive"):
        read_vcf(
            test_data_dir / "multiallelic_split.vcf",
            remove_multiallelics=True,
            split_multiallelics=True,
        )


def test_haploid_padding_is_ignored_after_sex_mask(tmp_path):
    path = tmp_path / "haploid.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=X,length=100>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tM\tF\n"
        "X\t1\thaploid\tA\tC,G\t.\tPASS\t.\tGT\t2|.\t0|1\n"
    )

    genotypes, _, variants, _ = read_vcf(path, split_multiallelics=True, sex=np.array([1, 0]))

    np.testing.assert_array_equal(genotypes.toarray(), [[0, 1], [0, 0], [1, 0]])
    assert variants["ALT"].to_list() == ["C", "G"]


def test_biallelic_output_is_unchanged_when_split_enabled(test_data_dir):
    path = test_data_dir / "1kg_small.vcf"
    default = read_vcf(path)
    split = read_vcf(path, split_multiallelics=True)

    np.testing.assert_array_equal(default[0].toarray(), split[0].toarray())
    np.testing.assert_array_equal(default[1], split[1])
    assert default[2].equals(split[2])
    assert default[3] == split[3]


def test_lineararg_reconstructs_native_split_matrix(test_data_dir):
    linarg, genotypes = LinearARG.from_vcf(
        test_data_dir / "multiallelic_split.vcf",
        split_multiallelics=True,
        return_genotypes=True,
    )

    np.testing.assert_allclose(linarg @ np.eye(genotypes.shape[1]), genotypes.toarray())


def test_native_split_matches_bcftools_118(test_data_dir, tmp_path):
    bcftools = shutil.which("bcftools")
    if bcftools is None:
        pytest.skip("bcftools 1.18 is required for the reference comparison")
    version = subprocess.run([bcftools, "--version"], check=True, capture_output=True, text=True).stdout
    if not version.startswith("bcftools 1.18"):
        pytest.skip("reference comparison is pinned to bcftools 1.18")

    split_vcf = tmp_path / "split.vcf"
    subprocess.run(
        [
            bcftools,
            "norm",
            "-m",
            "-any",
            "-Ov",
            "-o",
            split_vcf,
            test_data_dir / "multiallelic_split.vcf",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    native = read_vcf(test_data_dir / "multiallelic_split.vcf", split_multiallelics=True)
    # Reuse the enabled-path missing-record and zero-carrier policies on the
    # already split reference; raw bcftools intentionally retains both and all
    # duplicate records.
    reference = read_vcf(split_vcf, split_multiallelics=True)

    np.testing.assert_array_equal(native[0].toarray(), reference[0].toarray())
    np.testing.assert_array_equal(native[1], reference[1])
    assert native[2].equals(reference[2])
