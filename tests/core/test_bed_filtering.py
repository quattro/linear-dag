"""Tests for BED file filtering functionality."""

import os
import tempfile

import numpy as np
import polars as pl

from linear_dag.core.lineararg import LinearARG, list_blocks

# --- Tests for read_bed function ---


def test_read_bed_basic():
    """Test reading a basic 3-column BED file."""
    from linear_dag.bed_io import read_bed

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10100000\n")
        f.write("chr21\t10200000\t10300000\n")
        bed_path = f.name

    try:
        df = read_bed(bed_path)
        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 2
        assert set(df.columns) == {"chrom", "chromStart", "chromEnd"}
        assert df["chrom"][0] == "chr21"
        assert df["chromStart"][0] == 10000000
        assert df["chromEnd"][0] == 10100000
    finally:
        os.remove(bed_path)


def test_read_bed_skips_comments_and_headers():
    """Test that read_bed skips comment lines and track/browser lines."""
    from linear_dag.bed_io import read_bed

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("# This is a comment\n")
        f.write("browser position chr21:10000000-10100000\n")
        f.write("track name=test\n")
        f.write("chr21\t10000000\t10100000\n")
        bed_path = f.name

    try:
        df = read_bed(bed_path)
        assert df.shape[0] == 1
        assert df["chrom"][0] == "chr21"
    finally:
        os.remove(bed_path)


def test_read_bed_handles_whitespace():
    """Test that read_bed handles different whitespace separators."""
    from linear_dag.bed_io import read_bed

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21 10000000 10100000\n")  # spaces
        f.write("chr21\t10200000\t10300000\n")  # tabs
        bed_path = f.name

    try:
        df = read_bed(bed_path)
        assert df.shape[0] == 2
    finally:
        os.remove(bed_path)


# --- Tests for variants_in_bed_regions helper ---


def test_variants_in_bed_regions():
    """Test the helper function that checks which variants fall in BED regions."""
    from linear_dag.core.lineararg import variants_in_bed_regions

    bed_df = pl.DataFrame(
        {"chrom": ["chr21", "chr21"], "chromStart": [10000000, 10200000], "chromEnd": [10100000, 10300000]}
    )

    chrom = np.array(["chr21", "chr21", "chr21", "chr22"])
    pos = np.array([10050000, 10150000, 10250000, 10050000])

    mask = variants_in_bed_regions(chrom, pos, bed_df)

    expected = np.array([True, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_variants_in_bed_regions_edge_cases():
    """Test edge cases for BED region matching (0-based, half-open intervals)."""
    from linear_dag.core.lineararg import variants_in_bed_regions

    bed_df = pl.DataFrame({"chrom": ["chr21"], "chromStart": [100], "chromEnd": [200]})

    chrom = np.array(["chr21", "chr21", "chr21", "chr21"])
    pos = np.array([99, 100, 199, 200])  # BED is [start, end)

    mask = variants_in_bed_regions(chrom, pos, bed_df)

    # Position 100 is included (>=start), position 200 is excluded (<end)
    expected = np.array([False, True, True, False])
    np.testing.assert_array_equal(mask, expected)


# --- Tests for filter_variants_by_bed method ---


def test_filter_variants_by_bed(linarg_h5_path):
    """Test filtering a LinearARG by BED regions."""
    from linear_dag.bed_io import read_bed

    hdf5_path = linarg_h5_path
    blocks = list_blocks(hdf5_path)
    block_name = blocks["block_name"][0]

    linarg = LinearARG.read(hdf5_path, block=block_name, load_metadata=True)
    original_n_variants = linarg.shape[1]

    # Create a BED file covering part of the first block
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10050000\n")
        bed_path = f.name

    try:
        bed_df = read_bed(bed_path)
        linarg.filter_variants_by_bed(bed_df, maf_threshold=0.0)

        # Should have fewer variants after filtering
        assert linarg.shape[1] < original_n_variants
        assert linarg.shape[1] > 0
    finally:
        os.remove(bed_path)


# --- Tests for compute_filtered_variant_count ---


def test_compute_filtered_variant_count_maf_only(linarg_h5_path):
    """Test computing filtered variant count with MAF threshold only."""
    from linear_dag.core.lineararg import compute_filtered_variant_count

    hdf5_path = linarg_h5_path
    blocks = list_blocks(hdf5_path)
    block_name = blocks["block_name"][0]

    # Count with no filter
    count_all = compute_filtered_variant_count(hdf5_path, block_name, maf_threshold=0.0)

    # Count with MAF > 0.01
    count_filtered = compute_filtered_variant_count(hdf5_path, block_name, maf_threshold=0.01)

    assert count_filtered <= count_all
    assert count_filtered > 0


def test_compute_filtered_variant_count_bed_with_strict_outside(linarg_h5_path):
    """Test BED filtering with strict MAF threshold outside BED regions.

    When BED regions are provided:
    - Variants inside BED: use bed_maf_threshold (permissive)
    - Variants outside BED: use maf_threshold (stringent)

    Using a high maf_threshold outside BED should reduce total count compared
    to using a low threshold everywhere.
    """
    from linear_dag.bed_io import read_bed
    from linear_dag.core.lineararg import compute_filtered_variant_count

    hdf5_path = linarg_h5_path
    blocks = list_blocks(hdf5_path)
    block_name = blocks["block_name"][0]

    # Create a BED file covering part of the first block
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10050000\n")
        bed_path = f.name

    try:
        bed_df = read_bed(bed_path)

        # With permissive threshold everywhere
        count_all = compute_filtered_variant_count(hdf5_path, block_name, maf_threshold=0.0)

        # With strict threshold outside BED (MAF > 0.1), permissive inside (MAF > 0)
        count_bed_strict_outside = compute_filtered_variant_count(
            hdf5_path,
            block_name,
            maf_threshold=0.1,  # strict for outside BED
            bed_regions=bed_df,
            bed_maf_threshold=0.0,  # permissive for inside BED
        )

        # Strict filtering outside BED should reduce total count
        assert count_bed_strict_outside < count_all
        assert count_bed_strict_outside > 0
    finally:
        os.remove(bed_path)


def test_compute_filtered_variant_count_dual_threshold(linarg_h5_path):
    """Test computing filtered variant count with dual MAF thresholds."""
    from linear_dag.bed_io import read_bed
    from linear_dag.core.lineararg import compute_filtered_variant_count

    hdf5_path = linarg_h5_path
    blocks = list_blocks(hdf5_path)
    block_name = blocks["block_name"][0]

    # Create a BED file covering part of the first block
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10050000\n")
        bed_path = f.name

    try:
        bed_df = read_bed(bed_path)

        # More permissive threshold inside BED, more stringent outside
        count = compute_filtered_variant_count(
            hdf5_path,
            block_name,
            maf_threshold=0.05,  # stringent for outside BED
            bed_regions=bed_df,
            bed_maf_threshold=0.001,  # permissive for inside BED
        )

        assert count > 0
    finally:
        os.remove(bed_path)


# --- Tests for ParallelOperator with BED filtering ---


def test_parallel_operator_bed_filtering(linarg_h5_path):
    """Test that ParallelOperator correctly applies BED filtering."""
    from linear_dag.core.parallel_processing import ParallelOperator

    hdf5_path = linarg_h5_path

    # Create a BED file covering part of the data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10050000\n")
        f.write("chr21\t10270000\t10330000\n")
        bed_path = f.name

    try:
        # Without BED filtering
        with ParallelOperator.from_hdf5(hdf5_path, num_processes=2) as op_all:
            n_all = op_all.shape[1]

        # With BED filtering
        with ParallelOperator.from_hdf5(
            hdf5_path,
            num_processes=2,
            bed_file=bed_path,
            bed_maf_log10_threshold=-4,
        ) as op_bed:
            n_bed = op_bed.shape[1]

        # BED filtering should reduce variant count
        assert n_bed < n_all
        assert n_bed > 0
    finally:
        os.remove(bed_path)


def test_parallel_operator_bed_matmul_matches_serial(linarg_h5_path):
    """Test that parallel BED-filtered matmul matches serial version."""
    from linear_dag.bed_io import read_bed
    from linear_dag.core.lineararg import compute_variant_filter_mask
    from linear_dag.core.parallel_processing import ParallelOperator

    hdf5_path = linarg_h5_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10100000\n")
        bed_path = f.name

    try:
        bed_df = read_bed(bed_path)

        with ParallelOperator.from_hdf5(
            hdf5_path,
            num_processes=2,
            bed_file=bed_path,
            bed_maf_log10_threshold=-4,
            maf_log10_threshold=-2,
        ) as par:
            n, m = par.shape
            rng = np.random.default_rng(42)
            Y = rng.standard_normal((n, 3)).astype(np.float32)
            Z_par = par.T @ Y

        # Serial version
        blocks = list_blocks(hdf5_path)
        Z_parts = []
        for block_name in blocks["block_name"]:
            linarg = LinearARG.read(hdf5_path, block=block_name, load_metadata=True)
            mask = compute_variant_filter_mask(
                hdf5_path,
                block_name,
                maf_threshold=0.01,
                bed_regions=bed_df,
                bed_maf_threshold=0.0001,
            )
            linarg.filter_variants_by_mask(mask)
            Z_parts.append(linarg.T @ Y)
        Z_ser = np.vstack(Z_parts)

        np.testing.assert_allclose(Z_par, Z_ser, rtol=1e-5, atol=1e-5)
    finally:
        os.remove(bed_path)


def test_grm_operator_bed_filtering_shape_and_determinism(linarg_h5_path):
    from linear_dag.core.parallel_processing import GRMOperator

    hdf5_path = linarg_h5_path
    maf_log10_threshold = -1
    bed_maf_log10_threshold = -4

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10070000\n")
        f.write("chr21\t10270000\t10330000\n")
        bed_path = f.name

    try:
        rng = np.random.default_rng(1234)
        x = rng.standard_normal((50, 2)).astype(np.float32)
        with GRMOperator.from_hdf5(
            hdf5_path,
            num_processes=2,
            maf_log10_threshold=maf_log10_threshold,
            bed_file=bed_path,
            bed_maf_log10_threshold=bed_maf_log10_threshold,
        ) as grm:
            y_first = grm @ x
            expected_shape = grm.shape

        with GRMOperator.from_hdf5(
            hdf5_path,
            num_processes=2,
            maf_log10_threshold=maf_log10_threshold,
            bed_file=bed_path,
            bed_maf_log10_threshold=bed_maf_log10_threshold,
        ) as grm:
            y_second = grm @ x

        assert expected_shape[0] == expected_shape[1]
        assert y_first.shape == (expected_shape[0], x.shape[1])
        np.testing.assert_allclose(y_first, y_second, rtol=1e-5, atol=1e-5)
    finally:
        os.remove(bed_path)


def test_grm_operator_bed_filtering_matches_serial_and_filtered_counts(linarg_h5_path):
    from linear_dag.bed_io import read_bed
    from linear_dag.core.lineararg import compute_filtered_variant_count, compute_variant_filter_mask
    from linear_dag.core.parallel_processing import GRMOperator
    from scipy.sparse import diags
    from scipy.sparse.linalg import aslinearoperator

    hdf5_path = linarg_h5_path
    alpha = -1.0
    maf_log10_threshold = -1
    bed_maf_log10_threshold = -4
    maf_threshold = 10**maf_log10_threshold
    bed_maf_threshold = 10**bed_maf_log10_threshold

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write("chr21\t10000000\t10100000\n")
        bed_path = f.name

    try:
        bed_df = read_bed(bed_path)
        with GRMOperator.from_hdf5(
            hdf5_path,
            num_processes=2,
            maf_log10_threshold=maf_log10_threshold,
            bed_file=bed_path,
            bed_maf_log10_threshold=bed_maf_log10_threshold,
        ) as grm:
            n = grm.shape[0]
            rng = np.random.default_rng(9)
            x = rng.standard_normal((n, 3)).astype(np.float32)
            y_par = grm @ x

        y_ser = np.zeros_like(y_par)
        observed_filtered_variants = 0
        expected_filtered_variants = 0
        for block_name in list_blocks(hdf5_path)["block_name"]:
            expected_filtered_variants += compute_filtered_variant_count(
                hdf5_path,
                block_name,
                maf_threshold=maf_threshold,
                bed_regions=bed_df,
                bed_maf_threshold=bed_maf_threshold,
            )

            linarg = LinearARG.read(hdf5_path, block=block_name)
            mask = compute_variant_filter_mask(
                hdf5_path,
                block_name,
                maf_threshold=maf_threshold,
                bed_regions=bed_df,
                bed_maf_threshold=bed_maf_threshold,
            )
            observed_filtered_variants += int(mask.sum())
            allele_counts = linarg.allele_counts
            linarg.filter_variants_by_mask(mask)
            linarg.set_allele_counts(allele_counts[mask])
            linarg.nonunique_indices = None
            linarg.calculate_nonunique_indices()

            pq = linarg.allele_frequencies * (1 - linarg.allele_frequencies)
            K = aslinearoperator(diags(pq ** (1 + alpha)))
            y_ser += linarg.normalized @ K @ linarg.normalized.T @ x

        assert observed_filtered_variants == expected_filtered_variants
        np.testing.assert_allclose(y_par, y_ser, rtol=1e-3, atol=1e-3)
    finally:
        os.remove(bed_path)
