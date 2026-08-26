from __future__ import annotations

import h5py
import numpy as np
import polars as pl
import pytest

from scipy.sparse import csc_matrix

from linear_dag import augment_rare_variants_file, LinearARG, read_rare_variant_carriers


def _write_small_block(path):
    # Variant 100 is carried on the first haplotype. Sample nodes are trailing
    # nodes in reverse row order, matching the LinearARG storage invariant.
    adjacency = csc_matrix(
        (np.ones(1, dtype=np.int8), (np.array([4]), np.array([0]))),
        shape=(5, 5),
    )
    linarg = LinearARG(
        adjacency,
        np.array([0], dtype=np.int32),
        np.array([False]),
        np.int32(4),
        variants=pl.DataFrame({"CHROM": ["1"], "POS": [100], "ID": ["existing"], "REF": ["A"], "ALT": ["G"]}).lazy(),
        iids=pl.Series(["iid1", "iid2"]),
    ).add_individual_nodes()
    linarg.write(path, block_info={"chrom": "1", "start": 1, "end": 1000})
    return linarg


def test_augment_rare_variants_file_preserves_diploid_dosage(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    original = _write_small_block(input_h5)
    carrier_table.write_text(
        "CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n"
        "1\t125\treuse-existing\tC\tT\tiid1\t1\n"
        "1\t150\tnew-node\tG\tA\tiid2\t1\n"
        "1\t151\treuse-new\tT\tC\tiid2\t1\n"
        "1\t160\thomozygous\tA\tC\tiid1\t2\n"
    )

    stats = augment_rare_variants_file(input_h5, carrier_table, output_h5)

    assert stats.variants_added == 4
    assert stats.reused_existing_nodes == 1
    assert stats.reused_new_nodes == 1
    assert stats.nodes_added == 2
    assert stats.edges_added == 3
    assert output_h5.exists()
    assert original.shape == (4, 1)

    block = "1:1-1000"
    augmented = LinearARG.read(output_h5, block=block, load_metadata=True)
    dense_haplotypes = augmented @ np.eye(augmented.shape[1])
    dense_diploid = dense_haplotypes[0::2] + dense_haplotypes[1::2]
    np.testing.assert_array_equal(
        dense_diploid,
        np.array(
            [
                [1, 1, 0, 0, 2],
                [0, 0, 1, 1, 0],
            ]
        ),
    )
    assert augmented.variants.collect().get_column("POS").to_list() == [100, 125, 150, 151, 160]
    np.testing.assert_array_equal(augmented.allele_counts, np.array([1, 1, 1, 1, 2]))
    assert augmented.n_individuals == 2

    with h5py.File(output_h5, "r") as handle:
        group = handle[block]
        assert not bool(group.attrs["rare_variant_phase_is_inferred"])
        assert bool(group.attrs["rare_variant_diploid_dosage_preserved"])
        assert "nonunique_indices" not in group


def test_lineararg_static_augmentation_entry_point(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\trs1\tG\tA\tiid2\t1\n")

    stats = LinearARG.augment_rare_variants_file(input_h5, carrier_table, output_h5)

    assert stats.variants_added == 1
    assert output_h5.exists()


def test_read_rare_variant_carriers_rejects_non_rare_total(tmp_path):
    carrier_table = tmp_path / "carriers.tsv"
    carrier_table.write_text(
        "CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\trs1\tG\tA\tiid1\t2\n1\t150\trs1\tG\tA\tiid2\t1\n"
    )

    with pytest.raises(ValueError, match="allele count 3; expected 1 or 2"):
        read_rare_variant_carriers(carrier_table)


def test_augmentation_never_overwrites_existing_output(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n")
    output_h5.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="output already exists"):
        augment_rare_variants_file(input_h5, carrier_table, output_h5)
    assert output_h5.read_bytes() == b"keep"
