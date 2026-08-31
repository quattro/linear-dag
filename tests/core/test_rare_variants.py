from __future__ import annotations

import h5py
import numpy as np
import polars as pl
import pytest

from scipy.sparse import csc_matrix

from linear_dag import augment_rare_variants_file, LinearARG, read_rare_variant_carriers


def _write_small_block(path, iids=None):
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
        iids=pl.Series(iids or ["iid1", "iid1", "iid2", "iid2"]),
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

    variant_nodes = dict(
        zip(augmented.variants.collect().get_column("POS").to_list(), augmented.variant_indices.tolist())
    )
    assert variant_nodes[125] == variant_nodes[100]
    assert variant_nodes[150] == variant_nodes[151]
    assert all(node not in set(augmented.sample_indices) for node in variant_nodes.values())

    with h5py.File(output_h5, "r") as handle:
        group = handle[block]
        assert not bool(group.attrs["rare_variant_phase_is_inferred"])
        assert bool(group.attrs["rare_variant_diploid_dosage_preserved"])
        assert "rare_variant_singletons_direct" not in group.attrs
        assert group.attrs["rare_variant_nodes_added"] == 2
        assert group.attrs["rare_variant_edges_added"] == 3
        assert "nonunique_indices" not in group


def test_singletons_share_a_new_internal_node(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    carrier_table.write_text(
        "CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\tfirst\tG\tA\tiid2\t1\n1\t151\tsecond\tT\tC\tiid2\t1\n"
    )

    block = "1:1-1000"
    with h5py.File(input_h5, "r") as handle:
        original_n = int(handle[block].attrs["n"])
        original_entries = int(handle[block].attrs["n_entries"])

    stats = augment_rare_variants_file(input_h5, carrier_table, output_h5)

    assert stats.variants_added == 2
    assert stats.reused_new_nodes == 1
    assert stats.nodes_added == 1
    assert stats.edges_added == 1

    augmented = LinearARG.read(output_h5, block=block, load_metadata=True)
    new_nodes = augmented.variant_indices[1:]
    np.testing.assert_array_equal(new_nodes, np.repeat(new_nodes[0], 2))
    assert new_nodes[0] not in set(augmented.sample_indices)
    np.testing.assert_array_equal(
        augmented @ np.eye(augmented.shape[1]),
        np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 1],
                [0, 0, 0],
            ]
        ),
    )

    with h5py.File(output_h5, "r") as handle:
        group = handle[block]
        assert int(group.attrs["n"]) == original_n + 1
        assert int(group.attrs["n_entries"]) == original_entries + 1


def test_doubletons_share_a_new_internal_node(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    carrier_table.write_text(
        "CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n"
        "1\t150\tfirst\tG\tA\tiid1\t1\n"
        "1\t150\tfirst\tG\tA\tiid2\t1\n"
        "1\t151\tsecond\tT\tC\tiid1\t1\n"
        "1\t151\tsecond\tT\tC\tiid2\t1\n"
    )

    stats = augment_rare_variants_file(input_h5, carrier_table, output_h5)

    assert stats.variants_added == 2
    assert stats.reused_new_nodes == 1
    assert stats.nodes_added == 1
    assert stats.edges_added == 2

    augmented = LinearARG.read(output_h5, block="1:1-1000", load_metadata=True)
    assert augmented.variant_indices[1] == augmented.variant_indices[2]
    assert augmented.variant_indices[1] not in set(augmented.sample_indices)
    dense_diploid = augmented @ np.eye(augmented.shape[1])
    dense_diploid = dense_diploid[0::2] + dense_diploid[1::2]
    np.testing.assert_array_equal(dense_diploid[:, 1:], np.ones((2, 2)))


def test_augmentation_accepts_deployed_haplotype_iids(linarg_h5_path, first_block_name, tmp_path):
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    with h5py.File(linarg_h5_path, "r") as handle:
        group = handle[first_block_name]
        iid = group.file["iids"][0].decode("utf-8")
        existing_positions = set(map(int, group["POS"][:]))
        pos = int(group.attrs["start"])
        while pos in existing_positions:
            pos += 1
        assert pos <= int(group.attrs["end"])
        output_chrom = group["CHROM"][0].decode("utf-8")

    carrier_table.write_text(f"CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n21\t{pos}\tnew\tA\tC\t{iid}\t1\n")

    stats = augment_rare_variants_file(linarg_h5_path, carrier_table, output_h5)

    assert stats.variants_added == 1
    augmented = LinearARG.read(output_h5, block=first_block_name, load_metadata=True)
    metadata = augmented.variants.collect()
    column = metadata.get_column("POS").to_list().index(pos)
    coefficients = np.zeros(augmented.shape[1])
    coefficients[column] = 1
    haplotype_dosage = augmented @ coefficients
    diploid_dosage = haplotype_dosage[0::2] + haplotype_dosage[1::2]
    np.testing.assert_array_equal(diploid_dosage, np.r_[1, np.zeros(len(diploid_dosage) - 1)])
    assert metadata.get_column("CHROM")[column] == output_chrom


def test_chromosome_normalization_prevents_duplicate_alleles(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    with h5py.File(input_h5, "r+") as handle:
        handle["1:1-1000"]["CHROM"][0] = "chr1"
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t100\tduplicate\tA\tG\tiid1\t1\n")

    with pytest.raises(ValueError, match="already exists"):
        augment_rare_variants_file(input_h5, carrier_table, output_h5)


def test_augmentation_rejects_nonpaired_haplotype_iids(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    with h5py.File(input_h5, "r+") as handle:
        handle["iids"][1] = "different"
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\tnew\tG\tA\tiid2\t1\n")

    with pytest.raises(ValueError, match="not adjacent matching pairs"):
        augment_rare_variants_file(input_h5, carrier_table, output_h5)


def test_singleton_internal_nodes_survive_sample_removal(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5, iids=["1", "1", "2", "2"])
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\tnew\tG\tA\t2\t1\n")
    augment_rare_variants_file(input_h5, carrier_table, output_h5)
    augmented = LinearARG.read(output_h5, block="1:1-1000", load_metadata=True)

    assert augmented.variant_indices[1] not in set(augmented.sample_indices)
    without_noncarrier = augmented.remove_samples(np.array(["1"]))
    without_carrier = augmented.remove_samples(np.array(["2"]))
    np.testing.assert_array_equal(
        (without_noncarrier @ np.eye(without_noncarrier.shape[1])).sum(axis=0),
        np.array([0, 1]),
    )
    np.testing.assert_array_equal(
        (without_carrier @ np.eye(without_carrier.shape[1])).sum(axis=0),
        np.array([1, 0]),
    )


def test_augmentation_updates_threshold_counts(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    with h5py.File(input_h5, "r+") as handle:
        group = handle["1:1-1000"]
        group.attrs["threshold_values"] = np.array([0.3, 0.1])
        group.attrs["threshold_n_variants"] = np.array([0, 1])
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\tnew\tG\tA\tiid2\t1\n")

    augment_rare_variants_file(input_h5, carrier_table, output_h5)

    with h5py.File(output_h5, "r") as handle:
        np.testing.assert_array_equal(
            handle["1:1-1000"].attrs["threshold_n_variants"],
            np.array([0, 2]),
        )


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
