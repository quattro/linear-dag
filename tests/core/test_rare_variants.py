from __future__ import annotations

import h5py
import numpy as np
import polars as pl
import pytest

from scipy.sparse import csc_matrix

from linear_dag import augment_rare_variants_file, LinearARG, read_rare_variant_carriers
from linear_dag.rare_variants import _repack_hdf5_file


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


def _write_existing_doubleton_block(path):
    # The existing variant node has exactly two descendant haplotypes: the
    # first haplotype from each of the two individuals.
    adjacency = csc_matrix(
        (np.ones(2, dtype=np.int8), (np.array([4, 2]), np.array([0, 0]))),
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
    assert stats.direct_singletons == 3
    assert stats.reused_existing_nodes == 0
    assert stats.reused_new_nodes == 0
    assert stats.nodes_added == 1
    assert stats.edges_added == 2
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
    assert variant_nodes[125] == augmented.sample_indices[0]
    assert variant_nodes[150] == augmented.sample_indices[2]
    assert variant_nodes[151] == augmented.sample_indices[2]
    assert variant_nodes[160] not in set(augmented.sample_indices)

    with h5py.File(output_h5, "r") as handle:
        group = handle[block]
        assert not bool(group.attrs["rare_variant_phase_is_inferred"])
        assert bool(group.attrs["rare_variant_diploid_dosage_preserved"])
        assert group.attrs["rare_variant_singletons_direct"] == 3
        assert group.attrs["rare_variant_nodes_added"] == 1
        assert group.attrs["rare_variant_edges_added"] == 2
        assert "nonunique_indices" not in group


def test_singletons_add_no_graph_nodes_or_edges(tmp_path):
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
    assert stats.direct_singletons == 2
    assert stats.nodes_added == 0
    assert stats.edges_added == 0

    augmented = LinearARG.read(output_h5, block=block, load_metadata=True)
    new_nodes = augmented.variant_indices[1:]
    np.testing.assert_array_equal(new_nodes, np.repeat(augmented.sample_indices[2], 2))
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
        assert int(group.attrs["n"]) == original_n
        assert int(group.attrs["n_entries"]) == original_entries


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
    assert stats.direct_singletons == 0
    assert stats.reused_new_nodes == 1
    assert stats.nodes_added == 1
    assert stats.edges_added == 2

    augmented = LinearARG.read(output_h5, block="1:1-1000", load_metadata=True)
    assert augmented.variant_indices[1] == augmented.variant_indices[2]
    assert augmented.variant_indices[1] not in set(augmented.sample_indices)
    dense_diploid = augmented @ np.eye(augmented.shape[1])
    dense_diploid = dense_diploid[0::2] + dense_diploid[1::2]
    np.testing.assert_array_equal(dense_diploid[:, 1:], np.ones((2, 2)))


def test_doubleton_reuse_policies_separate_existing_and_within_run_reuse(tmp_path):
    input_h5 = tmp_path / "input.h5"
    carrier_table = tmp_path / "carriers.tsv"
    original = _write_existing_doubleton_block(input_h5)
    carrier_table.write_text(
        "CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n"
        "1\t150\tfirst\tG\tA\tiid1\t1\n"
        "1\t150\tfirst\tG\tA\tiid2\t1\n"
        "1\t151\tsecond\tT\tC\tiid1\t1\n"
        "1\t151\tsecond\tT\tC\tiid2\t1\n"
    )

    expected = {
        "existing_then_batch": {
            "reused_existing_nodes": 2,
            "distinct_existing_nodes_reused": 1,
            "reused_new_nodes": 0,
            "nodes_added": 0,
            "edges_added": 0,
            "existing_candidate_nodes_scanned": 1,
            "existing_signatures_available": 1,
        },
        "batch_only": {
            "reused_existing_nodes": 0,
            "distinct_existing_nodes_reused": 0,
            "reused_new_nodes": 1,
            "nodes_added": 1,
            "edges_added": 2,
            "existing_candidate_nodes_scanned": 0,
            "existing_signatures_available": 0,
        },
        "none": {
            "reused_existing_nodes": 0,
            "distinct_existing_nodes_reused": 0,
            "reused_new_nodes": 0,
            "nodes_added": 2,
            "edges_added": 4,
            "existing_candidate_nodes_scanned": 0,
            "existing_signatures_available": 0,
        },
    }
    diploid_outputs = {}
    for policy, expected_stats in expected.items():
        output_h5 = tmp_path / f"output-{policy}.h5"
        stats = augment_rare_variants_file(
            input_h5,
            carrier_table,
            output_h5,
            reuse_policy=policy,
        )

        assert stats.variants_added == 2
        assert stats.direct_singletons == 0
        assert stats.doubletons_added == 2
        for field, value in expected_stats.items():
            assert getattr(stats, field) == value
        assert stats.total_seconds > 0
        assert stats.file_repack_seconds > 0
        assert stats.matrix_load_seconds >= 0
        assert stats.existing_scan_seconds >= 0

        augmented = LinearARG.read(output_h5, block="1:1-1000", load_metadata=True)
        haplotypes = augmented @ np.eye(augmented.shape[1])
        diploid_outputs[policy] = haplotypes[0::2] + haplotypes[1::2]
        with h5py.File(output_h5, "r") as handle:
            assert handle.attrs["rare_variant_reuse_policy"] == policy
            assert handle["1:1-1000"].attrs["rare_variant_reuse_policy"] == policy

    for output in diploid_outputs.values():
        np.testing.assert_array_equal(output, diploid_outputs["existing_then_batch"])
    assert diploid_outputs["existing_then_batch"].shape == (2, 3)
    assert original.shape == (4, 1)


def test_augmentation_rejects_unknown_reuse_policy(tmp_path):
    input_h5 = tmp_path / "input.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n")

    with pytest.raises(ValueError, match="unknown reuse policy"):
        augment_rare_variants_file(
            input_h5,
            carrier_table,
            tmp_path / "output.h5",
            reuse_policy="invalid",
        )


def test_augmentation_accepts_adjacent_paired_haplotype_iids(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    with h5py.File(input_h5, "r+") as handle:
        individual_iids = handle["iids"][:]
        del handle["iids"]
        handle.create_dataset("iids", data=np.repeat(individual_iids, 2))
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\trs1\tG\tA\tiid2\t1\n")

    stats = augment_rare_variants_file(input_h5, carrier_table, output_h5)

    assert stats.variants_added == 1
    assert stats.iid_normalization_seconds >= 0
    assert output_h5.exists()


def test_augmentation_rejects_unpaired_haplotype_iids(tmp_path):
    input_h5 = tmp_path / "input.h5"
    output_h5 = tmp_path / "output.h5"
    carrier_table = tmp_path / "carriers.tsv"
    _write_small_block(input_h5)
    with h5py.File(input_h5, "r+") as handle:
        individual_iids = handle["iids"][:]
        del handle["iids"]
        handle.create_dataset("iids", data=np.tile(individual_iids, 2))
    carrier_table.write_text("CHROM\tPOS\tID\tREF\tALT\tIID\tDOSAGE\n1\t150\trs1\tG\tA\tiid2\t1\n")

    with pytest.raises(ValueError, match="must repeat in adjacent pairs"):
        augment_rare_variants_file(input_h5, carrier_table, output_h5)
    assert not output_h5.exists()


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


def test_repack_hdf5_file_reclaims_deleted_dataset_space(tmp_path):
    source_h5 = tmp_path / "source.h5"
    repacked_h5 = tmp_path / "repacked.h5"
    rng = np.random.default_rng(7)
    with h5py.File(source_h5, "w") as handle:
        handle.attrs["root"] = "preserved"
        group = handle.create_group("block")
        group.attrs["block"] = 1
        group.create_dataset(
            "discard",
            data=rng.integers(0, 256, size=1_000_000, dtype=np.uint8),
            compression="gzip",
        )
        group.create_dataset(
            "keep",
            data=np.arange(1000, dtype=np.int32),
            compression="gzip",
            shuffle=True,
        )
        del group["discard"]

    _repack_hdf5_file(source_h5, repacked_h5)

    assert repacked_h5.stat().st_size < source_h5.stat().st_size
    with h5py.File(source_h5, "r") as source, h5py.File(repacked_h5, "r") as repacked:
        assert dict(source.attrs) == dict(repacked.attrs)
        assert dict(source["block"].attrs) == dict(repacked["block"].attrs)
        np.testing.assert_array_equal(source["block/keep"][:], repacked["block/keep"][:])
        assert source["block/keep"].chunks == repacked["block/keep"].chunks
        assert source["block/keep"].compression == repacked["block/keep"].compression
        assert source["block/keep"].shuffle == repacked["block/keep"].shuffle
