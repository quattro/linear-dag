#!/usr/bin/env python3
"""Validate a full rare-variant augmentation with bounded carrier memory."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time

from pathlib import Path

import h5py
import numpy as np

from augment_rare_variants_blockwise import (
    block_intervals,
    extend_chromosome_edges,
    hdf5_summary,
    partition_carriers,
)

from linear_dag.rare_variants import (
    _block_names,
    _choose_signature,
    _decode,
    _diploid_iids,
    _new_metadata_values,
    _phase_candidates,
    _read_carrier_table,
    _sample_indices,
    META_COLUMNS,
)


def output_block_name(
    h5: h5py.File,
    source_name: str,
    chrom: object,
    source_start: int,
    source_end: int,
    observed_bounds: tuple[int | None, int | None],
) -> str:
    """Match an input block to its unchanged or edge-extended output block."""
    if source_name in h5:
        return source_name
    observed_start, observed_end = observed_bounds
    candidates = []
    for name in _block_names(h5):
        group = h5[name]
        start = int(group.attrs.get("start", 0))
        end = int(group.attrs.get("end", np.iinfo(np.int64).max))
        if not _decode(group.attrs.get("chrom", name.split(":", 1)[0])).removeprefix("chr") == _decode(
            chrom
        ).removeprefix("chr"):
            continue
        if source_start < start or source_end > end:
            continue
        if observed_start is not None and observed_start < start:
            continue
        if observed_end is not None and observed_end > end:
            continue
        candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(f"source block {source_name} matched {len(candidates)} output blocks")
    return candidates[0]


def validate_block(
    source: h5py.Group,
    output: h5py.Group,
    variants,
    iid_to_index: dict[str, int],
) -> dict[str, object]:
    """Replay batch-only placement and compare all affected arrays."""
    ordered = sorted(variants, key=lambda value: (value.pos, value.ref, value.alt, value.variant_id))
    old_n = int(source.attrs["n"])
    n_samples = int(source.attrs["n_samples"])
    n_individual_nodes = int(source.attrs.get("n_individuals", 0))
    insert_at = old_n - n_individual_nodes - n_samples
    old_samples = _sample_indices(old_n, n_samples, n_individual_nodes)

    selected: dict[tuple[int, ...], int] = {}
    signatures: list[tuple[int, ...]] = []
    source_kind = np.empty(len(ordered), dtype=np.uint8)
    source_value = np.empty(len(ordered), dtype=np.int32)
    counts = np.empty(len(ordered), dtype=np.int64)
    singletons = 0
    doubletons = 0
    reused = 0
    for index, variant in enumerate(ordered):
        candidates = _phase_candidates(variant, iid_to_index)
        counts[index] = variant.allele_count
        if variant.allele_count == 1:
            source_kind[index] = 0
            source_value[index] = old_samples[min(candidates)[0]]
            singletons += 1
            continue
        doubletons += 1
        signature, source_type = _choose_signature(candidates, {}, selected)
        source_kind[index] = 1
        if source_type == "selected":
            source_value[index] = selected[signature]
            reused += 1
        else:
            source_value[index] = len(signatures)
            selected[signature] = len(signatures)
            signatures.append(signature)

    nodes_added = len(signatures)
    edges_added = sum(map(len, signatures))
    mapping = np.arange(old_n, dtype=np.int32)
    mapping[insert_at:] += nodes_added
    new_variant_indices = np.empty(len(ordered), dtype=np.int32)
    sample_mask = source_kind == 0
    new_variant_indices[sample_mask] = mapping[source_value[sample_mask]]
    new_variant_indices[~sample_mask] = insert_at + source_value[~sample_mask]

    old_positions = np.asarray(source["POS"][:], dtype=np.int64)
    new_positions = _new_metadata_values(ordered, "POS")
    positions = np.concatenate((old_positions, new_positions))
    order = np.argsort(positions, kind="stable")
    mismatches = []

    expected_variant_indices = np.concatenate(
        (mapping[np.asarray(source["variant_indices"][:], dtype=np.int32)], new_variant_indices)
    )[order]
    if not np.array_equal(expected_variant_indices, output["variant_indices"][:]):
        mismatches.append("variant_indices")
    expected_flip = np.concatenate((np.asarray(source["flip"][:], dtype=bool), np.zeros(len(ordered), dtype=bool)))[
        order
    ]
    if not np.array_equal(expected_flip, output["flip"][:]):
        mismatches.append("flip")
    for name in META_COLUMNS:
        if name == "POS":
            expected = positions[order]
        else:
            new_values = _new_metadata_values(ordered, name)
            new_values = np.asarray(
                [value.encode("utf-8") if isinstance(value, str) else value for value in new_values],
                dtype=object,
            )
            expected = np.concatenate(
                (
                    np.asarray(source[name][:], dtype=object),
                    new_values,
                )
            )[order]
        if not np.array_equal(expected, output[name][:]):
            mismatches.append(name)
        del expected
    if "allele_counts" in source:
        expected_counts = np.concatenate((np.asarray(source["allele_counts"][:], dtype=np.int64), counts))[order]
        if not np.array_equal(expected_counts, output["allele_counts"][:]):
            mismatches.append("allele_counts")

    pointers = np.asarray(output["indptr"][insert_at : insert_at + nodes_added + 1], dtype=np.int64)
    if len(pointers) != nodes_added + 1 or not np.all(np.diff(pointers) == 2):
        mismatches.append("new_node_indptr")
    else:
        edge_start = int(pointers[0])
        edge_stop = int(pointers[-1])
        observed_rows = np.asarray(output["indices"][edge_start:edge_stop], dtype=np.int32)
        observed_data = np.asarray(output["data"][edge_start:edge_stop], dtype=np.int32)
        expected_rows = np.empty(edges_added, dtype=np.int32)
        cursor = 0
        for signature in signatures:
            stop = cursor + len(signature)
            expected_rows[cursor:stop] = np.sort(mapping[old_samples[np.asarray(signature, dtype=np.int64)]])
            cursor = stop
        if not np.array_equal(expected_rows, observed_rows):
            mismatches.append("new_node_indices")
        if not np.array_equal(np.ones(edges_added, dtype=np.int32), observed_data):
            mismatches.append("new_node_data")

    shape_checks = {
        "n": int(output.attrs["n"]) == old_n + nodes_added,
        "n_entries": int(output.attrs["n_entries"]) == int(source.attrs["n_entries"]) + edges_added,
        "n_variants": int(output.attrs["n_variants"]) == int(source.attrs["n_variants"]) + len(ordered),
        "indptr": len(output["indptr"]) == int(output.attrs["n"]) + 1,
        "indices": len(output["indices"]) == int(output.attrs["n_entries"]),
        "data": len(output["data"]) == int(output.attrs["n_entries"]),
    }
    mismatches.extend(f"shape:{name}" for name, passed in shape_checks.items() if not passed)
    return {
        "variants_checked": len(ordered),
        "singletons_checked": singletons,
        "doubletons_checked": doubletons,
        "within_batch_reuses_checked": reused,
        "new_nodes_checked": nodes_added,
        "new_edges_checked": edges_added,
        "mismatch_checks": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--carrier-table", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    if args.result_json.exists():
        raise FileExistsError(f"result already exists: {args.result_json}")

    started = time.perf_counter()
    source_intervals = block_intervals(args.input_h5)
    assignment_intervals = extend_chromosome_edges(source_intervals)
    blocks = []
    with tempfile.TemporaryDirectory(prefix="rare-validation-blocks-", dir=args.result_json.parent) as temp_dir:
        paths, _, bounds = partition_carriers(args.carrier_table, assignment_intervals, Path(temp_dir))
        with h5py.File(args.input_h5, "r") as source_h5, h5py.File(args.output_h5, "r") as output_h5:
            sample_counts = {int(output_h5[name].attrs["n_samples"]) for name in _block_names(output_h5)}
            if len(sample_counts) != 1:
                raise ValueError("output blocks have inconsistent sample counts")
            iids = _diploid_iids([_decode(value) for value in output_h5["iids"][:]], sample_counts.pop())
            iid_to_index = {iid: index for index, iid in enumerate(iids)}
            if len(iid_to_index) != len(iids):
                raise ValueError("output contains duplicate diploid IIDs")
            for source_name, chrom, start, end in source_intervals:
                target_name = output_block_name(
                    output_h5,
                    source_name,
                    chrom,
                    start,
                    end,
                    bounds[source_name],
                )
                variants = _read_carrier_table(paths[source_name])
                block_result = validate_block(
                    source_h5[source_name],
                    output_h5[target_name],
                    variants,
                    iid_to_index,
                )
                block_result["source_block"] = source_name
                block_result["output_block"] = target_name
                blocks.append(block_result)
                print(
                    json.dumps(
                        {
                            "output_block": target_name,
                            "variants_checked": block_result["variants_checked"],
                            "mismatch_checks": len(block_result["mismatch_checks"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del variants
                gc.collect()

            root_checks = {
                "reuse_policy": str(output_h5.attrs.get("rare_variant_reuse_policy")) == "batch_only",
                "dosage_preserved": bool(output_h5.attrs.get("rare_variant_diploid_dosage_preserved")),
                "variants_added": int(output_h5.attrs.get("rare_variants_added", -1))
                == sum(int(block["variants_checked"]) for block in blocks),
                "nodes_added": int(output_h5.attrs.get("rare_variant_nodes_added", -1))
                == sum(int(block["new_nodes_checked"]) for block in blocks),
                "edges_added": int(output_h5.attrs.get("rare_variant_edges_added", -1))
                == sum(int(block["new_edges_checked"]) for block in blocks),
            }

    result = {
        "input": hdf5_summary(args.input_h5),
        "output": hdf5_summary(args.output_h5),
        "blocks": blocks,
        "root_checks": root_checks,
        "wall_seconds": time.perf_counter() - started,
        "totals": {
            "variants_checked": sum(int(block["variants_checked"]) for block in blocks),
            "singletons_checked": sum(int(block["singletons_checked"]) for block in blocks),
            "doubletons_checked": sum(int(block["doubletons_checked"]) for block in blocks),
            "within_batch_reuses_checked": sum(int(block["within_batch_reuses_checked"]) for block in blocks),
            "new_nodes_checked": sum(int(block["new_nodes_checked"]) for block in blocks),
            "new_edges_checked": sum(int(block["new_edges_checked"]) for block in blocks),
            "mismatch_checks": sum(len(block["mismatch_checks"]) for block in blocks),
            "failed_root_checks": sum(not passed for passed in root_checks.values()),
        },
    }
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result_json": str(args.result_json), **result["totals"]}, sort_keys=True))


if __name__ == "__main__":
    main()
