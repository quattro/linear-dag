#!/usr/bin/env python3
"""Augment a block-structured LinearARG with bounded carrier-table memory."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import tempfile
import threading
import time

from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import psutil

from linear_dag import REUSE_POLICIES
from linear_dag.rare_variants import (
    _augment_block,
    _block_names,
    _chrom_equal,
    _decode,
    _diploid_iids,
    _phase_method,
    _read_carrier_table,
    _validate_reuse_policy,
    AugmentationStats,
    REQUIRED_COLUMNS,
)


def process_tree_rss(process: psutil.Process) -> int:
    """Return resident bytes for this process and its children."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def hdf5_summary(path: Path) -> dict[str, object]:
    """Return aggregate graph dimensions without exposing graph records."""
    with h5py.File(path, "r") as h5:
        blocks = [name for name in h5 if name != "iids" and isinstance(h5[name], h5py.Group)]
        return {
            "bytes": path.stat().st_size,
            "blocks": len(blocks),
            "nodes": sum(int(h5[name].attrs["n"]) for name in blocks),
            "edges": sum(int(h5[name].attrs["n_entries"]) for name in blocks),
            "variants": sum(int(h5[name].attrs["n_variants"]) for name in blocks),
        }


def block_intervals(path: Path) -> list[tuple[str, object, int, int]]:
    """Read block interval metadata in genomic order."""
    with h5py.File(path, "r") as h5:
        result = []
        for name in _block_names(h5):
            group = h5[name]
            result.append(
                (
                    name,
                    group.attrs.get("chrom", name.split(":", 1)[0]),
                    int(group.attrs.get("start", 0)),
                    int(group.attrs.get("end", np.iinfo(np.int64).max)),
                )
            )
    return sorted(result, key=lambda value: (str(value[1]), value[2], value[3], value[0]))


def extend_chromosome_edges(
    intervals: list[tuple[str, object, int, int]],
) -> list[tuple[str, object, int, int]]:
    """Extend only the first and last block of each chromosome for assignment."""
    result = list(intervals)
    chromosome_indices: dict[str, list[int]] = {}
    for index, (_, chrom, _, _) in enumerate(result):
        chromosome_indices.setdefault(_decode(chrom).removeprefix("chr"), []).append(index)
    for indices in chromosome_indices.values():
        first = indices[0]
        last = indices[-1]
        name, chrom, _, end = result[first]
        result[first] = (name, chrom, 1, end)
        name, chrom, start, _ = result[last]
        result[last] = (name, chrom, start, int(np.iinfo(np.int64).max))
    return result


def partition_carriers(
    carrier_table: Path,
    intervals: list[tuple[str, object, int, int]],
    directory: Path,
) -> tuple[dict[str, Path], dict[str, int], dict[str, tuple[int | None, int | None]]]:
    """Partition carrier rows by block while emitting no record content."""
    paths = {name: directory / f"block-{index:03d}.tsv" for index, (name, _, _, _) in enumerate(intervals)}
    counts = {name: 0 for name, _, _, _ in intervals}
    bounds: dict[str, list[int | None]] = {name: [None, None] for name, _, _, _ in intervals}
    handles = {}
    writers = {}
    try:
        with carrier_table.open("r", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"carrier table is missing columns: {sorted(missing)}")
            assert reader.fieldnames is not None
            for name, _, _, _ in intervals:
                handle = paths[name].open("w", newline="")
                handles[name] = handle
                writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, delimiter="\t", lineterminator="\n")
                writers[name] = writer
                writer.writeheader()

            for line_number, row in enumerate(reader, start=2):
                try:
                    position = int(row["POS"])
                except ValueError as exc:
                    raise ValueError(f"invalid POS on carrier-table line {line_number}") from exc
                matches = [
                    name
                    for name, chrom, start, end in intervals
                    if _chrom_equal(chrom, row["CHROM"]) and start <= position <= end
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"carrier-table line {line_number} matched {len(matches)} graph blocks; expected one"
                    )
                writers[matches[0]].writerow(row)
                counts[matches[0]] += 1
                block_bounds = bounds[matches[0]]
                block_bounds[0] = position if block_bounds[0] is None else min(block_bounds[0], position)
                block_bounds[1] = position if block_bounds[1] is None else max(block_bounds[1], position)
    finally:
        for handle in handles.values():
            handle.close()
    return paths, counts, {name: (values[0], values[1]) for name, values in bounds.items()}


def set_root_attributes(h5: h5py.File, policy: str, stats: AugmentationStats) -> None:
    """Persist the same aggregate root attributes as the public augmenter."""
    h5.attrs["rare_variant_phase_method"] = _phase_method(policy)
    h5.attrs["rare_variant_reuse_policy"] = policy
    h5.attrs["rare_variant_phase_is_inferred"] = False
    h5.attrs["rare_variant_diploid_dosage_preserved"] = True
    h5.attrs["rare_variants_added"] = stats.variants_added
    h5.attrs["rare_variant_singletons_direct"] = stats.direct_singletons
    h5.attrs["rare_variant_doubletons_added"] = stats.doubletons_added
    h5.attrs["rare_variant_existing_nodes_reused"] = stats.reused_existing_nodes
    h5.attrs["rare_variant_distinct_existing_nodes_reused"] = stats.distinct_existing_nodes_reused
    h5.attrs["rare_variant_new_nodes_reused"] = stats.reused_new_nodes
    h5.attrs["rare_variant_nodes_added"] = stats.nodes_added
    h5.attrs["rare_variant_edges_added"] = stats.edges_added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--carrier-table", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--reuse-policy", required=True, choices=REUSE_POLICIES)
    parser.add_argument("--rss-sample-seconds", type=float, default=0.1)
    args = parser.parse_args()

    policy = _validate_reuse_policy(args.reuse_policy)
    if args.output_h5.exists():
        raise FileExistsError(f"output already exists: {args.output_h5}")
    if args.result_json.exists():
        raise FileExistsError(f"result already exists: {args.result_json}")
    if args.rss_sample_seconds <= 0:
        raise ValueError("--rss-sample-seconds must be positive")

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    before = hdf5_summary(args.input_h5)
    source_intervals = block_intervals(args.input_h5)
    assignment_intervals = extend_chromosome_edges(source_intervals)
    process = psutil.Process()
    rss_samples = [process_tree_rss(process)]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(args.rss_sample_seconds):
            rss_samples.append(process_tree_rss(process))

    monitor = threading.Thread(target=sample_rss, daemon=True)
    monitor.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    total = AugmentationStats()
    block_results = []
    edge_block_extensions = []
    temporary_output: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="rare-carrier-blocks-", dir=args.output_h5.parent) as temp_dir:
            partition_started = time.perf_counter()
            partition_paths, partition_rows, partition_bounds = partition_carriers(
                args.carrier_table,
                assignment_intervals,
                Path(temp_dir),
            )
            total.block_assignment_seconds = time.perf_counter() - partition_started

            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{args.output_h5.name}.",
                suffix=".tmp",
                dir=args.output_h5.parent,
            )
            os.close(fd)
            temporary_output = Path(temporary_name)
            copy_started = time.perf_counter()
            shutil.copy2(args.input_h5, temporary_output)
            total.file_copy_seconds = time.perf_counter() - copy_started

            with h5py.File(temporary_output, "r+") as h5:
                if "iids" not in h5:
                    raise ValueError("input HDF5 is missing root-level iids")
                iid_started = time.perf_counter()
                sample_counts = {int(h5[name].attrs["n_samples"]) for name in _block_names(h5)}
                if len(sample_counts) != 1:
                    raise ValueError("LinearARG blocks have inconsistent haplotype sample counts")
                iids = _diploid_iids([_decode(value) for value in h5["iids"][:]], sample_counts.pop())
                total.iid_normalization_seconds = time.perf_counter() - iid_started

                for name, _, _, _ in source_intervals:
                    block_wall_started = time.perf_counter()
                    parse_started = time.perf_counter()
                    variants = _read_carrier_table(partition_paths[name])
                    parse_seconds = time.perf_counter() - parse_started
                    group = h5[name]
                    observed_start, observed_end = partition_bounds[name]
                    old_start = int(group.attrs.get("start", 0))
                    old_end = int(group.attrs.get("end", np.iinfo(np.int64).max))
                    new_start = old_start if observed_start is None else min(old_start, observed_start)
                    new_end = old_end if observed_end is None else max(old_end, observed_end)
                    target_name = name
                    if new_start != old_start or new_end != old_end:
                        chrom = _decode(group.attrs.get("chrom", name.split(":", 1)[0]))
                        target_name = f"{chrom}:{new_start}-{new_end}"
                        if target_name in h5:
                            raise ValueError(f"extended block name already exists: {target_name}")
                        h5.move(name, target_name)
                        group = h5[target_name]
                        group.attrs["start"] = str(new_start)
                        group.attrs["end"] = str(new_end)
                        edge_block_extensions.append(
                            {
                                "source_block": name,
                                "output_block": target_name,
                                "old_start": old_start,
                                "old_end": old_end,
                                "new_start": new_start,
                                "new_end": new_end,
                            }
                        )
                    block_stats = _augment_block(group, variants, iids, reuse_policy=policy)
                    block_stats.carrier_parse_seconds = parse_seconds
                    total.add(block_stats)
                    h5.flush()
                    block_results.append(
                        {
                            "block": target_name,
                            "source_block": name,
                            "carrier_rows": partition_rows[name],
                            "wall_seconds": time.perf_counter() - block_wall_started,
                            "stats": asdict(block_stats),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "block": target_name,
                                "variants_added": block_stats.variants_added,
                                "nodes_added": block_stats.nodes_added,
                                "edges_added": block_stats.edges_added,
                                "wall_seconds": block_results[-1]["wall_seconds"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    del variants
                    gc.collect()
                set_root_attributes(h5, policy, total)
            os.replace(temporary_output, args.output_h5)
            temporary_output = None
    finally:
        stop.set()
        monitor.join()
        rss_samples.append(process_tree_rss(process))
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)

    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started
    total.total_seconds = wall_seconds
    after = hdf5_summary(args.output_h5)
    result = {
        "reuse_policy": policy,
        "input_h5": str(args.input_h5.resolve()),
        "carrier_table": str(args.carrier_table.resolve()),
        "output_h5": str(args.output_h5.resolve()),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_process_tree_rss_bytes": max(rss_samples),
        "before": before,
        "after": after,
        "delta": {
            "bytes": int(after["bytes"]) - int(before["bytes"]),
            "nodes": int(after["nodes"]) - int(before["nodes"]),
            "edges": int(after["edges"]) - int(before["edges"]),
            "variants": int(after["variants"]) - int(before["variants"]),
        },
        "stats": asdict(total),
        "blocks": block_results,
        "edge_block_extensions": edge_block_extensions,
    }
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result_json": str(args.result_json),
                "wall_seconds": wall_seconds,
                "peak_process_tree_rss_bytes": max(rss_samples),
                "variants_added": total.variants_added,
                "nodes_added": total.nodes_added,
                "edges_added": total.edges_added,
                "bytes_added": result["delta"]["bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
