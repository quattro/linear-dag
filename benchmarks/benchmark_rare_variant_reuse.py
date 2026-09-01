#!/usr/bin/env python3
"""Benchmark one rare-variant node-reuse policy with aggregate-only output."""

from __future__ import annotations

import argparse
import json
import threading
import time

from dataclasses import asdict
from pathlib import Path

import h5py
import psutil

from linear_dag import augment_rare_variants_file, REUSE_POLICIES


def process_tree_rss(process: psutil.Process) -> int:
    """Return resident bytes for the benchmark process and its children."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def hdf5_summary(path: Path) -> dict[str, object]:
    """Read aggregate graph metadata without inspecting individual records."""
    with h5py.File(path, "r") as h5:
        blocks = [name for name in h5 if name != "iids" and isinstance(h5[name], h5py.Group)]
        return {
            "bytes": path.stat().st_size,
            "blocks": len(blocks),
            "nodes": sum(int(h5[name].attrs["n"]) for name in blocks),
            "edges": sum(int(h5[name].attrs["n_entries"]) for name in blocks),
            "variants": sum(int(h5[name].attrs["n_variants"]) for name in blocks),
            "reuse_policy": str(h5.attrs["rare_variant_reuse_policy"]),
        }


def fraction(numerator: int, denominator: int) -> float | None:
    """Return an aggregate fraction, or ``None`` when its denominator is zero."""
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--carrier-table", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--reuse-policy", required=True, choices=REUSE_POLICIES)
    parser.add_argument("--label")
    parser.add_argument("--rss-sample-seconds", type=float, default=0.05)
    args = parser.parse_args()

    if args.result_json.exists():
        raise FileExistsError(f"result already exists: {args.result_json}")
    if args.rss_sample_seconds <= 0:
        raise ValueError("--rss-sample-seconds must be positive")

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
    try:
        stats = augment_rare_variants_file(
            args.input_h5,
            args.carrier_table,
            args.output_h5,
            reuse_policy=args.reuse_policy,
        )
    finally:
        stop.set()
        monitor.join()
        rss_samples.append(process_tree_rss(process))
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started

    if stats.doubletons_added != (stats.reused_existing_nodes + stats.reused_new_nodes + stats.nodes_added):
        raise AssertionError("doubleton accounting invariant failed")

    result = {
        "label": args.label,
        "reuse_policy": args.reuse_policy,
        "input_h5": str(args.input_h5.resolve()),
        "carrier_table": str(args.carrier_table.resolve()),
        "output_h5": str(args.output_h5.resolve()),
        "input_h5_bytes": args.input_h5.stat().st_size,
        "carrier_table_bytes": args.carrier_table.stat().st_size,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "peak_process_tree_rss_bytes": max(rss_samples),
        "stats": asdict(stats),
        "fractions": {
            "existing_reuse": fraction(stats.reused_existing_nodes, stats.doubletons_added),
            "distinct_existing_reuse": fraction(stats.distinct_existing_nodes_reused, stats.doubletons_added),
            "within_run_reuse": fraction(stats.reused_new_nodes, stats.doubletons_added),
            "new_node": fraction(stats.nodes_added, stats.doubletons_added),
        },
        "output": hdf5_summary(args.output_h5),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result_json": str(args.result_json),
                "reuse_policy": args.reuse_policy,
                "wall_seconds": wall_seconds,
                "peak_process_tree_rss_bytes": result["peak_process_tree_rss_bytes"],
                "doubletons_added": stats.doubletons_added,
                "reused_existing_nodes": stats.reused_existing_nodes,
                "nodes_added": stats.nodes_added,
                "output_bytes": result["output"]["bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
