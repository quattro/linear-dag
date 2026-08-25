#!/usr/bin/env python3
"""Profile aggregate Step 1 disk-mode graph allocation and output equivalence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import threading
import time

from pathlib import Path

import h5py
import numpy as np
import psutil

from linear_dag.core.brick_graph import BrickGraph

EDGE_RECORD_BYTES = 56


def array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_artifact_summary(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        datasets = {
            name: {
                "dtype": str(handle[name].dtype),
                "shape": list(handle[name].shape),
                "sha256": array_digest(handle[name][:]),
            }
            for name in sorted(handle.keys())
        }
        attributes = {name: int(value) for name, value in sorted(handle.attrs.items())}
    return {
        "attributes": attributes,
        "datasets": datasets,
        "file_bytes": path.stat().st_size,
        "file_sha256": file_digest(path),
    }


def process_tree_rss(process: psutil.Process) -> int:
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("real", "synthetic"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--genotype")
    parser.add_argument("--capacity", type=int, default=1_000_000)
    args = parser.parse_args()

    if args.mode == "real" and args.genotype is None:
        parser.error("--genotype is required in real mode")
    if args.capacity < 2:
        parser.error("--capacity must be at least 2")

    process = psutil.Process()
    samples = [process_tree_rss(process)]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.005):
            samples.append(process_tree_rss(process))

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    cpu_started = time.process_time()
    aggregate: dict[str, object]
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            if args.mode == "real":
                prefix = temporary_path / "forward_backward"
                sample_indices = BrickGraph.forward_backward_from_hdf5(
                    args.genotype,
                    add_samples=True,
                    out=str(prefix),
                )
                aggregate = {
                    "sample_count": int(len(sample_indices)),
                    "sample_indices_sha256": array_digest(np.asarray(sample_indices)),
                    "forward": graph_artifact_summary(Path(f"{prefix}_forward_graph.h5")),
                    "backward": graph_artifact_summary(Path(f"{prefix}_backward_graph.h5")),
                }
            else:
                graph_path = temporary_path / "synthetic.h5"
                graph = BrickGraph(1, args.capacity - 1, save_to_disk=True, out=str(graph_path))
                if hasattr(graph, "_native_graph_stats"):
                    native_edges, native_capacity = graph._native_graph_stats
                else:
                    # Integration baseline 52e455b allocates one slot per node.
                    native_edges, native_capacity = 0, args.capacity
                aggregate = {
                    "requested_node_capacity": args.capacity,
                    "native_edge_count": int(native_edges),
                    "native_edge_capacity": int(native_capacity),
                    "native_edge_arena_bytes": int(native_capacity * EDGE_RECORD_BYTES),
                }
                del graph
                gc.collect()
                aggregate["artifact"] = graph_artifact_summary(graph_path)
    finally:
        stop.set()
        monitor.join()
    samples.append(process_tree_rss(process))

    result = {
        "mode": args.mode,
        "label": args.label,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "starting_rss_bytes": samples[0],
        "peak_process_tree_rss_bytes": max(samples),
        "peak_rss_increase_bytes": max(samples) - samples[0],
        "aggregate": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "mode": args.mode,
                "wall_seconds": result["wall_seconds"],
                "cpu_seconds": result["cpu_seconds"],
                "peak_process_tree_rss_bytes": result["peak_process_tree_rss_bytes"],
                "peak_rss_increase_bytes": result["peak_rss_increase_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
