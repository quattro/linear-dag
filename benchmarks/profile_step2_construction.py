#!/usr/bin/env python3
"""Profile aggregate Step 2 graph-construction phases on existing artifacts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import threading
import time

from pathlib import Path

import h5py
import numpy as np
import psutil

from linear_dag.core.brick_graph import BrickGraph, read_graph_from_disk
from linear_dag.core.recombination import Recombination


def array_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


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
    parser.add_argument("--root", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    partition = args.partition
    process = psutil.Process()
    started = time.perf_counter()
    cpu_started = time.process_time()
    samples: list[tuple[float, int]] = []
    marks: list[dict[str, float | int | str]] = []
    stop = threading.Event()

    def mark(name: str) -> None:
        marks.append(
            {
                "name": name,
                "elapsed_s": time.perf_counter() - started,
                "rss_bytes": process_tree_rss(process),
            }
        )

    def sample() -> None:
        while not stop.wait(0.01):
            samples.append((time.perf_counter() - started, process_tree_rss(process)))

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    aggregate: dict[str, int | str] = {}
    try:
        genotype_path = root / "genotype_matrices" / f"{partition}.h5"
        graph_prefix = root / "forward_backward_graphs" / partition
        with h5py.File(genotype_path, "r") as handle:
            _, num_variants = handle["shape"][:]
        mark("shape_loaded")

        forward_graph = read_graph_from_disk(f"{graph_prefix}_forward_graph.h5")
        aggregate["forward_nodes"] = int(forward_graph.number_of_nodes)
        aggregate["forward_edges"] = int(forward_graph.number_of_edges)
        mark("forward_loaded")
        backward_graph = read_graph_from_disk(f"{graph_prefix}_backward_graph.h5")
        aggregate["backward_nodes"] = int(backward_graph.number_of_nodes)
        aggregate["backward_edges"] = int(backward_graph.number_of_edges)
        mark("backward_loaded")
        sample_indices = np.atleast_1d(np.loadtxt(f"{graph_prefix}_sample_indices.txt")).astype(np.int64)
        mark("sample_indices_loaded")

        if hasattr(BrickGraph, "combine_graphs_packed"):
            packed_edges, variant_indices = BrickGraph.combine_graphs_packed(
                forward_graph, backward_graph, int(num_variants)
            )
            aggregate["construction"] = "packed"
            aggregate["brick_nodes_before_recombination"] = int(packed_edges.number_of_nodes)
            aggregate["brick_edges_before_recombination"] = int(packed_edges.number_of_edges)
            aggregate["packed_allocated_bytes"] = int(packed_edges.allocated_nbytes)
            mark("graphs_combined_packed")
            del forward_graph, backward_graph
            gc.collect()
            mark("source_graphs_released")
            recombination = Recombination.from_packed_edges(packed_edges)
            del packed_edges
            gc.collect()
            mark("recombination_initialized_and_packed_released")
        else:
            brick_graph, variant_indices = BrickGraph.combine_graphs(
                forward_graph, backward_graph, int(num_variants)
            )
            aggregate["construction"] = "intermediate_graph"
            aggregate["brick_nodes_before_recombination"] = int(brick_graph.number_of_nodes)
            aggregate["brick_edges_before_recombination"] = int(brick_graph.number_of_edges)
            mark("graphs_combined")
            del forward_graph, backward_graph
            gc.collect()
            mark("source_graphs_released")
            recombination = Recombination.from_graph(brick_graph)
            del brick_graph
            gc.collect()
            mark("recombination_initialized_and_brick_graph_released")

        heap, priorities = recombination.get_heap
        aggregate["heap_nodes_total"] = int(len(heap))
        aggregate["heap_priorities_nonzero"] = int(np.count_nonzero(np.asarray(priorities)))
        recombination.find_recombinations()
        mark("recombinations_found")
        indptr, indices, data, num_nodes = recombination.to_csc_arrays()
        mark("csc_arrays_created")
        aggregate.update(
            {
                "num_samples": int(len(sample_indices)),
                "num_variants": int(len(variant_indices)),
                "num_nodes": int(num_nodes),
                "num_edges": int(len(data)),
                "csc_bytes": int(indptr.nbytes + indices.nbytes + data.nbytes),
                "indptr_sha256": array_digest(np.asarray(indptr)),
                "indices_sha256": array_digest(np.asarray(indices)),
                "data_sha256": array_digest(np.asarray(data)),
                "variant_indices_sha256": array_digest(np.asarray(variant_indices)),
                "sample_indices_sha256": array_digest(np.asarray(sample_indices)),
            }
        )
        del recombination
        gc.collect()
        mark("recombination_released")
    finally:
        stop.set()
        monitor.join()

    samples.append((time.perf_counter() - started, process_tree_rss(process)))
    for index, current in enumerate(marks):
        previous_t = 0.0 if index == 0 else float(marks[index - 1]["elapsed_s"])
        current_t = float(current["elapsed_s"])
        interval = [rss for elapsed, rss in samples if previous_t <= elapsed <= current_t]
        current["peak_rss_since_previous_mark_bytes"] = max(interval, default=int(current["rss_bytes"]))

    result = {
        "root": str(root),
        "partition": partition,
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_process_tree_rss_bytes": max(rss for _, rss in samples),
        "marks": marks,
        "aggregate": aggregate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_keys = ("wall_seconds", "peak_process_tree_rss_bytes")
    print(json.dumps({"output": str(output), **{key: result[key] for key in summary_keys}}, sort_keys=True))


if __name__ == "__main__":
    main()
