"""Benchmark default-one edge-weight compression on HDF5 LinearARG blocks."""

import argparse
import json
import time

from pathlib import Path

import h5py
import numpy as np

from linear_dag.core.lineararg import LinearARG


def _core_nbytes(linarg: LinearARG) -> int:
    adjacency = linarg.A
    if hasattr(adjacency, "nbytes"):
        adjacency_nbytes = adjacency.nbytes
    else:
        adjacency_nbytes = adjacency.indptr.nbytes + adjacency.indices.nbytes + adjacency.data.nbytes
    return adjacency_nbytes + sum(
        array.nbytes
        for array in (linarg.variant_indices, linarg.flip, linarg.nonunique_indices)
        if array is not None
    )


def _median_runtime(default_fn, compressed_fn, argument, repeats: int) -> tuple[float, float]:
    timings = {"default": [], "compressed": []}
    default_fn(argument)
    compressed_fn(argument)
    for iteration in range(2 * repeats):
        key = "default" if iteration % 2 == 0 else "compressed"
        start = time.perf_counter()
        (default_fn if key == "default" else compressed_fn)(argument)
        timings[key].append(time.perf_counter() - start)
    return float(np.median(timings["default"])), float(np.median(timings["compressed"]))


def _select_blocks(path: Path, count: int) -> list[str]:
    with h5py.File(path, "r") as file:
        blocks = [(name, int(obj.attrs["n_entries"])) for name, obj in file.items() if isinstance(obj, h5py.Group)]
    blocks.sort(key=lambda item: item[1])
    indices = np.linspace(0, len(blocks) - 1, min(count, len(blocks))).round().astype(int)
    return [blocks[index][0] for index in np.unique(indices)]


def summarize_file(path: Path) -> dict:
    edges = ones = minus_ones = other = core_bytes = 0
    groups = 0
    with h5py.File(path, "r") as file:
        for group in (obj for obj in file.values() if isinstance(obj, h5py.Group)):
            if "data" not in group:
                continue
            data = group["data"][:]
            groups += 1
            edges += data.size
            ones += int(np.count_nonzero(data == 1))
            minus_ones += int(np.count_nonzero(data == -1))
            other += int(np.count_nonzero((data != 1) & (data != -1)))
            core_bytes += sum(
                dataset.size * dataset.dtype.itemsize
                for name, dataset in group.items()
                if isinstance(dataset, h5py.Dataset)
                and name in {"data", "indices", "indptr", "nonunique_indices", "variant_indices", "flip"}
            )
    nonunit = edges - ones
    dense_weight_bytes = 4 * edges
    compressed_weight_bytes = 8 * nonunit
    compressed_core_bytes = core_bytes - dense_weight_bytes + compressed_weight_bytes
    return {
        "path": str(path),
        "groups": groups,
        "edges": edges,
        "fraction_one": ones / edges,
        "fraction_minus_one": minus_ones / edges,
        "fraction_other": other / edges,
        "dense_weight_bytes": dense_weight_bytes,
        "compressed_weight_bytes": compressed_weight_bytes,
        "weight_memory_ratio": compressed_weight_bytes / dense_weight_bytes,
        "core_bytes": core_bytes,
        "compressed_core_bytes": compressed_core_bytes,
        "core_memory_ratio": compressed_core_bytes / core_bytes,
    }


def benchmark_block(path: Path, block: str, repeats: int) -> dict:
    default = LinearARG.read(path, block=block)
    compressed = LinearARG.read(path, block=block, compress_edge_weights=True)
    rng = np.random.default_rng(20260713)
    cases = {
        "matvec": (default._matvec, compressed._matvec, rng.standard_normal(default.shape[1])),
        "rmatvec": (default._rmatvec, compressed._rmatvec, rng.standard_normal(default.shape[0])),
        "matmat_8": (
            default._matmat,
            compressed._matmat,
            rng.standard_normal((default.shape[1], 8)).astype(np.float32),
        ),
        "rmatmat_8": (
            default._rmatmat,
            compressed._rmatmat,
            rng.standard_normal((default.shape[0], 8)).astype(np.float32),
        ),
        "matmat_32": (
            default._matmat,
            compressed._matmat,
            rng.standard_normal((default.shape[1], 32)).astype(np.float32),
        ),
        "rmatmat_32": (
            default._rmatmat,
            compressed._rmatmat,
            rng.standard_normal((default.shape[0], 32)).astype(np.float32),
        ),
    }
    timings = {}
    for name, (default_fn, compressed_fn, argument) in cases.items():
        default_result = default_fn(argument)
        compressed_result = compressed_fn(argument)
        default_seconds, compressed_seconds = _median_runtime(default_fn, compressed_fn, argument, repeats)
        timings[name] = {
            "default_seconds": default_seconds,
            "compressed_seconds": compressed_seconds,
            "runtime_ratio": compressed_seconds / default_seconds,
            "max_absolute_error": float(np.max(np.abs(default_result - compressed_result))),
        }
    default_bytes = _core_nbytes(default)
    compressed_bytes = _core_nbytes(compressed)
    return {
        "path": str(path),
        "block": block,
        "nodes": int(default.A.shape[0]),
        "edges": int(default.A.nnz),
        "variants": int(default.shape[1]),
        "default_core_bytes": default_bytes,
        "compressed_core_bytes": compressed_bytes,
        "core_memory_ratio": compressed_bytes / default_bytes,
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--blocks-per-file", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = {
        "files": [summarize_file(path) for path in args.paths],
        "blocks": [
            benchmark_block(path, block, args.repeats)
            for path in args.paths
            for block in _select_blocks(path, args.blocks_per_file)
        ],
    }
    text = json.dumps(result, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
