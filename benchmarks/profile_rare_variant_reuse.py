#!/usr/bin/env python3
"""Profile existing and within-run doubleton reuse without writing an augmented graph."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time

from pathlib import Path

import h5py
import numpy as np
import psutil

from linear_dag.rare_variants import (
    _block_names,
    _capped_leaf_signatures,
    _choose_signature,
    _decode,
    _diploid_iids,
    _phase_candidates,
    _read_matrix,
    CarrierCall,
    RareVariant,
)


def read_doubletons(path: Path) -> tuple[list[RareVariant], int, int]:
    """Stream a position-sorted carrier table and retain only doubletons."""
    doubletons = []
    singletons = 0
    carrier_rows = 0
    previous_key = None
    calls: list[CarrierCall] = []

    def finish(key: tuple[str, int, str, str, str] | None) -> None:
        nonlocal singletons
        if key is None:
            return
        allele_count = sum(call.dosage for call in calls)
        if allele_count == 1:
            singletons += 1
        elif allele_count == 2:
            chrom, pos, variant_id, ref, alt = key
            doubletons.append(RareVariant(chrom, pos, variant_id, ref, alt, tuple(calls)))
        else:
            raise ValueError("streamed allele count was not one or two")

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            carrier_rows += 1
            key = (row["CHROM"], int(row["POS"]), row["ID"] or ".", row["REF"], row["ALT"])
            if previous_key is not None and key != previous_key:
                if (key[0], key[1], key[3], key[4]) < (
                    previous_key[0],
                    previous_key[1],
                    previous_key[3],
                    previous_key[4],
                ):
                    raise ValueError("carrier table is not globally sorted by allele key")
                finish(previous_key)
                calls = []
            calls.append(CarrierCall(row["IID"], int(row["DOSAGE"])))
            previous_key = key
        finish(previous_key)
    return doubletons, singletons, carrier_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--carrier-table", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    if args.result_json.exists():
        raise FileExistsError(f"result already exists: {args.result_json}")

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.perf_counter()
    parse_started = time.perf_counter()
    doubletons, singletons, carrier_rows = read_doubletons(args.carrier_table)
    parse_seconds = time.perf_counter() - parse_started
    peak_rss = max(peak_rss, process.memory_info().rss)

    totals = {
        "candidate_nodes_scanned": 0,
        "existing_signatures_available": 0,
        "reused_existing_variants": 0,
        "distinct_existing_signatures_used": 0,
        "reused_within_run_variants": 0,
        "new_nodes_needed": 0,
        "batch_only_reused_within_run_variants": 0,
        "batch_only_new_nodes_needed": 0,
    }
    matrix_load_seconds = 0.0
    scan_seconds = 0.0
    selection_seconds = 0.0
    with h5py.File(args.input_h5, "r") as h5:
        blocks = sorted(_block_names(h5), key=lambda name: int(h5[name].attrs.get("start", 0)))
        n_samples = int(h5[blocks[0]].attrs["n_samples"])
        iids = _diploid_iids([_decode(value) for value in h5["iids"][:]], n_samples)
        iid_to_index = {iid: index for index, iid in enumerate(iids)}
        for block in blocks:
            group = h5[block]
            start = int(group.attrs.get("start", 0))
            end = int(group.attrs.get("end", np.iinfo(np.int64).max))
            block_doubletons = [variant for variant in doubletons if start <= variant.pos <= end]
            load_started = time.perf_counter()
            A = _read_matrix(group)
            matrix_load_seconds += time.perf_counter() - load_started
            scan_started = time.perf_counter()
            old_counts = np.asarray(group["allele_counts"][:], dtype=np.int64)
            old_indices = np.asarray(group["variant_indices"][:], dtype=np.int64)
            candidate_nodes = set(map(int, old_indices[np.flatnonzero(old_counts == 2)]))
            core_n = A.shape[0] - int(group.attrs.get("n_individuals", 0))
            signatures = _capped_leaf_signatures(A[:core_n, :core_n].tocsc(), candidate_nodes, n_samples)
            existing = {signature: node for node, signature in signatures.items() if len(signature) == 2}
            scan_seconds += time.perf_counter() - scan_started
            totals["candidate_nodes_scanned"] += len(candidate_nodes)
            totals["existing_signatures_available"] += len(existing)

            selected = {}
            batch_selected = {}
            used_existing = set()
            selection_started = time.perf_counter()
            for variant in block_doubletons:
                candidates = _phase_candidates(variant, iid_to_index)
                signature, source = _choose_signature(candidates, existing, selected)
                if source == "existing":
                    totals["reused_existing_variants"] += 1
                    used_existing.add(signature)
                elif source == "selected":
                    totals["reused_within_run_variants"] += 1
                else:
                    totals["new_nodes_needed"] += 1
                    selected[signature] = -1
                batch_signature, batch_source = _choose_signature(candidates, {}, batch_selected)
                if batch_source == "selected":
                    totals["batch_only_reused_within_run_variants"] += 1
                else:
                    totals["batch_only_new_nodes_needed"] += 1
                    batch_selected[batch_signature] = -1
            selection_seconds += time.perf_counter() - selection_started
            totals["distinct_existing_signatures_used"] += len(used_existing)
            del A, signatures, existing, selected, batch_selected
            gc.collect()
            peak_rss = max(peak_rss, process.memory_info().rss)

    result = {
        "carrier_rows": carrier_rows,
        "singletons": singletons,
        "doubletons": len(doubletons),
        **totals,
        "parse_seconds": parse_seconds,
        "matrix_load_seconds": matrix_load_seconds,
        "existing_scan_seconds": scan_seconds,
        "selection_seconds": selection_seconds,
        "total_seconds": time.perf_counter() - started,
        "observed_peak_rss_bytes": peak_rss,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
