#!/usr/bin/env python3
"""Validate augmented rare-variant dosages using aggregate-only results."""

from __future__ import annotations

import argparse
import json

from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from linear_dag import read_rare_variant_carriers
from linear_dag.rare_variants import (
    _block_names,
    _capped_leaf_signatures,
    _decode,
    _diploid_iids,
    _read_matrix,
    META_COLUMNS,
    RareVariant,
)


def allele_key(chrom: object, pos: object, ref: object, alt: object) -> tuple[str, int, str, str]:
    """Return a chromosome-normalized allele key."""
    chrom_string = _decode(chrom)
    if chrom_string.lower().startswith("chr"):
        chrom_string = chrom_string[3:]
    return chrom_string, int(pos), _decode(ref), _decode(alt)


def assigned_variants(h5: h5py.File, variants: list[RareVariant]) -> dict[str, list[RareVariant]]:
    """Assign variants to blocks using preloaded aggregate interval metadata."""
    intervals = []
    for name in _block_names(h5):
        group = h5[name]
        intervals.append(
            (
                name,
                allele_key(group.attrs.get("chrom", name.split(":", 1)[0]), 0, "", "")[0],
                int(group.attrs.get("start", 0)),
                int(group.attrs.get("end", np.iinfo(np.int64).max)),
            )
        )
    result = {name: [] for name, _, _, _ in intervals}
    for variant in variants:
        key = allele_key(*variant.allele_key)
        matches = [name for name, chrom, start, end in intervals if chrom == key[0] and start <= key[1] <= end]
        if len(matches) != 1:
            raise ValueError(f"aggregate assignment failed for {len(matches)} blocks")
        result[matches[0]].append(variant)
    return result


def output_validation(path: Path, variants: list[RareVariant]) -> dict[str, object]:
    """Validate one output and return counts without exposing carrier records."""
    with h5py.File(path, "r") as h5:
        blocks = _block_names(h5)
        sample_counts = {int(h5[name].attrs["n_samples"]) for name in blocks}
        if len(sample_counts) != 1:
            raise ValueError("output blocks have inconsistent sample counts")
        n_samples = sample_counts.pop()
        raw_iids = [_decode(value) for value in h5["iids"][:]]
        iids = _diploid_iids(raw_iids, n_samples)
        iid_to_index = {iid: index for index, iid in enumerate(iids)}
        if len(iid_to_index) != len(iids):
            raise ValueError("output contains duplicate diploid IIDs")

        assignments = assigned_variants(h5, variants)
        variants_checked = 0
        variants_missing = 0
        signatures_missing = 0
        dosage_mismatches = 0
        for block, block_variants in assignments.items():
            if not block_variants:
                continue
            group = h5[block]
            expected = {allele_key(*variant.allele_key): variant for variant in block_variants}
            positions = np.asarray(group["POS"][:], dtype=np.int64)
            minimum = min(variant.pos for variant in block_variants)
            maximum = max(variant.pos for variant in block_variants)
            left = int(np.searchsorted(positions, minimum, side="left"))
            right = int(np.searchsorted(positions, maximum, side="right"))
            columns = {name: group[name][left:right] for name in META_COLUMNS}
            variant_indices = np.asarray(group["variant_indices"][left:right], dtype=np.int64)
            found_nodes = {}
            for offset, node in enumerate(variant_indices):
                key = allele_key(
                    columns["CHROM"][offset],
                    columns["POS"][offset],
                    columns["REF"][offset],
                    columns["ALT"][offset],
                )
                if key in expected:
                    found_nodes[key] = int(node)
            variants_missing += len(expected) - len(found_nodes)

            A = _read_matrix(group)
            n_individual_nodes = int(group.attrs.get("n_individuals", 0))
            core_n = A.shape[0] - n_individual_nodes
            signatures = _capped_leaf_signatures(A[:core_n, :core_n].tocsc(), found_nodes.values(), n_samples)
            for key, variant in expected.items():
                if key not in found_nodes:
                    continue
                node = found_nodes[key]
                signature = signatures.get(node)
                if signature is None:
                    signatures_missing += 1
                    continue
                observed = Counter(haplotype // 2 for haplotype in signature)
                expected_dosage = Counter({iid_to_index[call.iid]: call.dosage for call in variant.calls})
                dosage_mismatches += observed != expected_dosage
                variants_checked += 1

        return {
            "path": str(path.resolve()),
            "reuse_policy": str(h5.attrs["rare_variant_reuse_policy"]),
            "variants_expected": len(variants),
            "variants_checked": variants_checked,
            "variants_missing": variants_missing,
            "signatures_missing": signatures_missing,
            "dosage_mismatches": dosage_mismatches,
        }


def metadata_mismatches(reference: Path, candidate: Path, chunk_size: int = 250_000) -> int:
    """Count chunks with unequal variant metadata without displaying values."""
    mismatch_chunks = 0
    datasets = (*META_COLUMNS, "flip", "allele_counts")
    with h5py.File(reference, "r") as left, h5py.File(candidate, "r") as right:
        if _block_names(left) != _block_names(right):
            raise ValueError("output block names differ")
        for block in _block_names(left):
            for name in datasets:
                left_dataset = left[block][name]
                right_dataset = right[block][name]
                if left_dataset.shape != right_dataset.shape:
                    mismatch_chunks += 1
                    continue
                for start in range(0, len(left_dataset), chunk_size):
                    stop = min(start + chunk_size, len(left_dataset))
                    mismatch_chunks += not np.array_equal(left_dataset[start:stop], right_dataset[start:stop])
    return mismatch_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-table", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, action="append", type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args()
    if args.result_json.exists():
        raise FileExistsError(f"result already exists: {args.result_json}")

    variants = read_rare_variant_carriers(args.carrier_table)
    outputs = [output_validation(path, variants) for path in args.output_h5]
    reference = args.output_h5[0]
    comparisons = [
        {
            "reference": str(reference.resolve()),
            "candidate": str(candidate.resolve()),
            "metadata_mismatch_chunks": metadata_mismatches(reference, candidate),
        }
        for candidate in args.output_h5[1:]
    ]
    result = {"outputs": outputs, "metadata_comparisons": comparisons}
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result_json": str(args.result_json),
                "outputs": len(outputs),
                "dosage_mismatches": sum(int(output["dosage_mismatches"]) for output in outputs),
                "metadata_mismatch_chunks": sum(
                    int(comparison["metadata_mismatch_chunks"]) for comparison in comparisons
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
