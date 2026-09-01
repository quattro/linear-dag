"""Add unphased singleton/doubleton dosages to a LinearARG HDF5 file.

The input is a tab-separated, sparse carrier table with columns
CHROM, POS, ID, REF, ALT, IID, and DOSAGE.  There is one row per non-reference
individual/variant call.  DOSAGE must be 1 or 2, and the total dosage for a
variant must be one or two.

Heterozygous calls are pseudo-phased to minimize incremental graph storage.
Singletons point directly to one deterministic haplotype sample node and add
no graph nodes or edges. For doubletons, an existing rare-variant node is
preferred, followed by a carrier pattern selected earlier in the same run.
Remaining ties are deterministic. The result preserves diploid dosage, but
the chosen rare-variant phase is not a biological inference. Input blocks must
contain two adjacent haplotype rows for every diploid IID.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile
import time

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import cast, Iterable, Literal, Sequence

import h5py
import numpy as np

from scipy.sparse import csc_matrix

try:  # Register Blosc filters when the source LinearARG uses them.
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


REQUIRED_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT", "IID", "DOSAGE")
META_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT")
ReusePolicy = Literal["existing_then_batch", "batch_only", "none"]
REUSE_POLICIES: tuple[ReusePolicy, ...] = ("existing_then_batch", "batch_only", "none")


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _normalize_chromosome(value: object) -> str:
    chrom = _decode(value)
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return chrom.upper()


def _allele_key(chrom: object, pos: int, ref: object, alt: object) -> tuple[str, int, str, str]:
    return _normalize_chromosome(chrom), int(pos), _decode(ref), _decode(alt)


@dataclass(frozen=True)
class CarrierCall:
    """One non-reference diploid call from a sparse carrier table.

    !!! Example

        ```python
        call = CarrierCall(iid="sample-1", dosage=1)
        ```
    """

    iid: str
    dosage: int


@dataclass(frozen=True)
class RareVariant:
    """Metadata and carrier calls for one singleton or doubleton.

    !!! Example

        ```python
        variant = RareVariant("1", 100, "rs1", "A", "C", (CarrierCall("sample-1", 1),))
        ```
    """

    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    calls: tuple[CarrierCall, ...]

    @property
    def allele_count(self) -> int:
        return sum(call.dosage for call in self.calls)

    @property
    def allele_key(self) -> tuple[str, int, str, str]:
        return _allele_key(self.chrom, self.pos, self.ref, self.alt)


@dataclass
class AugmentationStats:
    """Summary of graph reuse and additions during augmentation.

    !!! Example

        ```python
        stats = AugmentationStats(variants_added=2, nodes_added=1, edges_added=1)
        ```
    """

    variants_added: int = 0
    direct_singletons: int = 0
    doubletons_added: int = 0
    reused_existing_nodes: int = 0
    distinct_existing_nodes_reused: int = 0
    reused_new_nodes: int = 0
    nodes_added: int = 0
    edges_added: int = 0
    existing_candidate_nodes_scanned: int = 0
    existing_signatures_available: int = 0
    carrier_parse_seconds: float = 0.0
    file_copy_seconds: float = 0.0
    file_repack_seconds: float = 0.0
    iid_normalization_seconds: float = 0.0
    block_assignment_seconds: float = 0.0
    matrix_load_seconds: float = 0.0
    existing_scan_seconds: float = 0.0
    candidate_selection_seconds: float = 0.0
    graph_expansion_seconds: float = 0.0
    record_build_seconds: float = 0.0
    dataset_rewrite_seconds: float = 0.0
    total_seconds: float = 0.0

    def add(self, other: "AugmentationStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def _validate_reuse_policy(reuse_policy: str) -> ReusePolicy:
    if reuse_policy not in REUSE_POLICIES:
        raise ValueError(f"unknown reuse policy {reuse_policy!r}; expected one of {REUSE_POLICIES}")
    return cast(ReusePolicy, reuse_policy)


def _phase_method(reuse_policy: ReusePolicy) -> str:
    if reuse_policy == "existing_then_batch":
        return "direct_singletons_greedy_doubletons"
    if reuse_policy == "batch_only":
        return "direct_singletons_batch_reuse_doubletons"
    return "direct_singletons_independent_doubletons"


def _diploid_iids(iids: Sequence[str], n_samples: int) -> list[str]:
    """Normalize individual- or haplotype-aligned IDs to one IID per individual."""
    if n_samples % 2:
        raise ValueError(f"LinearARG has odd haplotype sample count {n_samples}; expected a diploid axis")
    n_individuals = n_samples // 2
    if len(iids) == n_individuals:
        return list(iids)
    if len(iids) == n_samples:
        if any(iids[index] != iids[index + 1] for index in range(0, n_samples, 2)):
            raise ValueError(
                "haplotype-aligned LinearARG IIDs must repeat in adjacent pairs to recover diploid individuals"
            )
        return list(iids[0::2])
    raise ValueError(
        f"LinearARG has {n_samples} haplotypes but {len(iids)} IIDs; expected "
        f"{n_individuals} individual IDs or {n_samples} adjacent-paired haplotype IDs"
    )

def _read_carrier_table(path: Path) -> list[RareVariant]:
    """Read and validate a sparse singleton/doubleton carrier table."""
    grouped: dict[tuple[str, int, str, str], dict[str, object]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"carrier table is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                pos = int(row["POS"])
                dosage = int(row["DOSAGE"])
            except ValueError as exc:
                raise ValueError(f"invalid POS or DOSAGE on line {line_number}") from exc
            if pos < 1 or dosage not in (1, 2):
                raise ValueError(f"invalid POS or DOSAGE on line {line_number}")
            key = _allele_key(row["CHROM"], pos, row["REF"], row["ALT"])
            item = grouped.setdefault(
                key,
                {"chrom": row["CHROM"], "id": row["ID"] or ".", "calls": {}},
            )
            if item["id"] != (row["ID"] or "."):
                raise ValueError(f"conflicting IDs for allele {key}")
            calls = item["calls"]
            assert isinstance(calls, dict)
            if row["IID"] in calls:
                raise ValueError(f"duplicate IID {row['IID']!r} for allele {key}")
            calls[row["IID"]] = dosage

    result = []
    for (_, pos, ref, alt), item in grouped.items():
        calls_dict = item["calls"]
        assert isinstance(calls_dict, dict)
        calls = tuple(CarrierCall(iid, dosage) for iid, dosage in sorted(calls_dict.items()))
        variant = RareVariant(str(item["chrom"]), pos, str(item["id"]), ref, alt, calls)
        if variant.allele_count not in (1, 2):
            raise ValueError(f"allele {variant.allele_key} has allele count {variant.allele_count}; expected 1 or 2")
        result.append(variant)
    return sorted(result, key=lambda v: (*v.allele_key, v.variant_id))


def _block_names(h5: h5py.File) -> list[str]:
    return [name for name in h5 if name != "iids" and isinstance(h5[name], h5py.Group)]


def _chrom_equal(left: object, right: str) -> bool:
    return _normalize_chromosome(left) == _normalize_chromosome(right)


def _assign_variants_to_blocks(h5: h5py.File, variants: Sequence[RareVariant]) -> dict[str, list[RareVariant]]:
    """Assign each variant to exactly one HDF5 genomic block."""
    blocks = _block_names(h5)
    if not blocks:
        raise ValueError("input HDF5 has no LinearARG block groups")
    block_intervals = []
    for name in blocks:
        group = h5[name]
        block_intervals.append(
            (
                name,
                group.attrs.get("chrom", name.split(":", 1)[0]),
                int(group.attrs.get("start", 0)),
                int(group.attrs.get("end", np.iinfo(np.int64).max)),
            )
        )
    assigned: dict[str, list[RareVariant]] = defaultdict(list)
    for variant in variants:
        matches = []
        for name, chrom, start, end in block_intervals:
            if _chrom_equal(chrom, variant.chrom) and start <= variant.pos <= end:
                matches.append(name)
        if len(matches) != 1:
            raise ValueError(
                f"allele {variant.allele_key} matched {len(matches)} blocks ({matches}); expected exactly one"
            )
        assigned[matches[0]].append(variant)
    return assigned


def _read_matrix(group: h5py.Group) -> csc_matrix:
    n = int(group.attrs["n"])
    return csc_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=(n, n))


def _sample_indices(n_nodes: int, n_samples: int, n_individual_nodes: int) -> np.ndarray:
    stop = n_nodes - n_individual_nodes
    return np.arange(stop - 1, stop - n_samples - 1, -1, dtype=np.int64)


def _capped_leaf_signatures(
    A: csc_matrix, nodes: Iterable[int], n_samples: int, cap: int = 2
) -> dict[int, tuple[int, ...]]:
    """Return descendant sample signatures for nodes having at most ``cap`` leaves."""
    n = A.shape[0]
    first_sample = n - n_samples
    memo: dict[int, tuple[int, ...] | None] = {}

    def visit(node: int) -> tuple[int, ...] | None:
        if node in memo:
            return memo[node]
        children = A.indices[A.indptr[node] : A.indptr[node + 1]]
        if len(children) == 0:
            value = (n - node - 1,) if node >= first_sample else None
            memo[node] = value
            return value
        leaves: set[int] = set()
        for child in children:
            child_value = visit(int(child))
            if child_value is None:
                memo[node] = None
                return None
            leaves.update(child_value)
            if len(leaves) > cap:
                memo[node] = None
                return None
        value = tuple(sorted(leaves))
        memo[node] = value
        return value

    result = {}
    for node in set(map(int, nodes)):
        signature = visit(node)
        if signature:
            result[node] = signature
    return result


def _phase_candidates(variant: RareVariant, iid_to_index: dict[str, int]) -> list[tuple[int, ...]]:
    fixed: list[int] = []
    choices: list[tuple[int, int]] = []
    for call in variant.calls:
        if call.iid not in iid_to_index:
            raise ValueError(f"IID {call.iid!r} for allele {variant.allele_key} is absent from LinearARG")
        individual = iid_to_index[call.iid]
        haps = (2 * individual, 2 * individual + 1)
        if call.dosage == 2:
            fixed.extend(haps)
        else:
            choices.append(haps)
    candidates = {tuple(sorted((*fixed, *selected))) for selected in product(*choices)}
    return sorted(candidates)


def _choose_signature(
    candidates: Sequence[tuple[int, ...]],
    existing_nodes: dict[tuple[int, ...], int],
    selected_nodes: dict[tuple[int, ...], int],
) -> tuple[tuple[int, ...], str]:
    existing = [signature for signature in candidates if signature in existing_nodes]
    if existing:
        return min(existing), "existing"
    selected = [signature for signature in candidates if signature in selected_nodes]
    if selected:
        return min(selected), "selected"
    # Every novel candidate costs one node and len(signature) direct edges.
    return min(candidates, key=lambda signature: (len(signature), signature)), "new"


def _expand_graph(
    A: csc_matrix,
    signatures: Sequence[tuple[int, ...]],
    n_samples: int,
    n_individual_nodes: int,
) -> tuple[csc_matrix, np.ndarray, list[int]]:
    """Insert mutation nodes immediately before the trailing sample nodes."""
    if not signatures:
        return A, np.arange(A.shape[0], dtype=np.int64), []
    # The previous COO round-trip both sorted column indices and combined any
    # duplicates. Preserve that behavior without materializing Python lists
    # for every graph edge.
    A.sum_duplicates()
    A.sort_indices()
    old_n = A.shape[0]
    insert_at = old_n - n_individual_nodes - n_samples
    count = len(signatures)
    mapping = np.arange(old_n, dtype=np.int32)
    mapping[insert_at:] += count
    new_nodes = list(range(insert_at, insert_at + count))
    old_samples = _sample_indices(old_n, n_samples, n_individual_nodes)
    signature_lengths = np.fromiter((len(signature) for signature in signatures), dtype=np.int32, count=count)
    edge_count = int(signature_lengths.sum())
    edge_insert = int(A.indptr[insert_at])

    indices = np.empty(A.nnz + edge_count, dtype=np.int32)
    data = np.empty(A.nnz + edge_count, dtype=np.int32)
    indices[:edge_insert] = mapping[A.indices[:edge_insert]]
    data[:edge_insert] = A.data[:edge_insert]
    cursor = edge_insert
    for signature in signatures:
        mapped_samples = mapping[old_samples[np.asarray(signature, dtype=np.int64)]]
        stop = cursor + len(signature)
        indices[cursor:stop] = np.sort(mapped_samples)
        data[cursor:stop] = 1
        cursor = stop
    indices[cursor:] = mapping[A.indices[edge_insert:]]
    data[cursor:] = A.data[edge_insert:]

    indptr = np.empty(old_n + count + 1, dtype=np.int32)
    indptr[: insert_at + 1] = A.indptr[: insert_at + 1]
    indptr[insert_at + 1 : insert_at + count + 1] = edge_insert + np.cumsum(signature_lengths, dtype=np.int64)
    indptr[insert_at + count :] = A.indptr[insert_at:] + edge_count
    expanded = csc_matrix((data, indices, indptr), shape=(old_n + count, old_n + count), copy=False)
    return expanded, mapping, new_nodes


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray, *, strings: bool = False) -> None:
    if name in group:
        del group[name]
    kwargs = {"compression": "gzip", "shuffle": True}
    if strings:
        kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
        data = np.asarray(data, dtype=object)
    group.create_dataset(name, data=data, **kwargs)


def _repack_hdf5_file(source_path: Path, output_path: Path) -> None:
    """Copy every live HDF5 object into a compact new container."""
    if source_path.resolve() == output_path.resolve():
        raise ValueError("HDF5 repack source and output must differ")
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        for name in source:
            source.copy(name, output, name=name)
        output.flush()


def _metadata(group: h5py.Group) -> dict[str, list[object]]:
    missing = [name for name in META_COLUMNS if name not in group]
    if missing:
        raise ValueError(f"block {group.name} is missing metadata datasets: {missing}")
    return {name: list(group[name][:]) for name in META_COLUMNS}


def _new_metadata_values(variants: Sequence[RareVariant], name: str) -> np.ndarray:
    """Return one incoming metadata column without building per-record dicts."""
    attributes = {
        "CHROM": "chrom",
        "POS": "pos",
        "ID": "variant_id",
        "REF": "ref",
        "ALT": "alt",
    }
    attribute = attributes[name]
    if name == "POS":
        return np.fromiter((variant.pos for variant in variants), dtype=np.int64, count=len(variants))
    return np.asarray([getattr(variant, attribute) for variant in variants], dtype=object)


def _check_new_alleles_absent(
    group: h5py.Group,
    variants: Sequence[RareVariant],
    *,
    chunk_size: int = 250_000,
) -> None:
    """Check old/new allele overlap using memory bounded by the incoming keys."""
    new_keys = {variant.allele_key for variant in variants}
    duplicates = []
    n_variants = len(group["POS"])
    for start in range(0, n_variants, chunk_size):
        stop = min(start + chunk_size, n_variants)
        columns = {name: group[name][start:stop] for name in ("CHROM", "POS", "REF", "ALT")}
        for chrom, pos, ref, alt in zip(
            columns["CHROM"],
            columns["POS"],
            columns["REF"],
            columns["ALT"],
        ):
            key = _allele_key(chrom, pos, ref, alt)
            if key in new_keys:
                duplicates.append(key)
                if len(duplicates) == 5:
                    break
        if len(duplicates) == 5:
            break
    if duplicates:
        raise ValueError(f"rare-variant input already exists in block {group.name}: {duplicates}")


def _update_threshold_counts(group: h5py.Group, variants: Sequence[RareVariant], n_samples: int) -> None:
    """Add the new variants to persisted MAF-threshold summary counts."""
    has_values = "threshold_values" in group.attrs
    has_counts = "threshold_n_variants" in group.attrs
    if not has_values and not has_counts:
        return
    if has_values != has_counts:
        raise ValueError(f"block {group.name} has incomplete threshold summary attributes")

    thresholds = np.asarray(group.attrs["threshold_values"])
    counts = np.asarray(group.attrs["threshold_n_variants"])
    if thresholds.ndim != 1 or counts.shape != thresholds.shape:
        raise ValueError(f"block {group.name} has inconsistent threshold summary attributes")

    added_counts = np.asarray([variant.allele_count for variant in variants], dtype=np.float64)
    added_frequencies = added_counts / n_samples
    added_maf = np.minimum(added_frequencies, 1 - added_frequencies)
    increments = (added_maf[:, np.newaxis] > thresholds).sum(axis=0)
    updated = counts + increments
    group.attrs.modify("threshold_n_variants", updated.astype(counts.dtype, copy=False))


def _augment_block_batch_only(
    group: h5py.Group,
    variants: Sequence[RareVariant],
    iids: Sequence[str],
) -> AugmentationStats:
    """Augment one block with batch-only reuse and bounded metadata memory."""
    stats = AugmentationStats(variants_added=len(variants))
    if not variants:
        return stats
    n_nodes = int(group.attrs["n"])
    n_samples = int(group.attrs["n_samples"])
    n_individual_nodes = int(group.attrs.get("n_individuals", 0))
    if n_samples != 2 * len(iids):
        raise ValueError(
            f"block {group.name} has {n_samples} haplotypes for {len(iids)} IIDs; "
            "this utility currently requires two haplotypes per IID"
        )
    iid_to_index = {iid: index for index, iid in enumerate(iids)}
    if len(iid_to_index) != len(iids):
        raise ValueError("LinearARG contains duplicate IIDs")
    missing = [name for name in META_COLUMNS if name not in group]
    if missing:
        raise ValueError(f"block {group.name} is missing metadata datasets: {missing}")

    old_variant_indices = np.asarray(group["variant_indices"][:], dtype=np.int32)
    old_flip = np.asarray(group["flip"][:], dtype=bool)
    old_counts = np.asarray(group["allele_counts"][:], dtype=np.int64) if "allele_counts" in group else None
    _check_new_alleles_absent(group, variants)
    output_chrom = _decode(group["CHROM"][0]) if len(group["CHROM"]) else _decode(group.attrs.get("chrom", ""))

    ordered = sorted(variants, key=lambda value: (value.pos, value.ref, value.alt, value.variant_id))
    old_samples = _sample_indices(n_nodes, n_samples, n_individual_nodes)
    selected_nodes: dict[tuple[int, ...], int] = {}
    novel_signatures: list[tuple[int, ...]] = []
    source_kind = np.empty(len(ordered), dtype=np.uint8)
    source_value = np.empty(len(ordered), dtype=np.int32)
    new_counts = np.empty(len(ordered), dtype=np.int64)
    candidate_selection_started = time.perf_counter()
    for index, variant in enumerate(ordered):
        candidates = _phase_candidates(variant, iid_to_index)
        new_counts[index] = variant.allele_count
        if variant.allele_count == 1:
            signature = min(candidates)
            source_kind[index] = 0
            source_value[index] = old_samples[signature[0]]
            stats.direct_singletons += 1
            continue
        stats.doubletons_added += 1
        signature, source = _choose_signature(candidates, {}, selected_nodes)
        source_kind[index] = 1
        if source == "selected":
            source_value[index] = selected_nodes[signature]
            stats.reused_new_nodes += 1
        else:
            slot = len(novel_signatures)
            novel_signatures.append(signature)
            selected_nodes[signature] = slot
            source_value[index] = slot
            stats.nodes_added += 1
            stats.edges_added += len(signature)
    stats.candidate_selection_seconds += time.perf_counter() - candidate_selection_started

    matrix_load_started = time.perf_counter()
    A = _read_matrix(group)
    stats.matrix_load_seconds += time.perf_counter() - matrix_load_started
    graph_expansion_started = time.perf_counter()
    expanded, mapping, _ = _expand_graph(A, novel_signatures, n_samples, n_individual_nodes)
    stats.graph_expansion_seconds += time.perf_counter() - graph_expansion_started
    insert_at = n_nodes - n_individual_nodes - n_samples
    mapped_old_indices = mapping[old_variant_indices]
    new_variant_indices = np.empty(len(ordered), dtype=np.int32)
    sample_mask = source_kind == 0
    new_variant_indices[sample_mask] = mapping[source_value[sample_mask]]
    new_variant_indices[~sample_mask] = insert_at + source_value[~sample_mask]

    dataset_rewrite_started = time.perf_counter()
    _replace_dataset(group, "indptr", expanded.indptr)
    _replace_dataset(group, "indices", expanded.indices)
    _replace_dataset(group, "data", expanded.data)
    del A, expanded, mapping, novel_signatures, selected_nodes, source_kind, source_value

    record_build_started = time.perf_counter()
    old_positions = np.asarray(group["POS"][:], dtype=np.int64)
    new_positions = _new_metadata_values(ordered, "POS")
    combined_positions = np.concatenate((old_positions, new_positions))
    order = np.argsort(combined_positions, kind="stable")
    merged_positions = combined_positions[order]
    stats.record_build_seconds += time.perf_counter() - record_build_started

    _replace_dataset(
        group,
        "variant_indices",
        np.concatenate((mapped_old_indices, new_variant_indices))[order],
    )
    _replace_dataset(group, "flip", np.concatenate((old_flip, np.zeros(len(ordered), dtype=bool)))[order])
    for name in META_COLUMNS:
        if name == "POS":
            _replace_dataset(group, name, merged_positions)
            continue
        old_values = np.asarray(group[name][:], dtype=object)
        new_values = (
            np.repeat(output_chrom, len(ordered)).astype(object)
            if name == "CHROM"
            else _new_metadata_values(ordered, name)
        )
        merged_values = np.concatenate((old_values, new_values))[order]
        _replace_dataset(group, name, merged_values, strings=True)
        del old_values, new_values, merged_values
    if old_counts is not None:
        _replace_dataset(group, "allele_counts", np.concatenate((old_counts, new_counts))[order])
    if "nonunique_indices" in group:
        del group["nonunique_indices"]

    group.attrs["n"] = expanded_n = n_nodes + stats.nodes_added
    group.attrs["n_entries"] = int(group.attrs["n_entries"]) + stats.edges_added
    group.attrs["n_variants"] = len(order)
    _update_threshold_counts(group, variants, n_samples)
    group.attrs["rare_variant_phase_method"] = _phase_method("batch_only")
    group.attrs["rare_variant_reuse_policy"] = "batch_only"
    group.attrs["rare_variant_phase_is_inferred"] = False
    group.attrs["rare_variant_diploid_dosage_preserved"] = True
    group.attrs["rare_variants_added"] = stats.variants_added
    group.attrs["rare_variant_singletons_direct"] = stats.direct_singletons
    group.attrs["rare_variant_doubletons_added"] = stats.doubletons_added
    group.attrs["rare_variant_existing_nodes_reused"] = 0
    group.attrs["rare_variant_distinct_existing_nodes_reused"] = 0
    group.attrs["rare_variant_new_nodes_reused"] = stats.reused_new_nodes
    group.attrs["rare_variant_nodes_added"] = stats.nodes_added
    group.attrs["rare_variant_edges_added"] = stats.edges_added
    group.attrs["rare_variant_existing_candidate_nodes_scanned"] = 0
    group.attrs["rare_variant_existing_signatures_available"] = 0
    if expanded_n != len(group["indptr"]) - 1:
        raise AssertionError("expanded graph dimension does not match indptr")
    stats.dataset_rewrite_seconds += time.perf_counter() - dataset_rewrite_started
    return stats


def _augment_block(
    group: h5py.Group,
    variants: Sequence[RareVariant],
    iids: Sequence[str],
    *,
    reuse_policy: ReusePolicy = "existing_then_batch",
) -> AugmentationStats:
    """Augment one LinearARG HDF5 block in place."""
    if reuse_policy == "batch_only":
        return _augment_block_batch_only(group, variants, iids)
    stats = AugmentationStats(variants_added=len(variants))
    if not variants:
        return stats
    n_nodes = int(group.attrs["n"])
    n_samples = int(group.attrs["n_samples"])
    n_individual_nodes = int(group.attrs.get("n_individuals", 0))
    if n_samples != 2 * len(iids):
        raise ValueError(
            f"block {group.name} has {n_samples} haplotypes for {len(iids)} IIDs; "
            "this utility currently requires two haplotypes per IID"
        )
    iid_to_index = {iid: index for index, iid in enumerate(iids)}
    if len(iid_to_index) != len(iids):
        raise ValueError("LinearARG contains duplicate diploid IIDs")

    old_variant_indices = np.asarray(group["variant_indices"][:], dtype=np.int64)
    old_flip = np.asarray(group["flip"][:], dtype=bool)
    metadata = _metadata(group)
    old_keys = {
        _allele_key(chrom, pos, ref, alt)
        for chrom, pos, ref, alt in zip(metadata["CHROM"], metadata["POS"], metadata["REF"], metadata["ALT"])
    }
    duplicate = [variant.allele_key for variant in variants if variant.allele_key in old_keys]
    if duplicate:
        raise ValueError(f"rare-variant input already exists in block {group.name}: {duplicate[:5]}")

    has_doubletons = any(variant.allele_count == 2 for variant in variants)
    if "allele_counts" in group:
        old_counts = np.asarray(group["allele_counts"][:], dtype=np.int64)
        doubleton_mask = np.flatnonzero(old_counts == 2)
    else:
        old_counts = None
        doubleton_mask = np.arange(len(old_variant_indices))
    A: csc_matrix | None = None
    existing_nodes: dict[tuple[int, ...], int] = {}
    if has_doubletons and reuse_policy == "existing_then_batch":
        matrix_load_started = time.perf_counter()
        A = _read_matrix(group)
        stats.matrix_load_seconds += time.perf_counter() - matrix_load_started

        existing_scan_started = time.perf_counter()
        core_n = n_nodes - n_individual_nodes
        A_core = A[:core_n, :core_n].tocsc()
        candidate_nodes = set(map(int, old_variant_indices[doubleton_mask]))
        stats.existing_candidate_nodes_scanned = len(candidate_nodes)
        signatures_by_node = _capped_leaf_signatures(A_core, candidate_nodes, n_samples)
        for node, signature in signatures_by_node.items():
            if len(signature) == 2:
                existing_nodes.setdefault(signature, node)
        stats.existing_signatures_available = len(existing_nodes)
        stats.existing_scan_seconds += time.perf_counter() - existing_scan_started

    selected_nodes: dict[tuple[int, ...], int] = {}
    chosen: list[tuple[RareVariant, tuple[int, ...], str, int | None]] = []
    novel_signatures: list[tuple[int, ...]] = []
    used_existing_signatures: set[tuple[int, ...]] = set()
    candidate_selection_started = time.perf_counter()
    for variant in sorted(variants, key=lambda value: (value.pos, value.ref, value.alt, value.variant_id)):
        candidates = _phase_candidates(variant, iid_to_index)
        new_node_slot: int | None = None
        if variant.allele_count == 1:
            # Either haplotype preserves the unphased dosage. Pointing the
            # variant directly at the deterministic first candidate avoids
            # both carrier-signature traversal and a singleton graph node.
            signature = min(candidates)
            source = "sample"
            stats.direct_singletons += 1
        else:
            stats.doubletons_added += 1
            reusable_new_nodes = selected_nodes if reuse_policy != "none" else {}
            signature, source = _choose_signature(candidates, existing_nodes, reusable_new_nodes)
        if source == "existing":
            stats.reused_existing_nodes += 1
            used_existing_signatures.add(signature)
        elif source == "selected":
            stats.reused_new_nodes += 1
            new_node_slot = selected_nodes[signature]
        elif source == "new":
            new_node_slot = len(novel_signatures)
            novel_signatures.append(signature)
            if reuse_policy != "none":
                selected_nodes[signature] = new_node_slot
            stats.nodes_added += 1
            stats.edges_added += len(signature)
        chosen.append((variant, signature, source, new_node_slot))
    stats.distinct_existing_nodes_reused = len(used_existing_signatures)
    stats.candidate_selection_seconds += time.perf_counter() - candidate_selection_started

    old_samples = _sample_indices(n_nodes, n_samples, n_individual_nodes)
    if novel_signatures:
        assert has_doubletons
        if A is None:
            matrix_load_started = time.perf_counter()
            A = _read_matrix(group)
            stats.matrix_load_seconds += time.perf_counter() - matrix_load_started
        graph_expansion_started = time.perf_counter()
        expanded, mapping, new_nodes = _expand_graph(A, novel_signatures, n_samples, n_individual_nodes)
        stats.graph_expansion_seconds += time.perf_counter() - graph_expansion_started
    else:
        expanded = None
        mapping = None
        new_nodes = []
    mapped_existing_nodes = {
        signature: int(node if mapping is None else mapping[node]) for signature, node in existing_nodes.items()
    }

    record_build_started = time.perf_counter()
    additions = []
    output_chrom = (
        _decode(metadata["CHROM"][0]) if metadata["CHROM"] else _decode(group.attrs.get("chrom", ""))
    )
    for ordinal, (variant, signature, source, new_node_slot) in enumerate(chosen):
        if source == "sample":
            sample_node = old_samples[signature[0]]
            node = int(sample_node if mapping is None else mapping[sample_node])
        elif source == "existing":
            node = mapped_existing_nodes[signature]
        else:
            assert new_node_slot is not None
            node = new_nodes[new_node_slot]
        additions.append(
            {
                "CHROM": output_chrom,
                "POS": variant.pos,
                "ID": variant.variant_id,
                "REF": variant.ref,
                "ALT": variant.alt,
                "variant_index": node,
                "flip": False,
                "allele_count": variant.allele_count,
                "is_new": True,
                "ordinal": ordinal,
            }
        )

    records = []
    for index in range(len(old_variant_indices)):
        records.append(
            {
                **{name: metadata[name][index] for name in META_COLUMNS},
                "variant_index": int(
                    old_variant_indices[index] if mapping is None else mapping[old_variant_indices[index]]
                ),
                "flip": bool(old_flip[index]),
                "allele_count": None if old_counts is None else int(old_counts[index]),
                "is_new": False,
                "ordinal": index,
            }
        )
    records.extend(additions)
    # Stable POS-only ordering preserves the established order of same-position
    # alleles; new alleles follow existing alleles at that position.
    records.sort(key=lambda record: (int(record["POS"]), bool(record["is_new"]), int(record["ordinal"])))
    stats.record_build_seconds += time.perf_counter() - record_build_started

    dataset_rewrite_started = time.perf_counter()
    if expanded is not None:
        _replace_dataset(group, "indptr", expanded.indptr)
        _replace_dataset(group, "indices", expanded.indices)
        _replace_dataset(group, "data", expanded.data)
    _replace_dataset(group, "variant_indices", np.asarray([r["variant_index"] for r in records], dtype=np.int32))
    _replace_dataset(group, "flip", np.asarray([r["flip"] for r in records], dtype=bool))
    for name in META_COLUMNS:
        values = np.asarray([r[name] for r in records])
        _replace_dataset(group, name, values.astype(np.int64) if name == "POS" else values, strings=name != "POS")
    if old_counts is not None:
        _replace_dataset(group, "allele_counts", np.asarray([r["allele_count"] for r in records], dtype=np.int64))
    if "nonunique_indices" in group:
        del group["nonunique_indices"]

    if expanded is not None:
        group.attrs["n"] = expanded.shape[0]
        group.attrs["n_entries"] = expanded.nnz
    group.attrs["n_variants"] = len(records)
    _update_threshold_counts(group, variants, n_samples)
    group.attrs["rare_variant_phase_method"] = _phase_method(reuse_policy)
    group.attrs["rare_variant_reuse_policy"] = reuse_policy
    group.attrs["rare_variant_phase_is_inferred"] = False
    group.attrs["rare_variant_diploid_dosage_preserved"] = True
    group.attrs["rare_variants_added"] = stats.variants_added
    group.attrs["rare_variant_singletons_direct"] = stats.direct_singletons
    group.attrs["rare_variant_doubletons_added"] = stats.doubletons_added
    group.attrs["rare_variant_existing_nodes_reused"] = stats.reused_existing_nodes
    group.attrs["rare_variant_distinct_existing_nodes_reused"] = stats.distinct_existing_nodes_reused
    group.attrs["rare_variant_new_nodes_reused"] = stats.reused_new_nodes
    group.attrs["rare_variant_nodes_added"] = stats.nodes_added
    group.attrs["rare_variant_edges_added"] = stats.edges_added
    group.attrs["rare_variant_existing_candidate_nodes_scanned"] = stats.existing_candidate_nodes_scanned
    group.attrs["rare_variant_existing_signatures_available"] = stats.existing_signatures_available
    stats.dataset_rewrite_seconds += time.perf_counter() - dataset_rewrite_started
    return stats


def _augment_file(
    input_h5: Path,
    carrier_table: Path,
    output_h5: Path,
    *,
    reuse_policy: ReusePolicy = "existing_then_batch",
) -> AugmentationStats:
    """Copy and augment a block-structured LinearARG HDF5 file."""
    total_started = time.perf_counter()
    reuse_policy = _validate_reuse_policy(reuse_policy)
    if output_h5.exists():
        raise FileExistsError(f"output already exists: {output_h5}")

    carrier_parse_started = time.perf_counter()
    variants = _read_carrier_table(carrier_table)
    carrier_parse_seconds = time.perf_counter() - carrier_parse_started
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    repacked: Path | None = None
    try:
        file_copy_started = time.perf_counter()
        shutil.copy2(input_h5, temporary)
        file_copy_seconds = time.perf_counter() - file_copy_started
        total = AugmentationStats()
        total.carrier_parse_seconds = carrier_parse_seconds
        total.file_copy_seconds = file_copy_seconds
        with h5py.File(temporary, "r+") as h5:
            if "iids" not in h5:
                raise ValueError("input HDF5 is missing root-level iids")
            iid_normalization_started = time.perf_counter()
            raw_iids = [_decode(value) for value in h5["iids"][:]]
            sample_counts = {int(h5[name].attrs["n_samples"]) for name in _block_names(h5)}
            if len(sample_counts) != 1:
                raise ValueError(f"LinearARG blocks have inconsistent haplotype sample counts: {sorted(sample_counts)}")
            iids = _diploid_iids(raw_iids, sample_counts.pop())
            total.iid_normalization_seconds = time.perf_counter() - iid_normalization_started
            block_assignment_started = time.perf_counter()
            assignments = _assign_variants_to_blocks(h5, variants)
            total.block_assignment_seconds = time.perf_counter() - block_assignment_started
            for block, block_variants in assignments.items():
                total.add(_augment_block(h5[block], block_variants, iids, reuse_policy=reuse_policy))
            h5.attrs["rare_variant_phase_method"] = _phase_method(reuse_policy)
            h5.attrs["rare_variant_reuse_policy"] = reuse_policy
            h5.attrs["rare_variant_phase_is_inferred"] = False
            h5.attrs["rare_variant_diploid_dosage_preserved"] = True
            h5.attrs["rare_variants_added"] = total.variants_added
            h5.attrs["rare_variant_singletons_direct"] = total.direct_singletons
            h5.attrs["rare_variant_doubletons_added"] = total.doubletons_added
            h5.attrs["rare_variant_existing_nodes_reused"] = total.reused_existing_nodes
            h5.attrs["rare_variant_distinct_existing_nodes_reused"] = total.distinct_existing_nodes_reused
            h5.attrs["rare_variant_new_nodes_reused"] = total.reused_new_nodes
            h5.attrs["rare_variant_nodes_added"] = total.nodes_added
            h5.attrs["rare_variant_edges_added"] = total.edges_added
        fd, repacked_name = tempfile.mkstemp(
            prefix=f".{output_h5.name}.repack.",
            suffix=".tmp",
            dir=output_h5.parent,
        )
        os.close(fd)
        repacked = Path(repacked_name)
        repack_started = time.perf_counter()
        _repack_hdf5_file(temporary, repacked)
        total.file_repack_seconds = time.perf_counter() - repack_started
        temporary.unlink()
        os.replace(repacked, output_h5)
        repacked = None
        total.total_seconds = time.perf_counter() - total_started
        return total
    except BaseException:
        temporary.unlink(missing_ok=True)
        if repacked is not None:
            repacked.unlink(missing_ok=True)
        raise


def read_rare_variant_carriers(path: Path | str) -> list[RareVariant]:
    """Read and validate a sparse singleton/doubleton carrier table.

    **Arguments:**

    - `path`: Tab-separated carrier table with `CHROM`, `POS`, `ID`, `REF`,
      `ALT`, `IID`, and `DOSAGE` columns.

    **Returns:**

    - Rare variants sorted by chromosome, position, and allele.

    **Raises:**

    - `ValueError`: If required columns or calls are invalid, an IID is
      duplicated within an allele, or total allele count is not one or two.
    """
    return _read_carrier_table(Path(path))


def augment_rare_variants_file(
    input_h5: Path | str,
    carrier_table: Path | str,
    output_h5: Path | str,
    *,
    reuse_policy: ReusePolicy = "existing_then_batch",
) -> AugmentationStats:
    """Copy and augment a block-structured LinearARG HDF5 file.

    The output is assembled under a temporary name, repacked into a compact
    HDF5 container, and atomically installed only after every block succeeds.
    An output path that already exists when augmentation begins is rejected.

    !!! info

        Heterozygous calls are pseudo-phased to minimize incremental graph
        storage. Diploid dosage is preserved, but the selected phase is not a
        biological inference. Input blocks must contain uniformly diploid data,
        represented either by one IID per individual or two adjacent haplotype
        rows per IID. `existing_then_batch` preserves the production behavior;
        `batch_only` skips existing-graph traversal while retaining within-run
        reuse; `none` creates an independent node per doubleton.

    **Arguments:**

    - `input_h5`: Existing block-structured LinearARG HDF5 file.
    - `carrier_table`: Sparse singleton/doubleton carrier table.
    - `output_h5`: Path for the new augmented copy.
    - `reuse_policy`: Doubleton-node reuse policy. One of
      `existing_then_batch`, `batch_only`, or `none`.

    **Returns:**

    - Aggregate augmentation statistics across blocks.

    **Raises:**

    - `FileExistsError`: If `output_h5` already exists.
    - `ValueError`: If carrier data cannot be assigned or augmented safely.
    """
    return _augment_file(
        Path(input_h5),
        Path(carrier_table),
        Path(output_h5),
        reuse_policy=reuse_policy,
    )
