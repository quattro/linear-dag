"""Add unphased singleton/doubleton dosages to a LinearARG HDF5 file.

The input is a tab-separated, sparse carrier table with columns
CHROM, POS, ID, REF, ALT, IID, and DOSAGE.  There is one row per non-reference
individual/variant call.  DOSAGE must be 1 or 2, and the total dosage for a
variant must be one or two.

Heterozygous calls are pseudo-phased to minimize incremental graph storage.
Singletons point directly to one deterministic haplotype sample node and add
no graph nodes or edges.  For doubletons, an existing rare-variant node is
preferred, followed by a carrier pattern selected earlier in the same run.
Remaining ties are deterministic.  The result preserves diploid dosage, but
the chosen rare-variant phase is not a biological inference.
"""

from __future__ import annotations

import csv
import os
import shutil
import tempfile

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from scipy.sparse import coo_matrix, csc_matrix

try:  # Register Blosc filters when the source LinearARG uses them.
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


REQUIRED_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT", "IID", "DOSAGE")
META_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT")


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
        return self.chrom, self.pos, self.ref, self.alt


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
    reused_existing_nodes: int = 0
    reused_new_nodes: int = 0
    nodes_added: int = 0
    edges_added: int = 0

    def add(self, other: "AugmentationStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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
            key = (row["CHROM"], pos, row["REF"], row["ALT"])
            item = grouped.setdefault(key, {"id": row["ID"] or ".", "calls": {}})
            if item["id"] != (row["ID"] or "."):
                raise ValueError(f"conflicting IDs for allele {key}")
            calls = item["calls"]
            assert isinstance(calls, dict)
            if row["IID"] in calls:
                raise ValueError(f"duplicate IID {row['IID']!r} for allele {key}")
            calls[row["IID"]] = dosage

    result = []
    for (chrom, pos, ref, alt), item in grouped.items():
        calls_dict = item["calls"]
        assert isinstance(calls_dict, dict)
        calls = tuple(CarrierCall(iid, dosage) for iid, dosage in sorted(calls_dict.items()))
        variant = RareVariant(chrom, pos, str(item["id"]), ref, alt, calls)
        if variant.allele_count not in (1, 2):
            raise ValueError(f"allele {variant.allele_key} has allele count {variant.allele_count}; expected 1 or 2")
        result.append(variant)
    return sorted(result, key=lambda v: (v.chrom, v.pos, v.ref, v.alt, v.variant_id))


def _block_names(h5: h5py.File) -> list[str]:
    return [name for name in h5 if name != "iids" and isinstance(h5[name], h5py.Group)]


def _chrom_equal(left: object, right: str) -> bool:
    return _decode(left).removeprefix("chr") == right.removeprefix("chr")


def _assign_variants_to_blocks(h5: h5py.File, variants: Sequence[RareVariant]) -> dict[str, list[RareVariant]]:
    """Assign each variant to exactly one HDF5 genomic block."""
    blocks = _block_names(h5)
    if not blocks:
        raise ValueError("input HDF5 has no LinearARG block groups")
    assigned: dict[str, list[RareVariant]] = defaultdict(list)
    for variant in variants:
        matches = []
        for name in blocks:
            group = h5[name]
            chrom = group.attrs.get("chrom", name.split(":", 1)[0])
            start = int(group.attrs.get("start", 0))
            end = int(group.attrs.get("end", np.iinfo(np.int64).max))
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
    old_n = A.shape[0]
    insert_at = old_n - n_individual_nodes - n_samples
    count = len(signatures)
    mapping = np.arange(old_n, dtype=np.int64)
    mapping[insert_at:] += count

    coo = A.tocoo()
    rows = list(mapping[coo.row])
    cols = list(mapping[coo.col])
    data = list(coo.data)
    new_nodes = list(range(insert_at, insert_at + count))
    old_samples = _sample_indices(old_n, n_samples, n_individual_nodes)
    for node, signature in zip(new_nodes, signatures):
        for haplotype in signature:
            rows.append(int(mapping[old_samples[haplotype]]))
            cols.append(node)
            data.append(1)
    expanded = coo_matrix(
        (
            np.asarray(data, dtype=np.int32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(old_n + count, old_n + count),
    ).tocsc()
    # The accelerated triangular solvers use C ``int`` memoryviews for all
    # graph arrays, so SciPy's COO conversion must not promote sparse indices.
    expanded.indices = expanded.indices.astype(np.int32, copy=False)
    expanded.indptr = expanded.indptr.astype(np.int32, copy=False)
    expanded.sort_indices()
    return expanded, mapping, new_nodes


def _replace_dataset(group: h5py.Group, name: str, data: np.ndarray, *, strings: bool = False) -> None:
    if name in group:
        del group[name]
    kwargs = {"compression": "gzip", "shuffle": True}
    if strings:
        kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
        data = np.asarray(data, dtype=object)
    group.create_dataset(name, data=data, **kwargs)


def _metadata(group: h5py.Group) -> dict[str, list[object]]:
    missing = [name for name in META_COLUMNS if name not in group]
    if missing:
        raise ValueError(f"block {group.name} is missing metadata datasets: {missing}")
    return {name: list(group[name][:]) for name in META_COLUMNS}


def _augment_block(
    group: h5py.Group,
    variants: Sequence[RareVariant],
    iids: Sequence[str],
) -> AugmentationStats:
    """Augment one LinearARG HDF5 block in place."""
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

    old_variant_indices = np.asarray(group["variant_indices"][:], dtype=np.int64)
    old_flip = np.asarray(group["flip"][:], dtype=bool)
    metadata = _metadata(group)
    old_keys = {
        (_decode(chrom), int(pos), _decode(ref), _decode(alt))
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
    if has_doubletons:
        A = _read_matrix(group)
        core_n = n_nodes - n_individual_nodes
        A_core = A[:core_n, :core_n].tocsc()
        signatures_by_node = _capped_leaf_signatures(A_core, old_variant_indices[doubleton_mask], n_samples)
    else:
        signatures_by_node = {}
    existing_nodes: dict[tuple[int, ...], int] = {}
    for node, signature in signatures_by_node.items():
        if len(signature) == 2:
            existing_nodes.setdefault(signature, node)

    selected_nodes: dict[tuple[int, ...], int] = {}
    chosen: list[tuple[RareVariant, tuple[int, ...], str]] = []
    novel_signatures: list[tuple[int, ...]] = []
    for variant in sorted(variants, key=lambda value: (value.pos, value.ref, value.alt, value.variant_id)):
        candidates = _phase_candidates(variant, iid_to_index)
        if variant.allele_count == 1:
            # Either haplotype preserves the unphased dosage.  Pointing the
            # variant directly at the deterministic first candidate avoids
            # both carrier-signature traversal and a singleton graph node.
            signature = min(candidates)
            source = "sample"
            stats.direct_singletons += 1
        else:
            signature, source = _choose_signature(candidates, existing_nodes, selected_nodes)
        if source == "existing":
            stats.reused_existing_nodes += 1
        elif source == "selected":
            stats.reused_new_nodes += 1
        elif source == "new":
            selected_nodes[signature] = -1
            novel_signatures.append(signature)
            stats.nodes_added += 1
            stats.edges_added += len(signature)
        chosen.append((variant, signature, source))

    old_samples = _sample_indices(n_nodes, n_samples, n_individual_nodes)
    if novel_signatures:
        assert has_doubletons
        expanded, mapping, new_nodes = _expand_graph(A, novel_signatures, n_samples, n_individual_nodes)
    else:
        expanded = None
        mapping = None
        new_nodes = []
    for signature, node in zip(novel_signatures, new_nodes):
        selected_nodes[signature] = node
    mapped_existing_nodes = {
        signature: int(node if mapping is None else mapping[node]) for signature, node in existing_nodes.items()
    }

    additions = []
    for ordinal, (variant, signature, source) in enumerate(chosen):
        if source == "sample":
            sample_node = old_samples[signature[0]]
            node = int(sample_node if mapping is None else mapping[sample_node])
        elif source == "existing":
            node = mapped_existing_nodes[signature]
        else:
            node = selected_nodes[signature]
        additions.append(
            {
                "CHROM": variant.chrom,
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
    group.attrs["rare_variant_phase_method"] = "direct_singletons_greedy_doubletons"
    group.attrs["rare_variant_phase_is_inferred"] = False
    group.attrs["rare_variant_diploid_dosage_preserved"] = True
    group.attrs["rare_variants_added"] = stats.variants_added
    group.attrs["rare_variant_singletons_direct"] = stats.direct_singletons
    group.attrs["rare_variant_nodes_added"] = stats.nodes_added
    group.attrs["rare_variant_edges_added"] = stats.edges_added
    return stats


def _augment_file(input_h5: Path, carrier_table: Path, output_h5: Path) -> AugmentationStats:
    """Copy and augment a block-structured LinearARG HDF5 file."""
    if output_h5.exists():
        raise FileExistsError(f"output already exists: {output_h5}")
    variants = _read_carrier_table(carrier_table)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_h5.name}.", suffix=".tmp", dir=output_h5.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(input_h5, temporary)
        total = AugmentationStats()
        with h5py.File(temporary, "r+") as h5:
            if "iids" not in h5:
                raise ValueError("input HDF5 is missing root-level iids")
            iids = [_decode(value) for value in h5["iids"][:]]
            assignments = _assign_variants_to_blocks(h5, variants)
            for block, block_variants in assignments.items():
                total.add(_augment_block(h5[block], block_variants, iids))
            h5.attrs["rare_variant_phase_method"] = "minimum_incremental_edges_greedy"
            h5.attrs["rare_variant_phase_is_inferred"] = False
            h5.attrs["rare_variant_diploid_dosage_preserved"] = True
        os.replace(temporary, output_h5)
        return total
    except Exception:
        temporary.unlink(missing_ok=True)
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
    input_h5: Path | str, carrier_table: Path | str, output_h5: Path | str
) -> AugmentationStats:
    """Copy and augment a block-structured LinearARG HDF5 file.

    The output is assembled under a temporary name and atomically installed
    only after every block succeeds. Existing output paths are never replaced.

    !!! info

        Heterozygous calls are pseudo-phased to minimize incremental graph
        storage. Diploid dosage is preserved, but the selected phase is not a
        biological inference.

    **Arguments:**

    - `input_h5`: Existing block-structured LinearARG HDF5 file.
    - `carrier_table`: Sparse singleton/doubleton carrier table.
    - `output_h5`: Path for the new augmented copy.

    **Returns:**

    - Aggregate augmentation statistics across blocks.

    **Raises:**

    - `FileExistsError`: If `output_h5` already exists.
    - `ValueError`: If carrier data cannot be assigned or augmented safely.
    """
    return _augment_file(Path(input_h5), Path(carrier_table), Path(output_h5))
