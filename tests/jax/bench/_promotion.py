# pattern: Functional Core

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import jax
import jaxlib
import numpy as np

from linear_dag.core.lineararg import list_blocks

SCHEMA_VERSION = "2026-08-13+2"
CURRENT_REFERENCE_COMMIT = "b68e7da"
REQUIRED_PRODUCT_KS = (4, 20)
REQUIRED_PRODUCT_OPERATIONS = ("matmat", "rmatmat")
KNOWN_SCHEMA_VERSIONS = {SCHEMA_VERSION}
KNOWN_DTYPES = {"float16", "bfloat16", "float32", "float64"}
KNOWN_PHASES = {item.value for item in []}


class TimingPhase(str, Enum):
    CONSTRUCTION = "construction"
    LOWERING = "lowering"
    COMPILATION = "compilation"
    FIRST_EXECUTION = "first_execution"
    WARM_EXECUTION = "warm_execution"


class Representation(str, Enum):
    PACKED_CANDIDATE = "packed_candidate"
    RETAINED_EXACT_RAGGED = "retained_exact_ragged"
    NUMPY_CYTHON = "numpy_cython"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    MISSING = "missing"


class Decision(str, Enum):
    PROMOTE = "promote"
    CONTINUE_COEXISTENCE = "continue_coexistence"
    REJECT = "reject"


class CachePolicy(str, Enum):
    FRESH = "fresh"
    REUSED = "reused"


class GateFailureReason(str, Enum):
    REPO_MISMATCH = "repo_mismatch"
    MISSING_EXACT = "missing_exact_baseline"
    MISSING_EVIDENCE = "missing_required_evidence"
    NONPOSITIVE_TIMING = "non_positive_timing"
    DIRTY_WORKTREE = "dirty_worktree"
    CANDIDATE_MISMATCH = "candidate_commit_mismatch"
    UNKNOWN_SCHEMA = "unknown_schema_version"


KNOWN_PHASES = {item.value for item in TimingPhase}


def none_or_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _normalize_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _repo_root(path: Path | None = None) -> Path:
    base = path if path is not None else Path(__file__).resolve().parents[3]
    return Path(
        subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )


def git_commit(repo_root: Path | str | None = None) -> str:
    root = _repo_root(Path(repo_root) if repo_root is not None else None)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def is_git_dirty(repo_root: Path | str | None = None) -> bool:
    root = _repo_root(Path(repo_root) if repo_root is not None else None)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return bool(status)


@dataclass(frozen=True)
class DatasetFingerprint:
    sha256: str
    size_bytes: int
    block_count: int
    n_samples: int
    n_variants: int

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError(f"sha256 must be a 64-character hex digest, observed {self.sha256!r}")
        for field_name, value in (
            ("size_bytes", self.size_bytes),
            ("block_count", self.block_count),
            ("n_samples", self.n_samples),
            ("n_variants", self.n_variants),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative int, observed {value!r}")

    @property
    def logical_shape(self) -> tuple[int, int]:
        return (self.n_samples, self.n_variants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "block_count": self.block_count,
            "n_samples": self.n_samples,
            "n_variants": self.n_variants,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetFingerprint":
        required = {"sha256", "size_bytes", "block_count", "n_samples", "n_variants"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"dataset fingerprint missing required fields: {sorted(missing)}")
        return cls(
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            block_count=int(payload["block_count"]),
            n_samples=int(payload["n_samples"]),
            n_variants=int(payload["n_variants"]),
        )


@dataclass(frozen=True)
class EnvironmentState:
    platform_label: str
    python_version: str
    numpy_version: str
    jax_version: str
    jaxlib_version: str
    os_name: str
    machine: str
    architecture: str
    xla_flags: str
    devices: tuple[str, ...]
    xla_cache_dir: str | None
    command: str
    dirty_worktree: bool

    def __post_init__(self) -> None:
        if not self.platform_label:
            raise ValueError("platform_label must be non-empty")
        if not self.python_version:
            raise ValueError("python_version must be non-empty")
        if not self.command:
            raise ValueError("command must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_label": self.platform_label,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "os_name": self.os_name,
            "machine": self.machine,
            "architecture": self.architecture,
            "xla_flags": self.xla_flags,
            "devices": list(self.devices),
            "xla_cache_dir": self.xla_cache_dir,
            "command": self.command,
            "dirty_worktree": self.dirty_worktree,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnvironmentState":
        required = {
            "platform_label",
            "python_version",
            "numpy_version",
            "jax_version",
            "jaxlib_version",
            "os_name",
            "machine",
            "architecture",
            "xla_flags",
            "devices",
            "xla_cache_dir",
            "command",
            "dirty_worktree",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"environment missing required fields: {sorted(missing)}")
        devices = payload.get("devices")
        if not isinstance(devices, list):
            raise ValueError("environment.devices must be a list")
        return cls(
            platform_label=str(payload["platform_label"]),
            python_version=str(payload["python_version"]),
            numpy_version=str(payload["numpy_version"]),
            jax_version=str(payload["jax_version"]),
            jaxlib_version=str(payload["jaxlib_version"]),
            os_name=str(payload["os_name"]),
            machine=str(payload["machine"]),
            architecture=str(payload["architecture"]),
            xla_flags=str(payload["xla_flags"]),
            devices=tuple(str(item) for item in devices),
            xla_cache_dir=None if payload["xla_cache_dir"] is None else str(payload["xla_cache_dir"]),
            command=str(payload["command"]),
            dirty_worktree=bool(payload["dirty_worktree"]),
        )


@dataclass(frozen=True)
class TimedPhase:
    phase: str
    seconds: float | None
    null_reason: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in KNOWN_PHASES:
            raise ValueError(f"unknown timing phase: {self.phase!r}")
        if self.seconds is None:
            if not self.null_reason or not self.null_reason.strip():
                raise ValueError(f"timed phase {self.phase!r} requires non-empty null_reason when timing is null")
            return
        if self.null_reason is not None:
            raise ValueError(f"timed phase {self.phase!r} cannot include null_reason when timing is present")
        if not isinstance(self.seconds, (int, float)):
            raise ValueError(f"timed phase {self.phase!r} seconds must be numeric")
        if self.seconds <= 0:
            raise ValueError(f"timed phase {self.phase!r} seconds must be positive")
        object.__setattr__(self, "seconds", float(self.seconds))

    @property
    def is_measured(self) -> bool:
        return self.seconds is not None

    @property
    def kind(self) -> str:
        return "measured" if self.seconds is not None else "missing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "seconds": self.seconds,
            "null_reason": self.null_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TimedPhase":
        for key in ("phase", "seconds"):
            if key not in payload:
                raise ValueError(f"timed phase missing {key!r}")
        return cls(
            phase=str(payload["phase"]),
            seconds=None if payload["seconds"] is None else float(payload["seconds"]),
            null_reason=None if payload.get("null_reason") is None else str(payload["null_reason"]),
        )


@dataclass(frozen=True)
class PerformanceMetrics:
    canonical_graph_bytes: int | None = None
    padded_graph_bytes: int | None = None
    descriptor_bytes: int | None = None
    resident_graph_bytes: int | None = None
    max_device_graph_bytes: int | None = None
    staging_bytes: int | None = None
    component_count: int | None = None
    pytree_leaf_count: int | None = None
    graph_constant_bytes: int | None = None
    graph_operand_count: int | None = None
    stablehlo_operation_count: int | None = None
    xla_buffer_assignment_total_bytes: int | None = None
    logical_collective_bytes: int | None = None
    graph_bytes_by_device_count: int | None = None
    graph_bytes_by_device_max: int | None = None

    def __post_init__(self) -> None:
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"metric {key} must be a non-negative int, observed {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_graph_bytes": self.canonical_graph_bytes,
            "padded_graph_bytes": self.padded_graph_bytes,
            "descriptor_bytes": self.descriptor_bytes,
            "resident_graph_bytes": self.resident_graph_bytes,
            "max_device_graph_bytes": self.max_device_graph_bytes,
            "staging_bytes": self.staging_bytes,
            "component_count": self.component_count,
            "pytree_leaf_count": self.pytree_leaf_count,
            "graph_constant_bytes": self.graph_constant_bytes,
            "graph_operand_count": self.graph_operand_count,
            "stablehlo_operation_count": self.stablehlo_operation_count,
            "xla_buffer_assignment_total_bytes": self.xla_buffer_assignment_total_bytes,
            "logical_collective_bytes": self.logical_collective_bytes,
            "graph_bytes_by_device_count": self.graph_bytes_by_device_count,
            "graph_bytes_by_device_max": self.graph_bytes_by_device_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PerformanceMetrics":
        if not isinstance(payload, dict):
            raise ValueError("metric payload must be an object")
        return cls(
            canonical_graph_bytes=none_or_int(payload, "canonical_graph_bytes"),
            padded_graph_bytes=none_or_int(payload, "padded_graph_bytes"),
            descriptor_bytes=none_or_int(payload, "descriptor_bytes"),
            resident_graph_bytes=none_or_int(payload, "resident_graph_bytes"),
            max_device_graph_bytes=none_or_int(payload, "max_device_graph_bytes"),
            staging_bytes=none_or_int(payload, "staging_bytes"),
            component_count=none_or_int(payload, "component_count"),
            pytree_leaf_count=none_or_int(payload, "pytree_leaf_count"),
            graph_constant_bytes=none_or_int(payload, "graph_constant_bytes"),
            graph_operand_count=none_or_int(payload, "graph_operand_count"),
            stablehlo_operation_count=none_or_int(payload, "stablehlo_operation_count"),
            xla_buffer_assignment_total_bytes=none_or_int(payload, "xla_buffer_assignment_total_bytes"),
            logical_collective_bytes=none_or_int(payload, "logical_collective_bytes"),
            graph_bytes_by_device_count=none_or_int(payload, "graph_bytes_by_device_count"),
            graph_bytes_by_device_max=none_or_int(payload, "graph_bytes_by_device_max"),
        )


@dataclass(frozen=True)
class BenchmarkRecord:
    record_id: str
    platform_label: str
    cache_policy: str
    candidate_commit: str
    behavioral_reference_commit: str
    dirty_worktree: bool
    dataset: DatasetFingerprint
    representation: str
    operation: str
    phase: str
    workload_size: int | None
    dtype: str
    requested_backend: str | None
    resolved_backend: str | None
    device_count: int
    timed: TimedPhase
    metric: PerformanceMetrics
    numeric_passed: bool = True
    status: str = "pass"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if not self.platform_label:
            raise ValueError("platform_label must be non-empty")
        if self.behavioral_reference_commit != CURRENT_REFERENCE_COMMIT:
            raise ValueError(
                f"behavioral_reference_commit mismatch: {self.behavioral_reference_commit!r} != "
                f"{CURRENT_REFERENCE_COMMIT!r}"
            )
        if self.cache_policy not in {item.value for item in CachePolicy}:
            raise ValueError(f"unknown cache policy {self.cache_policy!r}")
        if self.representation not in {item.value for item in Representation}:
            raise ValueError(f"unknown representation {self.representation!r}")
        if self.phase not in KNOWN_PHASES:
            raise ValueError(f"unknown phase {self.phase!r}")
        if self.workload_size is not None and self.workload_size < 1:
            raise ValueError("workload_size must be >= 1")
        if self.device_count < 1:
            raise ValueError("device_count must be >= 1")
        if self.dtype not in KNOWN_DTYPES:
            raise ValueError(f"unsupported dtype label {self.dtype!r}")
        if self.status not in {"pass", "fail", "skip"}:
            raise ValueError("status must be one of pass/fail/skip")
        if not isinstance(self.numeric_passed, bool):
            raise ValueError("numeric_passed must be a bool")
        if self.behavioral_reference_commit != CURRENT_REFERENCE_COMMIT:
            raise ValueError("behavioral_reference_commit mismatch")

    @property
    def is_warm(self) -> bool:
        return self.phase == TimingPhase.WARM_EXECUTION.value

    @property
    def is_pairable(self) -> bool:
        if not self.is_warm:
            return False
        if self.representation != Representation.PACKED_CANDIDATE.value:
            return False
        if self.operation not in REQUIRED_PRODUCT_OPERATIONS:
            return False
        if self.workload_size is None:
            return False
        if self.workload_size not in REQUIRED_PRODUCT_KS:
            return False
        if self.status != "pass":
            return False
        if not self.timed.seconds:
            return False
        return True

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.platform_label,
            self.cache_policy,
            self.representation,
            self.operation,
            self.workload_size,
            self.dtype,
            self.requested_backend,
            self.resolved_backend,
            self.device_count,
            self.timed.phase if isinstance(self.timed, TimedPhase) else self.phase,
            self.dataset.sha256,
        )

    @property
    def warm_key(self) -> tuple[Any, ...]:
        return (
            self.platform_label,
            self.cache_policy,
            self.operation,
            self.workload_size,
            self.dtype,
            self.requested_backend,
            self.resolved_backend,
            self.device_count,
            self.dataset.sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "platform_label": self.platform_label,
            "cache_policy": self.cache_policy,
            "candidate_commit": self.candidate_commit,
            "behavioral_reference_commit": self.behavioral_reference_commit,
            "dirty_worktree": self.dirty_worktree,
            "dataset": self.dataset.to_dict(),
            "representation": self.representation,
            "operation": self.operation,
            "phase": self.phase,
            "workload_size": self.workload_size,
            "dtype": self.dtype,
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "device_count": self.device_count,
            "timed": self.timed.to_dict(),
            "metric": self.metric.to_dict(),
            "numeric_passed": self.numeric_passed,
            "status": self.status,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkRecord":
        required = {
            "record_id",
            "platform_label",
            "cache_policy",
            "candidate_commit",
            "behavioral_reference_commit",
            "dirty_worktree",
            "dataset",
            "representation",
            "operation",
            "phase",
            "workload_size",
            "dtype",
            "requested_backend",
            "resolved_backend",
            "device_count",
            "timed",
            "metric",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"benchmark row missing required fields: {sorted(missing)}")
        return cls(
            record_id=str(payload["record_id"]),
            platform_label=str(payload["platform_label"]),
            cache_policy=str(payload["cache_policy"]),
            candidate_commit=str(payload["candidate_commit"]),
            behavioral_reference_commit=str(payload["behavioral_reference_commit"]),
            dirty_worktree=bool(payload["dirty_worktree"]),
            dataset=DatasetFingerprint.from_dict(payload["dataset"]),
            representation=str(payload["representation"]),
            operation=str(payload["operation"]),
            phase=str(payload["phase"]),
            workload_size=None if payload["workload_size"] is None else int(payload["workload_size"]),
            dtype=str(payload["dtype"]),
            requested_backend=None if payload.get("requested_backend") is None else str(payload["requested_backend"]),
            resolved_backend=None if payload.get("resolved_backend") is None else str(payload["resolved_backend"]),
            device_count=int(payload["device_count"]),
            timed=TimedPhase.from_dict(payload["timed"]),
            metric=PerformanceMetrics.from_dict(payload["metric"]),
            numeric_passed=bool(payload.get("numeric_passed", True)),
            status=str(payload.get("status", "pass")),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class PromotionEvidence:
    schema_version: str
    candidate_commit: str
    dirty_worktree: bool
    behavioral_reference_commit: str
    dataset: DatasetFingerprint
    produced_at_utc: str
    cache_label: str
    environment: EnvironmentState
    records: tuple[BenchmarkRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version not in KNOWN_SCHEMA_VERSIONS:
            raise ValueError(f"unknown schema version: {self.schema_version!r}")
        if self.behavioral_reference_commit != CURRENT_REFERENCE_COMMIT:
            raise ValueError("behavioral_reference_commit mismatch")
        if self.cache_label not in {item.value for item in CachePolicy}:
            raise ValueError("cache_label must be one of 'fresh' or 'reused'")
        datetime.fromisoformat(self.produced_at_utc)
        seen: set[str] = set()
        for record in self.records:
            if record.candidate_commit != self.candidate_commit:
                raise ValueError("row candidate_commit mismatch")
            if record.dataset != self.dataset:
                raise ValueError("row dataset mismatch")
            if record.dirty_worktree != self.dirty_worktree:
                raise ValueError("row dirty_worktree mismatch")
            if record.record_id in seen:
                raise ValueError(f"duplicate record_id {record.record_id!r}")
            seen.add(record.record_id)

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def has_records(self) -> bool:
        return bool(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_commit": self.candidate_commit,
            "dirty_worktree": self.dirty_worktree,
            "behavioral_reference_commit": self.behavioral_reference_commit,
            "dataset": self.dataset.to_dict(),
            "produced_at_utc": self.produced_at_utc,
            "cache_label": self.cache_label,
            "environment": self.environment.to_dict(),
            "records": [item.to_dict() for item in self.records],
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        allow_repo_mismatch: bool = False,
        repo_root: Path | None = None,
    ) -> "PromotionEvidence":
        required = {
            "schema_version",
            "candidate_commit",
            "dirty_worktree",
            "behavioral_reference_commit",
            "dataset",
            "produced_at_utc",
            "cache_label",
            "environment",
            "records",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"evidence missing required fields: {sorted(missing)}")
        if not isinstance(payload.get("records"), list):
            raise ValueError("records must be a list")

        records = tuple(BenchmarkRecord.from_dict(item) for item in payload["records"])
        evidence = cls(
            schema_version=str(payload["schema_version"]),
            candidate_commit=str(payload["candidate_commit"]),
            dirty_worktree=bool(payload["dirty_worktree"]),
            behavioral_reference_commit=str(payload["behavioral_reference_commit"]),
            dataset=DatasetFingerprint.from_dict(payload["dataset"]),
            produced_at_utc=str(payload["produced_at_utc"]),
            cache_label=str(payload["cache_label"]),
            environment=EnvironmentState.from_dict(payload["environment"]),
            records=records,
        )
        if allow_repo_mismatch:
            return evidence

        root = _repo_root(repo_root)
        if evidence.candidate_commit != git_commit(root):
            raise ValueError("candidate_commit mismatch for local evidence")
        if evidence.dirty_worktree != is_git_dirty(root):
            raise ValueError("dirty_worktree mismatch for local evidence")
        return evidence

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(
        cls,
        payload: str | Path,
        *,
        allow_repo_mismatch: bool = False,
        repo_root: Path | None = None,
    ) -> "PromotionEvidence":
        if isinstance(payload, Path):
            payload = payload.read_text(encoding="utf-8")
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("evidence JSON must contain an object")
        return cls.from_dict(data, allow_repo_mismatch=allow_repo_mismatch, repo_root=repo_root)


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    reason: str


@dataclass(frozen=True)
class PromotionDecision:
    decision: Decision
    gates: tuple[GateResult, ...]
    blocker_count: int


def compute_dataset_fingerprint(h5_path: Path | str) -> DatasetFingerprint:
    h5_path = _normalize_path(h5_path)
    if not h5_path.is_file():
        raise ValueError(f"missing h5 file: {h5_path}")

    hasher = hashlib.sha256()
    with h5_path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)

    metadata = list_blocks(h5_path)
    if metadata is None:
        raise ValueError(f"could not read block metadata from {h5_path}")
    if metadata.height == 0:
        raise ValueError(f"h5 metadata has no blocks: {h5_path}")

    if "n_samples" in metadata.columns:
        n_samples = int(metadata.get_column("n_samples")[0])
    elif "num_samples" in metadata.columns:
        n_samples = int(metadata.get_column("num_samples")[0])
    else:
        raise ValueError(f"missing sample-count metadata column in {h5_path}")

    if "n_variants" in metadata.columns:
        n_variants = int(metadata.get_column("n_variants").sum())
    elif "num_variants" in metadata.columns:
        n_variants = int(metadata.get_column("num_variants").sum())
    else:
        raise ValueError(f"missing variant-count metadata column in {h5_path}")

    if n_samples <= 0 or n_variants <= 0:
        raise ValueError(f"metadata has non-positive dimensions for {h5_path}")

    return DatasetFingerprint(
        sha256=hasher.hexdigest(),
        size_bytes=int(h5_path.stat().st_size),
        block_count=int(metadata.height),
        n_samples=n_samples,
        n_variants=n_variants,
    )


def gather_environment(platform_label: str) -> EnvironmentState:
    return EnvironmentState(
        platform_label=platform_label,
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__,
        os_name=os.name,
        machine=platform.machine(),
        architecture=platform.platform(),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        devices=tuple(str(device.platform) for device in jax.devices()),
        xla_cache_dir=os.environ.get("XLA_CACHE_DIR"),
        command=" ".join(shlex.quote(item) for item in sys.argv),
        dirty_worktree=is_git_dirty(),
    )


def _build_record_id(
    *,
    platform_label: str,
    cache_label: str,
    representation: str,
    operation: str,
    phase: str,
    workload_size: int | None,
    dtype: str,
    device_count: int,
    requested_backend: str | None,
    resolved_backend: str | None,
    dataset_sha256: str,
) -> str:
    return "|".join(
        [
            platform_label,
            cache_label,
            representation,
            operation,
            phase,
            "na" if workload_size is None else str(workload_size),
            dtype,
            str(device_count),
            requested_backend or "na",
            resolved_backend or "na",
            dataset_sha256,
        ]
    )


def make_record(
    *,
    platform_label: str,
    cache_label: str,
    candidate_commit: str,
    dataset: DatasetFingerprint,
    representation: str,
    operation: str,
    phase: str,
    workload_size: int | None,
    dtype: str,
    requested_backend: str | None,
    resolved_backend: str | None,
    device_count: int,
    timed: TimedPhase,
    metric: PerformanceMetrics | None = None,
    numeric_passed: bool = True,
    status: str = "pass",
    notes: str = "",
    dirty_worktree: bool | None = None,
) -> BenchmarkRecord:
    if dirty_worktree is None:
        dirty_worktree = is_git_dirty()
    return BenchmarkRecord(
        record_id=_build_record_id(
            platform_label=platform_label,
            cache_label=cache_label,
            representation=representation,
            operation=operation,
            phase=phase,
            workload_size=workload_size,
            dtype=dtype,
            device_count=device_count,
            requested_backend=requested_backend,
            resolved_backend=resolved_backend,
            dataset_sha256=dataset.sha256,
        ),
        platform_label=platform_label,
        cache_policy=cache_label,
        candidate_commit=candidate_commit,
        behavioral_reference_commit=CURRENT_REFERENCE_COMMIT,
        dirty_worktree=dirty_worktree,
        dataset=dataset,
        representation=representation,
        operation=operation,
        phase=phase,
        workload_size=workload_size,
        dtype=dtype,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        device_count=device_count,
        timed=timed,
        metric=metric or PerformanceMetrics(),
        numeric_passed=numeric_passed,
        status=status,
        notes=notes,
    )


def build_promotion_evidence(
    *,
    cache_label: str,
    platform_label: str,
    records: tuple[BenchmarkRecord, ...],
    candidate_commit: str | None = None,
    dataset: DatasetFingerprint | None = None,
    produced_at_utc: str | None = None,
) -> PromotionEvidence:
    if cache_label not in {item.value for item in CachePolicy}:
        raise ValueError(f"unknown cache policy {cache_label!r}")
    if not records:
        raise ValueError("promotion evidence requires at least one row")
    if dataset is None:
        first = records[0].dataset
        if any(record.dataset != first for record in records):
            raise ValueError("records must share the same dataset fingerprint")
        dataset = first
    candidate = candidate_commit or records[0].candidate_commit
    if any(record.candidate_commit != candidate for record in records):
        raise ValueError("records must share one candidate commit")
    produced = produced_at_utc or datetime.now(timezone.utc).isoformat()
    if dataset is None:
        raise ValueError("dataset must be provided")
    return PromotionEvidence(
        schema_version=SCHEMA_VERSION,
        candidate_commit=candidate,
        dirty_worktree=records[0].dirty_worktree,
        behavioral_reference_commit=CURRENT_REFERENCE_COMMIT,
        dataset=dataset,
        produced_at_utc=produced,
        cache_label=cache_label,
        environment=gather_environment(platform_label),
        records=tuple(records),
    )


def _exact_warm_lookup(rows: Iterable[BenchmarkRecord], *, row: BenchmarkRecord) -> BenchmarkRecord | None:
    for candidate in rows:
        if candidate.phase != TimingPhase.WARM_EXECUTION.value:
            continue
        if candidate.operation != row.operation:
            continue
        if candidate.workload_size != row.workload_size:
            continue
        if candidate.dtype != row.dtype:
            continue
        if candidate.requested_backend != row.requested_backend:
            continue
        if candidate.resolved_backend != row.resolved_backend:
            continue
        if candidate.device_count != row.device_count:
            continue
        if candidate.cache_policy != row.cache_policy:
            continue
        if candidate.platform_label != row.platform_label:
            continue
        if candidate.dataset.sha256 != row.dataset.sha256:
            continue
        if candidate.status != "pass":
            continue
        if candidate.timed.seconds is None or candidate.timed.seconds <= 0:
            continue
        if candidate.representation == Representation.RETAINED_EXACT_RAGGED.value:
            return candidate
    return None


def evaluate_ratio_gates(
    evidence: PromotionEvidence,
    *,
    required_ratio: float = 1.20,
) -> tuple[Decision, tuple[GateResult, ...], int]:
    rows = evidence.records
    packed_rows = [row for row in rows if row.representation == Representation.PACKED_CANDIDATE.value]
    gate_results: list[GateResult] = []
    blocker_count = 0

    if evidence.dirty_worktree:
        return (
            Decision.REJECT,
            (
                GateResult(
                    gate="clean_checkout",
                    status=GateStatus.FAIL,
                    reason=GateFailureReason.DIRTY_WORKTREE.value,
                ),
            ),
            1,
        )

    for packed in packed_rows:
        if packed.status != "pass":
            continue
        if not packed.is_warm:
            continue
        exact = _exact_warm_lookup(rows, row=packed)
        if exact is None:
            gate_results.append(
                GateResult(
                    gate=f"ratio_{packed.operation}_k{packed.workload_size}",
                    status=GateStatus.MISSING,
                    reason=f"{GateFailureReason.MISSING_EXACT.value} for packed warm measurement",
                )
            )
            blocker_count = max(1, blocker_count)
            continue
        if not packed.numeric_passed:
            gate_results.append(
                GateResult(
                    gate=f"ratio_{packed.operation}_k{packed.workload_size}",
                    status=GateStatus.FAIL,
                    reason="packed numerical checks failed for this workload",
                )
            )
            blocker_count += 1
            continue
        assert packed.timed.seconds is not None and exact.timed.seconds is not None
        ratio = packed.timed.seconds / exact.timed.seconds
        status = GateStatus.PASS if ratio <= required_ratio else GateStatus.FAIL
        gate_results.append(
            GateResult(
                gate=f"ratio_{packed.operation}_k{packed.workload_size}",
                status=status,
                reason=(
                    f"packed warm / retained-exact warm ratio={ratio:.6f}, threshold={required_ratio:.2f}"
                    if status == GateStatus.PASS
                    else f"packed warm / retained-exact warm ratio {ratio:.6f} exceeds {required_ratio:.2f}"
                ),
            )
        )
        if status == GateStatus.FAIL:
            blocker_count += 1

    for operation in REQUIRED_PRODUCT_OPERATIONS:
        for k in REQUIRED_PRODUCT_KS:
            has_packed = any(
                row.representation == Representation.PACKED_CANDIDATE.value
                and row.operation == operation
                and row.phase == TimingPhase.WARM_EXECUTION.value
                and row.workload_size == k
                and row.status == "pass"
                and row.timed.seconds is not None
                and row.timed.seconds > 0
                for row in rows
            )
            if not has_packed:
                continue
            has_exact = any(
                row.representation == Representation.RETAINED_EXACT_RAGGED.value
                and row.operation == operation
                and row.phase == TimingPhase.WARM_EXECUTION.value
                and row.workload_size == k
                and row.status == "pass"
                and row.timed.seconds is not None
                and row.timed.seconds > 0
                for row in rows
            )
            if not has_exact:
                gate_results.append(
                    GateResult(
                        gate=f"ratio_{operation}_k{k}",
                        status=GateStatus.MISSING,
                        reason=GateFailureReason.MISSING_EXACT.value,
                    )
                )
                blocker_count = max(1, blocker_count)

    if blocker_count:
        return Decision.CONTINUE_COEXISTENCE, tuple(gate_results), blocker_count
    return Decision.PROMOTE, tuple(gate_results), 0


def evaluate_evidence_set(
    evidences: tuple[PromotionEvidence, ...],
    *,
    required_ratio: float = 1.20,
) -> PromotionDecision:
    if not evidences:
        return PromotionDecision(
            decision=Decision.REJECT,
            gates=(
                GateResult(
                    gate="evidence_set",
                    status=GateStatus.MISSING,
                    reason=GateFailureReason.MISSING_EVIDENCE.value,
                ),
            ),
            blocker_count=1,
        )

    candidate_commits = {ev.candidate_commit for ev in evidences}
    if len(candidate_commits) != 1:
        return PromotionDecision(
            decision=Decision.REJECT,
            gates=(
                GateResult(
                    gate="candidate_commit",
                    status=GateStatus.FAIL,
                    reason=GateFailureReason.CANDIDATE_MISMATCH.value,
                ),
            ),
            blocker_count=1,
        )

    all_gates: list[GateResult] = []
    blocker_count = 0
    for evidence in evidences:
        decision, gates, local_blockers = evaluate_ratio_gates(evidence, required_ratio=required_ratio)
        all_gates.extend(gates)
        blocker_count += local_blockers
        if decision == Decision.REJECT:
            return PromotionDecision(
                decision=Decision.REJECT,
                gates=tuple(all_gates),
                blocker_count=max(1, blocker_count),
            )

    if blocker_count > 0:
        return PromotionDecision(
            decision=Decision.CONTINUE_COEXISTENCE,
            gates=tuple(all_gates),
            blocker_count=blocker_count,
        )
    return PromotionDecision(
        decision=Decision.PROMOTE,
        gates=tuple(all_gates),
        blocker_count=0,
    )


def format_markdown_decision(decision: PromotionDecision) -> str:
    rows = [
        "| gate | status | reason |",
        "|---|---|---|",
    ]
    for gate in decision.gates:
        rows.append(f"| {gate.gate} | {gate.status.value} | {gate.reason} |")
    header = f"Decision: {decision.decision.value}\\nBlockers: {decision.blocker_count}"
    if not rows:
        return f"{header}\\n\\n(no gates)"
    return f"{header}\\n\\n" + "\\n".join(rows)


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = [path] if path.is_file() else sorted(path.glob("*.json"))
    for entry in paths:
        payload = json.loads(entry.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows.extend(payload)
        elif isinstance(payload, dict):
            rows.append(payload)
        else:
            raise ValueError(f"unsupported json payload in {entry}")
    return rows


def load_evidences(raw_outputs: Path | str) -> list[PromotionEvidence]:
    path = _normalize_path(raw_outputs)
    payloads = _read_table_rows(path)
    return [PromotionEvidence.from_dict(item, allow_repo_mismatch=True) for item in payloads]


def build_promotion_pytest_command(
    *,
    module: str,
    repo_root: Path,
    h5_path: Path,
    output_path: Path,
    platform_label: str,
    cache_policy: str,
    linarg_benchmark_k: tuple[int, ...],
    rhe_benchmark_num_matvecs: tuple[int, ...],
    enforce_gates: bool = False,
) -> list[str]:
    if not module:
        raise ValueError("module must be non-empty")
    if not isinstance(repo_root, Path):
        repo_root = _normalize_path(repo_root)
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must exist: {repo_root}")
    if not h5_path.is_file():
        raise ValueError(f"h5 path must exist: {h5_path}")
    if platform_label is None or not platform_label:
        raise ValueError("platform_label must be non-empty")
    if cache_policy not in {item.value for item in CachePolicy}:
        raise ValueError(f"unknown cache policy {cache_policy!r}")
    if not linarg_benchmark_k:
        raise ValueError("linarg_benchmark_k must be non-empty")
    if any(value < 1 for value in linarg_benchmark_k):
        raise ValueError("linarg_benchmark_k values must be positive")
    if not rhe_benchmark_num_matvecs:
        raise ValueError("rhe_benchmark_num_matvecs must be non-empty")
    if any(value < 1 for value in rhe_benchmark_num_matvecs):
        raise ValueError("rhe_benchmark_num_matvecs values must be positive")

    return [
        "uv",
        "run",
        "pytest",
        "-p",
        "no:capture",
        "--runbench",
        "--linarg-h5-path",
        str(h5_path),
        "--linarg-parallel-processes",
        str(2),
        "--linarg-benchmark-k",
        *[str(item) for item in linarg_benchmark_k],
        "--rhe-benchmark-num-matvecs",
        *[str(item) for item in rhe_benchmark_num_matvecs],
        "--jax-promotion-output",
        str(output_path),
        "--cache-policy",
        cache_policy,
        "--platform-label",
        platform_label,
        module,
    ] + (["--jax-enforce-promotion-gates"] if enforce_gates else [])


def normalize_command(path: Path | str) -> str:
    path = _normalize_path(path)
    return f"pytest {path}"
