# pattern: Functional Core

from __future__ import annotations

import json

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal, overload

SCHEMA_VERSION = "2026-08-13+3"
LEGACY_SCHEMA_VERSION = "2026-08-13+2"
CURRENT_REFERENCE_COMMIT = "b68e7da"
REQUIRED_PRODUCT_KS = (4, 20)
REQUIRED_PRODUCT_OPERATIONS = ("matmat", "rmatmat")
REQUIRED_PLATFORM_LABELS = ("arm64-cpu", "x86_64-cpu", "forced-two-device-cpu", "gpu")
KNOWN_SCHEMA_VERSIONS = {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}
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


def _require_bool(payload: dict[str, Any], key: str, *, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a bool")
    return value


@overload
def _require_int(value: Any, *, field: str, allow_none: Literal[False] = False) -> int: ...


@overload
def _require_int(value: Any, *, field: str, allow_none: Literal[True]) -> int | None: ...


def _require_int(value: Any, *, field: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    return value


def none_or_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return _require_int(value, field=key, allow_none=True)


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
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("sha256 must contain only hexadecimal characters") from error
        for field_name, value in (
            ("size_bytes", self.size_bytes),
            ("block_count", self.block_count),
            ("n_samples", self.n_samples),
            ("n_variants", self.n_variants),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
            size_bytes=_require_int(payload["size_bytes"], field="size_bytes"),
            block_count=_require_int(payload["block_count"], field="block_count"),
            n_samples=_require_int(payload["n_samples"], field="n_samples"),
            n_variants=_require_int(payload["n_variants"], field="n_variants"),
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
    device_platforms: tuple[str, ...] = ()
    cache_policy: str | None = None

    def __post_init__(self) -> None:
        if not self.platform_label:
            raise ValueError("platform_label must be non-empty")
        if not self.python_version:
            raise ValueError("python_version must be non-empty")
        if not self.command:
            raise ValueError("command must be non-empty")
        if not isinstance(self.dirty_worktree, bool):
            raise ValueError("environment.dirty_worktree must be a bool")
        if self.cache_policy is not None and self.cache_policy not in {item.value for item in CachePolicy}:
            raise ValueError("environment.cache_policy must be fresh, reused, or null")
        if not self.device_platforms:
            inferred = tuple(device.lower() for device in self.devices if device.lower() in {"cpu", "gpu", "tpu"})
            object.__setattr__(self, "device_platforms", inferred)
        if len(self.device_platforms) != len(self.devices):
            raise ValueError("environment device_platforms and devices must have equal length")
        if any(item not in {"cpu", "gpu", "tpu"} for item in self.device_platforms):
            raise ValueError("environment.device_platforms contains an unsupported JAX platform")
        _validate_platform_attestation(self)

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
            "device_platforms": list(self.device_platforms),
            "cache_policy": self.cache_policy,
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
        device_platforms = payload.get("device_platforms")
        if device_platforms is not None and not isinstance(device_platforms, list):
            raise ValueError("environment.device_platforms must be a list")
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
            device_platforms=tuple(str(item) for item in device_platforms or ()),
            cache_policy=None if payload.get("cache_policy") is None else str(payload["cache_policy"]),
            xla_cache_dir=None if payload["xla_cache_dir"] is None else str(payload["xla_cache_dir"]),
            command=str(payload["command"]),
            dirty_worktree=_require_bool(payload, "dirty_worktree", context="environment"),
        )


def normalize_machine(machine: str) -> str:
    normalized = machine.strip().lower().replace("-", "_")
    if normalized in {"arm64", "aarch64"}:
        return "arm64"
    if normalized in {"x86_64", "amd64", "x64"}:
        return "x86_64"
    return normalized


def _validate_platform_attestation(environment: EnvironmentState) -> None:
    label = environment.platform_label
    machine = normalize_machine(environment.machine)
    platforms = environment.device_platforms
    if label == "arm64-cpu" and machine != "arm64":
        raise ValueError(f"platform label arm64-cpu does not match normalized machine {machine!r}")
    if label == "x86_64-cpu" and machine != "x86_64":
        raise ValueError(f"platform label x86_64-cpu does not match normalized machine {machine!r}")
    if label in {"arm64-cpu", "x86_64-cpu", "forced-two-device-cpu"}:
        if not platforms or any(item != "cpu" for item in platforms):
            raise ValueError(f"platform label {label} requires actual JAX CPU devices")
    if label == "forced-two-device-cpu" and len(platforms) < 2:
        raise ValueError("platform label forced-two-device-cpu requires at least 2 actual JAX CPU devices")
    if label == "gpu" and "gpu" not in platforms:
        raise ValueError("platform label gpu requires an actual JAX GPU device")


def attested_platforms(environment: EnvironmentState) -> tuple[str, ...]:
    """Return required platform roles proven by normalized architecture/devices."""
    roles: list[str] = []
    machine = normalize_machine(environment.machine)
    if environment.device_platforms and all(item == "cpu" for item in environment.device_platforms):
        if machine == "arm64":
            roles.append("arm64-cpu")
        elif machine == "x86_64":
            roles.append("x86_64-cpu")
        if len(environment.device_platforms) >= 2:
            roles.append("forced-two-device-cpu")
    if "gpu" in environment.device_platforms:
        roles.append("gpu")
    return tuple(roles)


def environment_comparison_key(environment: EnvironmentState) -> tuple[Any, ...]:
    """Return normalized child context, excluding only process command text."""
    return (
        environment.platform_label,
        environment.python_version,
        environment.jax_version,
        environment.jaxlib_version,
        environment.numpy_version,
        environment.os_name,
        normalize_machine(environment.machine),
        environment.architecture,
        environment.xla_flags,
        environment.xla_cache_dir,
        environment.devices,
        environment.device_platforms,
        environment.cache_policy,
        environment.dirty_worktree,
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
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)):
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
        seconds = payload["seconds"]
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))):
            raise ValueError("timed phase seconds must be numeric or null")
        return cls(
            phase=str(payload["phase"]),
            seconds=None if seconds is None else float(seconds),
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
    stablehlo_bytes: int | None = None
    stablehlo_operation_count: int | None = None
    xla_buffer_assignment_total_bytes: int | None = None
    logical_collective_bytes: int | None = None
    graph_bytes_by_device_count: int | None = None
    graph_bytes_by_device_max: int | None = None

    def __post_init__(self) -> None:
        for key, value in self.__dict__.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
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
            "stablehlo_bytes": self.stablehlo_bytes,
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
            stablehlo_bytes=none_or_int(payload, "stablehlo_bytes"),
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
        if not isinstance(self.dirty_worktree, bool):
            raise ValueError("record.dirty_worktree must be a bool")
        if isinstance(self.workload_size, bool):
            raise ValueError("workload_size must be an int or null")
        if self.workload_size is not None and self.workload_size < 1:
            raise ValueError("workload_size must be >= 1")
        if isinstance(self.device_count, bool) or not isinstance(self.device_count, int) or self.device_count < 1:
            raise ValueError("device_count must be >= 1")
        if self.dtype not in KNOWN_DTYPES:
            raise ValueError(f"unsupported dtype label {self.dtype!r}")
        if self.status not in {"pass", "fail", "skip"}:
            raise ValueError("status must be one of pass/fail/skip")
        if not isinstance(self.numeric_passed, bool):
            raise ValueError("numeric_passed must be a bool")
        if self.timed.phase != self.phase:
            raise ValueError(f"timed phase {self.timed.phase!r} must equal record phase {self.phase!r}")
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
            dirty_worktree=_require_bool(payload, "dirty_worktree", context="record"),
            dataset=DatasetFingerprint.from_dict(payload["dataset"]),
            representation=str(payload["representation"]),
            operation=str(payload["operation"]),
            phase=str(payload["phase"]),
            workload_size=_require_int(payload["workload_size"], field="workload_size", allow_none=True),
            dtype=str(payload["dtype"]),
            requested_backend=None if payload.get("requested_backend") is None else str(payload["requested_backend"]),
            resolved_backend=None if payload.get("resolved_backend") is None else str(payload["resolved_backend"]),
            device_count=_require_int(payload["device_count"], field="device_count"),
            timed=TimedPhase.from_dict(payload["timed"]),
            metric=PerformanceMetrics.from_dict(payload["metric"]),
            numeric_passed=(
                True if "numeric_passed" not in payload else _require_bool(payload, "numeric_passed", context="record")
            ),
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
        if not isinstance(self.dirty_worktree, bool):
            raise ValueError("evidence.dirty_worktree must be a bool")
        datetime.fromisoformat(self.produced_at_utc)
        if self.environment.dirty_worktree != self.dirty_worktree:
            raise ValueError("environment dirty_worktree mismatch")
        if self.environment.cache_policy is not None and self.environment.cache_policy != self.cache_label:
            raise ValueError("environment cache_policy mismatch")
        seen: set[str] = set()
        for record in self.records:
            if record.candidate_commit != self.candidate_commit:
                raise ValueError("row candidate_commit mismatch")
            if record.dataset != self.dataset:
                raise ValueError("row dataset mismatch")
            if record.dirty_worktree != self.dirty_worktree:
                raise ValueError("row dirty_worktree mismatch")
            if record.cache_policy != self.cache_label:
                raise ValueError("row cache_policy mismatch")
            if record.platform_label != self.environment.platform_label:
                raise ValueError("row platform_label mismatch")
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
        if payload["schema_version"] == SCHEMA_VERSION:
            missing_environment = {"device_platforms", "cache_policy"} - set(payload["environment"])
            if missing_environment:
                raise ValueError(f"environment missing required fields: {sorted(missing_environment)}")

        records = tuple(BenchmarkRecord.from_dict(item) for item in payload["records"])
        evidence = cls(
            schema_version=str(payload["schema_version"]),
            candidate_commit=str(payload["candidate_commit"]),
            dirty_worktree=_require_bool(payload, "dirty_worktree", context="evidence"),
            behavioral_reference_commit=str(payload["behavioral_reference_commit"]),
            dataset=DatasetFingerprint.from_dict(payload["dataset"]),
            produced_at_utc=str(payload["produced_at_utc"]),
            cache_label=str(payload["cache_label"]),
            environment=EnvironmentState.from_dict(payload["environment"]),
            records=records,
        )
        return evidence

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(
        cls,
        payload: str,
    ) -> "PromotionEvidence":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("evidence JSON must contain an object")
        return cls.from_dict(data)


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    reason: str
    blocking: bool = True


@dataclass(frozen=True)
class PromotionDecision:
    decision: Decision
    gates: tuple[GateResult, ...]
    blocker_count: int


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
    dirty_worktree: bool = False,
) -> BenchmarkRecord:
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


@dataclass(frozen=True)
class ProductGateKey:
    platform: str
    cache_policy: str
    operation: str
    workload_size: int
    backend: str
    dtype: str
    device_count: int

    @property
    def gate_name(self) -> str:
        return (
            "product_ratio["
            f"platform={self.platform},cache={self.cache_policy},operation={self.operation},"
            f"k={self.workload_size},backend={self.backend},dtype={self.dtype},devices={self.device_count}]"
        )


def required_product_gate_keys() -> tuple[ProductGateKey, ...]:
    """Return the explicit AC8.3 promotion matrix."""
    platform_configs = (
        ("arm64-cpu", 1, ("pure_jax", "ffi_cpu")),
        ("x86_64-cpu", 1, ("pure_jax", "ffi_cpu")),
        ("forced-two-device-cpu", 2, ("pure_jax", "ffi_cpu")),
        ("gpu", 1, ("pure_jax",)),
    )
    return tuple(
        ProductGateKey(platform, cache.value, operation, k, backend, "float32", device_count)
        for platform, device_count, backends in platform_configs
        for cache in CachePolicy
        for operation in REQUIRED_PRODUCT_OPERATIONS
        for k in REQUIRED_PRODUCT_KS
        for backend in backends
    )


def _record_matches_product_key(record: BenchmarkRecord, key: ProductGateKey, *, representation: str) -> bool:
    return (
        record.representation == representation
        and record.operation == key.operation
        and record.phase == TimingPhase.WARM_EXECUTION.value
        and record.workload_size == key.workload_size
        and record.dtype == key.dtype
        and record.requested_backend == key.backend
        and record.resolved_backend == key.backend
        and record.device_count == key.device_count
        and record.cache_policy == key.cache_policy
        and record.status == "pass"
        and record.timed.seconds is not None
        and record.timed.seconds > 0
    )


def _evaluate_product_key(
    rows: tuple[BenchmarkRecord, ...],
    key: ProductGateKey,
    *,
    required_ratio: float,
) -> GateResult:
    packed = tuple(
        row
        for row in rows
        if _record_matches_product_key(row, key, representation=Representation.PACKED_CANDIDATE.value)
    )
    exact = tuple(
        row
        for row in rows
        if _record_matches_product_key(row, key, representation=Representation.RETAINED_EXACT_RAGGED.value)
    )
    if not packed or not exact:
        missing = []
        if not packed:
            missing.append("packed candidate")
        if not exact:
            missing.append("retained exact baseline")
        return GateResult(
            gate=key.gate_name,
            status=GateStatus.MISSING,
            reason=f"{GateFailureReason.MISSING_EVIDENCE.value}: missing {' and '.join(missing)}",
        )
    if len(packed) != 1 or len(exact) != 1:
        return GateResult(
            gate=key.gate_name,
            status=GateStatus.FAIL,
            reason=f"ambiguous evidence: packed={len(packed)}, retained_exact={len(exact)}",
        )
    packed_row, exact_row = packed[0], exact[0]
    if not packed_row.numeric_passed or not exact_row.numeric_passed:
        return GateResult(
            gate=key.gate_name,
            status=GateStatus.FAIL,
            reason="product numerical parity failed",
        )
    assert packed_row.timed.seconds is not None and exact_row.timed.seconds is not None
    ratio = packed_row.timed.seconds / exact_row.timed.seconds
    status = GateStatus.PASS if ratio <= required_ratio else GateStatus.FAIL
    return GateResult(
        gate=key.gate_name,
        status=status,
        reason=(
            f"packed warm / retained-exact warm ratio={ratio:.6f}, threshold={required_ratio:.2f}"
            if status is GateStatus.PASS
            else f"packed warm / retained-exact warm ratio {ratio:.6f} exceeds {required_ratio:.2f}"
        ),
    )


def _rhe_diagnostic_gates(rows: tuple[BenchmarkRecord, ...], *, required_ratio: float) -> tuple[GateResult, ...]:
    gates = []
    for packed in rows:
        if (
            packed.representation != Representation.PACKED_CANDIDATE.value
            or packed.operation != "rhe"
            or not packed.is_warm
            or packed.status != "pass"
        ):
            continue
        exact = _exact_warm_lookup(rows, row=packed)
        gate_name = (
            "rhe_diagnostic["
            f"platform={packed.platform_label},cache={packed.cache_policy},k={packed.workload_size},"
            f"backend={packed.resolved_backend},dtype={packed.dtype},devices={packed.device_count}]"
        )
        if exact is None:
            gates.append(
                GateResult(
                    gate=gate_name,
                    status=GateStatus.MISSING,
                    reason=GateFailureReason.MISSING_EXACT.value,
                    blocking=False,
                )
            )
            continue
        assert packed.timed.seconds is not None and exact.timed.seconds is not None
        ratio = packed.timed.seconds / exact.timed.seconds
        gates.append(
            GateResult(
                gate=gate_name,
                status=GateStatus.PASS if ratio <= required_ratio else GateStatus.FAIL,
                reason=f"diagnostic packed warm / retained-exact warm ratio={ratio:.6f}; no AC8.3 RHE threshold",
                blocking=False,
            )
        )
    return tuple(gates)


def evaluate_ratio_gates(
    evidence: PromotionEvidence,
    *,
    required_ratio: float = 1.20,
) -> tuple[Decision, tuple[GateResult, ...], int]:
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

    platform_roles = attested_platforms(evidence.environment)
    if platform_roles:
        keys = tuple(
            key
            for key in required_product_gate_keys()
            if key.platform in platform_roles and key.cache_policy == evidence.cache_label
        )
    else:
        packed_configs = {
            (row.requested_backend, row.resolved_backend, row.dtype, row.device_count)
            for row in evidence.records
            if row.representation == Representation.PACKED_CANDIDATE.value
            and row.operation in REQUIRED_PRODUCT_OPERATIONS
            and row.phase == TimingPhase.WARM_EXECUTION.value
        }
        keys = tuple(
            ProductGateKey(
                evidence.environment.platform_label,
                evidence.cache_label,
                operation,
                k,
                str(resolved_backend),
                dtype,
                device_count,
            )
            for requested_backend, resolved_backend, dtype, device_count in sorted(packed_configs, key=str)
            if requested_backend == resolved_backend and resolved_backend is not None
            for operation in REQUIRED_PRODUCT_OPERATIONS
            for k in REQUIRED_PRODUCT_KS
        )
    if not keys:
        product_gates = (
            GateResult(
                gate="product_ratio_matrix",
                status=GateStatus.MISSING,
                reason=GateFailureReason.MISSING_EVIDENCE.value,
            ),
        )
    else:
        product_gates = tuple(
            _evaluate_product_key(evidence.records, key, required_ratio=required_ratio) for key in keys
        )
    diagnostics = _rhe_diagnostic_gates(evidence.records, required_ratio=required_ratio)
    gates = (*product_gates, *diagnostics)
    blocker_count = sum(gate.blocking and gate.status is not GateStatus.PASS for gate in gates)
    decision = Decision.PROMOTE if blocker_count == 0 else Decision.CONTINUE_COEXISTENCE
    return decision, gates, blocker_count


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

    datasets = {ev.dataset for ev in evidences}
    if len(datasets) != 1:
        return PromotionDecision(
            decision=Decision.REJECT,
            gates=(GateResult(gate="dataset", status=GateStatus.FAIL, reason="dataset fingerprint mismatch"),),
            blocker_count=1,
        )
    if any(ev.dirty_worktree for ev in evidences):
        return PromotionDecision(
            decision=Decision.REJECT,
            gates=(
                GateResult(
                    gate="clean_checkout",
                    status=GateStatus.FAIL,
                    reason=GateFailureReason.DIRTY_WORKTREE.value,
                ),
            ),
            blocker_count=1,
        )

    product_gates = []
    for key in required_product_gate_keys():
        rows = tuple(
            row
            for evidence in evidences
            if key.platform in attested_platforms(evidence.environment) and evidence.cache_label == key.cache_policy
            for row in evidence.records
        )
        product_gates.append(_evaluate_product_key(rows, key, required_ratio=required_ratio))
    diagnostics = tuple(
        gate
        for evidence in evidences
        for gate in _rhe_diagnostic_gates(evidence.records, required_ratio=required_ratio)
    )
    gates = (*product_gates, *diagnostics)
    blocker_count = sum(gate.blocking and gate.status is not GateStatus.PASS for gate in gates)
    return PromotionDecision(
        decision=Decision.PROMOTE if blocker_count == 0 else Decision.CONTINUE_COEXISTENCE,
        gates=gates,
        blocker_count=blocker_count,
    )


def format_markdown_decision(decision: PromotionDecision) -> str:
    rows = [
        "| gate | status | reason |",
        "|---|---|---|",
    ]
    for gate in decision.gates:
        rows.append(f"| {gate.gate} | {gate.status.value} | {gate.reason} |")
    header = f"Decision: {decision.decision.value}\nBlockers: {decision.blocker_count}"
    if not rows:
        return f"{header}\n\n(no gates)"
    return f"{header}\n\n" + "\n".join(rows)
