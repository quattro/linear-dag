# pattern: Imperative Shell

"""Filesystem, process, runtime, and persistence shell for promotion evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import jax
import jaxlib
import numpy as np

from linear_dag.core.jaxlinarg.build_config import show_build_config
from linear_dag.core.lineararg import list_blocks
from tests.jax.bench._promotion import (
    BenchmarkRecord,
    BuildConfiguration,
    CachePolicy,
    DatasetFingerprint,
    EnvironmentState,
    PromotionEvidence,
    SCHEMA_VERSION,
)

_T = TypeVar("_T")


def normalize_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def repo_root(path: Path | None = None) -> Path:
    base = path if path is not None else Path(__file__).resolve().parents[3]
    return Path(
        subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--show-toplevel"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )


def git_commit(repository: Path | str | None = None) -> str:
    root = repo_root(Path(repository) if repository is not None else None)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def is_git_dirty(repository: Path | str | None = None) -> bool:
    root = repo_root(Path(repository) if repository is not None else None)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    relevant = []
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        if line.startswith("?? ") and Path(path).suffix.lower() in {".h5", ".hdf5"}:
            continue
        relevant.append(line)
    return bool(relevant)


def compute_dataset_fingerprint(h5_path: Path | str) -> DatasetFingerprint:
    path = normalize_path(h5_path)
    if not path.is_file():
        raise ValueError(f"missing h5 file: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)

    metadata = list_blocks(path)
    if metadata is None:
        raise ValueError(f"could not read block metadata from {path}")
    if metadata.height == 0:
        raise ValueError(f"h5 metadata has no blocks: {path}")

    if "n_samples" in metadata.columns:
        n_samples = int(metadata.get_column("n_samples")[0])
    elif "num_samples" in metadata.columns:
        n_samples = int(metadata.get_column("num_samples")[0])
    else:
        raise ValueError(f"missing sample-count metadata column in {path}")

    if "n_variants" in metadata.columns:
        n_variants = int(metadata.get_column("n_variants").sum())
    elif "num_variants" in metadata.columns:
        n_variants = int(metadata.get_column("num_variants").sum())
    else:
        raise ValueError(f"missing variant-count metadata column in {path}")
    if n_samples <= 0 or n_variants <= 0:
        raise ValueError(f"metadata has non-positive dimensions for {path}")

    return DatasetFingerprint(
        sha256=hasher.hexdigest(),
        size_bytes=int(path.stat().st_size),
        block_count=int(metadata.height),
        n_samples=n_samples,
        n_variants=n_variants,
    )


def gather_environment(
    platform_label: str,
    *,
    cache_policy: str,
    requested_device_count: int,
) -> EnvironmentState:
    devices = tuple(jax.devices())
    if requested_device_count > len(devices):
        raise ValueError(f"requested {requested_device_count} JAX device(s), but only {len(devices)} are visible")
    selected = devices[:requested_device_count]
    native = show_build_config()
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
        devices=tuple(str(device) for device in devices),
        device_platforms=tuple(str(device.platform) for device in devices),
        xla_cache_dir=os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.environ.get("XLA_CACHE_DIR"),
        command=" ".join(shlex.quote(item) for item in sys.argv),
        dirty_worktree=is_git_dirty(),
        cache_policy=cache_policy,
        build_configuration=BuildConfiguration(
            backend=str(native["backend"]),
            ffi_cpu_built=native["ffi_cpu_built"],
            ffi_cpu_available=native["ffi_cpu_available"],
            ffi_cpu_exact_available=native["ffi_cpu_exact_available"],
            ffi_cpu_packed_available=native["ffi_cpu_packed_available"],
            ffi_cpu_blas_enabled=native["ffi_cpu_blas_enabled"],
            ffi_cpu_blas_backend=(
                None if native["ffi_cpu_blas_backend"] is None else str(native["ffi_cpu_blas_backend"])
            ),
            ffi_cpu_native_tuning=native["ffi_cpu_native_tuning"],
            ffi_cpu_error=(None if native["ffi_cpu_error"] is None else str(native["ffi_cpu_error"])),
            ffi_cpu_exact_error=(None if native["ffi_cpu_exact_error"] is None else str(native["ffi_cpu_exact_error"])),
            ffi_cpu_packed_error=(
                None if native["ffi_cpu_packed_error"] is None else str(native["ffi_cpu_packed_error"])
            ),
        ),
        requested_device_count=requested_device_count,
        selected_devices=tuple(str(device) for device in selected),
        selected_device_platforms=tuple(str(device.platform) for device in selected),
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
    device_counts = {record.device_count for record in records}
    if len(device_counts) != 1:
        raise ValueError(f"records must share one selected device count, observed {sorted(device_counts)}")
    return PromotionEvidence(
        schema_version=SCHEMA_VERSION,
        candidate_commit=candidate,
        dirty_worktree=records[0].dirty_worktree,
        behavioral_reference_commit=records[0].behavioral_reference_commit,
        dataset=dataset,
        produced_at_utc=produced_at_utc or datetime.now(timezone.utc).isoformat(),
        cache_label=cache_label,
        environment=gather_environment(
            platform_label,
            cache_policy=cache_label,
            requested_device_count=device_counts.pop(),
        ),
        records=records,
    )


def load_evidence(path: Path | str) -> PromotionEvidence:
    return PromotionEvidence.from_json(normalize_path(path).read_text(encoding="utf-8"))


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
    return [PromotionEvidence.from_dict(item) for item in _read_table_rows(normalize_path(raw_outputs))]


def write_evidence_fragment(path: Path | str, evidence: PromotionEvidence) -> None:
    resolved = normalize_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(evidence.to_json(), encoding="utf-8")


def resolve_requested_device_count(cli_count: int) -> int:
    """Require runner topology metadata to agree with the pytest device count."""
    raw = os.environ.get("LINEAR_DAG_PROMOTION_DEVICE_COUNT")
    if raw is None:
        return cli_count
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError("LINEAR_DAG_PROMOTION_DEVICE_COUNT must be a positive integer") from error
    if requested < 1:
        raise ValueError("LINEAR_DAG_PROMOTION_DEVICE_COUNT must be a positive integer")
    if requested != cli_count:
        raise ValueError(
            f"promotion topology mismatch: runner requested {requested} device(s), pytest requested {cli_count}"
        )
    return requested


def time_synchronized_construction(
    factory: Callable[[], _T],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[_T, float]:
    """Construct a JAX graph and synchronize every leaf inside its timing window."""
    start = clock()
    value = factory()

    def synchronize_leaf(leaf: Any) -> Any:
        block = getattr(leaf, "block_until_ready", None)
        return block() if block is not None else leaf

    value = jax.tree.map(synchronize_leaf, value)
    return value, clock() - start


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
    pytest_args: tuple[str, ...] | None = None,
    linarg_parallel_processes: int = 2,
) -> list[str]:
    if not module:
        raise ValueError("module must be non-empty")
    if not repo_root.is_dir():
        raise ValueError(f"repo_root must exist: {repo_root}")
    if not h5_path.is_file():
        raise ValueError(f"h5 path must exist: {h5_path}")
    if not platform_label:
        raise ValueError("platform_label must be non-empty")
    if cache_policy not in {item.value for item in CachePolicy}:
        raise ValueError(f"unknown cache policy {cache_policy!r}")
    if not linarg_benchmark_k or any(value < 1 for value in linarg_benchmark_k):
        raise ValueError("linarg_benchmark_k values must be non-empty and positive")
    if not rhe_benchmark_num_matvecs or any(value < 1 for value in rhe_benchmark_num_matvecs):
        raise ValueError("rhe_benchmark_num_matvecs values must be non-empty and positive")
    if linarg_parallel_processes < 1:
        raise ValueError("linarg_parallel_processes must be positive")

    args = [
        "uv",
        "run",
        "pytest",
        "-p",
        "no:capture",
        "--runbench",
        "--linarg-h5-path",
        str(h5_path),
        "--linarg-parallel-processes",
        str(linarg_parallel_processes),
        "--linarg-benchmark-k",
        *map(str, linarg_benchmark_k),
        "--rhe-benchmark-num-matvecs",
        *map(str, rhe_benchmark_num_matvecs),
        "--jax-promotion-output",
        str(output_path),
        "--cache-policy",
        cache_policy,
        "--platform-label",
        platform_label,
        module,
    ]
    if enforce_gates:
        args.append("--jax-enforce-promotion-gates")
    if pytest_args:
        args.extend(pytest_args)
    return args


def normalize_command(path: Path | str) -> str:
    return f"pytest {normalize_path(path)}"
