# pattern: Imperative Shell

"""Opt-in subprocess orchestration for matched JAX promotion benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest

from linear_dag.core.jaxlinarg.packing import PACKED_COMPONENT_NAMES
from tests.jax.bench._promotion import (
    attested_platforms,
    Decision,
    environment_comparison_key,
    evaluate_ratio_gates,
    EvidenceGateOutcome,
    expected_logical_collective_bytes,
    GateResult,
    GateStatus,
    PromotionEvidence,
    Representation,
    REQUIRED_PRODUCT_KS,
    TimingPhase,
)
from tests.jax.bench._promotion_io import build_promotion_pytest_command, load_evidence, write_evidence_fragment


@dataclass(frozen=True)
class ChildRun:
    """One representation-isolated benchmark subprocess."""

    name: str
    command: tuple[str, ...]
    output_path: Path


_CHILDREN = (
    ("product-packed", "tests/jax/bench/test_parallel_benchmarks.py::test_promotion_packed_product_child"),
    ("product-exact", "tests/jax/bench/test_parallel_benchmarks.py::test_promotion_exact_product_child"),
    (
        "product-numpy-cython",
        "tests/jax/bench/test_parallel_benchmarks.py::test_promotion_numpy_cython_product_child",
    ),
    ("rhe-packed", "tests/jax/bench/test_rhe_benchmarks.py::test_promotion_packed_rhe_child"),
    ("rhe-exact", "tests/jax/bench/test_rhe_benchmarks.py::test_promotion_exact_rhe_child"),
    ("rhe-numpy-cython", "tests/jax/bench/test_rhe_benchmarks.py::test_promotion_numpy_cython_rhe_child"),
)


def build_child_runs(
    *,
    repo_root: Path,
    h5_path: Path,
    fragment_dir: Path,
    platform_label: str,
    cache_policy: str,
    linarg_benchmark_k: tuple[int, ...],
    rhe_benchmark_num_matvecs: tuple[int, ...],
    parallel_processes: int,
) -> tuple[ChildRun, ...]:
    """Build representation-isolated pytest child commands with matched context."""
    if parallel_processes < 1:
        raise ValueError("parallel_processes must be at least 1")
    runs = []
    for name, nodeid in _CHILDREN:
        output_path = fragment_dir / f"{name}.json"
        command = build_promotion_pytest_command(
            module=nodeid,
            repo_root=repo_root,
            h5_path=h5_path,
            output_path=output_path,
            platform_label=platform_label,
            cache_policy=cache_policy,
            linarg_benchmark_k=linarg_benchmark_k,
            rhe_benchmark_num_matvecs=rhe_benchmark_num_matvecs,
            linarg_parallel_processes=parallel_processes,
        )
        runs.append(ChildRun(name=name, command=tuple(command), output_path=output_path))
    return tuple(runs)


def run_child(run: ChildRun, *, cwd: Path) -> None:
    """Run one isolated child and require its machine-readable fragment."""
    completed = subprocess.run(
        run.command,
        cwd=cwd,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = " ".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
        raise RuntimeError(f"promotion child {run.name!r} failed with exit code {completed.returncode}: {diagnostic}")
    if not run.output_path.is_file():
        raise RuntimeError(f"promotion child {run.name!r} did not write {run.output_path}")


def aggregate_child_fragments(
    paths: tuple[Path, ...],
    *,
    platform_label: str,
) -> PromotionEvidence:
    """Validate child schemas and combine records only when contexts match exactly."""
    if not paths:
        raise ValueError("at least one child fragment is required")
    evidences = []
    for path in paths:
        try:
            evidences.append(load_evidence(path))
        except (OSError, ValueError) as error:
            raise ValueError(f"child fragment {path.name} is invalid: {error}") from error

    first = evidences[0]
    for evidence in evidences:
        if evidence.candidate_commit != first.candidate_commit:
            raise ValueError("child candidate commit mismatch")
        if evidence.dataset != first.dataset:
            raise ValueError("child dataset fingerprint mismatch")
        if evidence.cache_label != first.cache_label:
            raise ValueError("child cache policy mismatch")
        if evidence.dirty_worktree != first.dirty_worktree:
            raise ValueError("child dirty-worktree status mismatch")
        if evidence.environment.platform_label != platform_label:
            raise ValueError("child platform label mismatch")
        if environment_comparison_key(evidence.environment) != environment_comparison_key(first.environment):
            fields = (
                "platform_label",
                "python_version",
                "jax_version",
                "jaxlib_version",
                "numpy_version",
                "os_name",
                "machine",
                "architecture",
                "xla_flags",
                "xla_cache_dir",
                "devices",
                "device_platforms",
                "cache_policy",
                "dirty_worktree",
                "build_configuration",
                "requested_device_count",
                "selected_devices",
                "selected_device_platforms",
            )
            mismatches = [
                field for field in fields if getattr(evidence.environment, field) != getattr(first.environment, field)
            ]
            raise ValueError(f"child environment mismatch: {', '.join(mismatches)}")

    records = tuple(record for evidence in evidences for record in evidence.records)
    _validate_product_input_parity(records)
    records = _apply_rhe_parity(records)
    return PromotionEvidence(
        schema_version=first.schema_version,
        candidate_commit=first.candidate_commit,
        dirty_worktree=first.dirty_worktree,
        behavioral_reference_commit=first.behavioral_reference_commit,
        dataset=first.dataset,
        produced_at_utc=datetime.now(timezone.utc).isoformat(),
        cache_label=first.cache_label,
        environment=first.environment,
        records=records,
    )


def _validate_product_input_parity(records: tuple[Any, ...]) -> None:
    hashes: dict[tuple[str, int | None, str], set[str]] = {}
    for record in records:
        if record.operation not in {"matmat", "rmatmat"}:
            continue
        try:
            input_sha256 = str(json.loads(record.notes)["input_sha256"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"product record {record.record_id!r} lacks input_sha256") from error
        key = (record.operation, record.workload_size, record.dtype)
        hashes.setdefault(key, set()).add(input_sha256)
    for key, observed_hashes in hashes.items():
        if len(observed_hashes) != 1:
            raise ValueError(f"product input parity mismatch for {key}: {sorted(observed_hashes)}")


def _apply_rhe_parity(records: tuple[Any, ...]) -> tuple[Any, ...]:
    baselines = {}
    for record in records:
        if record.operation != "rhe" or record.representation != Representation.NUMPY_CYTHON.value:
            continue
        try:
            baselines[record.workload_size] = np.asarray(json.loads(record.notes)["estimate"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue

    normalized = []
    for record in records:
        if record.operation != "rhe" or record.representation == Representation.NUMPY_CYTHON.value:
            normalized.append(record)
            continue
        try:
            observed = np.asarray(json.loads(record.notes)["estimate"])
            expected = baselines[record.workload_size]
            parity = bool(np.allclose(observed, expected, rtol=2e-5, atol=2e-5))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            parity = False
        if parity:
            normalized.append(record)
        else:
            try:
                notes = json.loads(record.notes)
            except (json.JSONDecodeError, TypeError):
                notes = {"original_notes": record.notes}
            notes["parity_error"] = "RHE estimate parity failed against fixed-probe NumPy/Cython baseline"
            normalized.append(
                replace(
                    record,
                    numeric_passed=False,
                    notes=json.dumps(notes, sort_keys=True),
                )
            )
    return tuple(normalized)


def synchronize_tree(value: Any) -> Any:
    """Block every asynchronous JAX-like result leaf before stopping a timer."""

    def synchronize_leaf(leaf: Any) -> Any:
        block = getattr(leaf, "block_until_ready", None)
        return block() if block is not None else leaf

    return jax.tree.map(synchronize_leaf, value)


def render_promotion_markdown(evidence: PromotionEvidence) -> str:
    """Render phase-level evidence without combining cold and warm timings."""
    numpy_baselines = {
        (record.operation, record.phase, record.workload_size): record.timed.seconds
        for record in evidence.records
        if record.representation == Representation.NUMPY_CYTHON.value and record.timed.seconds is not None
    }
    lines = [
        "| representation | operation | phase | workload | backend | devices | seconds | ratio to NumPy/Cython |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for record in sorted(
        evidence.records,
        key=lambda row: (row.operation, row.workload_size or 0, row.representation, row.phase),
    ):
        seconds = "" if record.timed.seconds is None else f"{record.timed.seconds:.6f}"
        backend = record.resolved_backend or "n/a"
        workload = "" if record.workload_size is None else str(record.workload_size)
        baseline = numpy_baselines.get((record.operation, record.phase, record.workload_size))
        ratio = "" if record.timed.seconds is None or baseline is None else f"{record.timed.seconds / baseline:.3f}"
        lines.append(
            f"| {record.representation} | {record.operation} | {record.phase} | "
            f"{workload} | {backend} | {record.device_count} | {seconds} | {ratio} |"
        )
    return "\n".join(lines)


def local_benchmark_gates(
    evidence: PromotionEvidence,
    *,
    production: bool,
    configured_rhe_ks: tuple[int, ...] | None = None,
) -> tuple[GateResult, ...]:
    """Evaluate named local metric gates without weakening missing production data."""
    packed_warm = tuple(
        record
        for record in evidence.records
        if record.representation == Representation.PACKED_CANDIDATE.value
        and record.phase == TimingPhase.WARM_EXECUTION.value
    )
    gates = []

    rhe_warm = tuple(
        record
        for record in evidence.records
        if record.operation == "rhe" and record.phase == TimingPhase.WARM_EXECUTION.value
    )
    if configured_rhe_ks is None:
        configured_rhe_ks = (
            REQUIRED_PRODUCT_KS
            if production
            else tuple(sorted({record.workload_size for record in rhe_warm if record.workload_size is not None}))
        )
    required_rhe_ks = (
        tuple(dict.fromkeys((*REQUIRED_PRODUCT_KS, *configured_rhe_ks))) if production else configured_rhe_ks
    )
    rhe_rows_by_key = {
        (k, representation.value): tuple(
            record for record in rhe_warm if record.workload_size == k and record.representation == representation.value
        )
        for k in required_rhe_ks
        for representation in Representation
    }
    rhe_complete = bool(required_rhe_ks) and all(
        len(rows) == 1
        and rows[0].status == "pass"
        and rows[0].numeric_passed
        and rows[0].timed.seconds is not None
        and rows[0].timed.seconds > 0
        for rows in rhe_rows_by_key.values()
    )
    product_numerical_rows = tuple(record for record in packed_warm if record.operation in {"matmat", "rmatmat"})
    numerical_passed = (
        bool(product_numerical_rows)
        and all(record.numeric_passed for record in product_numerical_rows)
        and rhe_complete
    )
    gates.append(
        GateResult(
            gate="numerical",
            status=GateStatus.PASS if numerical_passed else GateStatus.FAIL,
            reason=(
                "packed products and exact, packed, and NumPy/Cython RHE rows passed for every required K"
                if numerical_passed
                else (
                    "packed product plus exactly one valid exact, packed, and NumPy/Cython "
                    f"RHE warm row at K={','.join(str(k) for k in required_rhe_ks)} are required"
                )
            ),
        )
    )

    product_rows = tuple(record for record in packed_warm if record.operation in {"matmat", "rmatmat"})
    constants_passed = bool(product_rows) and all(record.metric.graph_constant_bytes == 0 for record in product_rows)
    gates.append(
        GateResult(
            gate="graph_constants",
            status=GateStatus.PASS if constants_passed else GateStatus.FAIL,
            reason=(
                "packed graphs remain explicit operands"
                if constants_passed
                else "packed StableHLO captured graph constants"
            ),
        )
    )
    expected_operands = len(PACKED_COMPONENT_NAMES) - 1
    operands_passed = bool(product_rows) and all(
        record.metric.graph_operand_count == expected_operands for record in product_rows
    )
    gates.append(
        GateResult(
            gate="graph_operands",
            status=GateStatus.PASS if operands_passed else GateStatus.FAIL,
            reason=(
                f"packed graph operand count is {expected_operands}"
                if operands_passed
                else f"packed graph operand count differs from {expected_operands}"
            ),
        )
    )
    ir_passed = bool(product_rows) and all(
        record.metric.stablehlo_operation_count is not None and record.metric.stablehlo_operation_count > 0
        for record in product_rows
    )
    gates.append(
        GateResult(
            gate="stablehlo",
            status=GateStatus.PASS if ir_passed else GateStatus.FAIL,
            reason="StableHLO operation counts recorded" if ir_passed else "missing StableHLO operation count",
        )
    )

    if not production:
        return tuple(gates)

    if "forced-two-device-cpu" not in attested_platforms(evidence.environment):
        return tuple(gates)

    two_device = tuple(record for record in product_rows if record.device_count == 2)
    if not two_device:
        for name in ("padding", "residency", "dense_communication"):
            gates.append(
                GateResult(
                    gate=name,
                    status=GateStatus.MISSING,
                    reason="forced two-device run is missing two-device packed evidence",
                )
            )
        return tuple(gates)

    padding_passed = all(
        record.metric.canonical_graph_bytes is not None
        and record.metric.padded_graph_bytes is not None
        and record.metric.padded_graph_bytes <= 1.25 * record.metric.canonical_graph_bytes
        for record in two_device
    )
    gates.append(
        GateResult(
            gate="padding",
            status=GateStatus.PASS if padding_passed else GateStatus.FAIL,
            reason=(
                "padded/canonical graph bytes <= 1.25" if padding_passed else "padded/canonical graph bytes exceed 1.25"
            ),
        )
    )
    residency_passed = all(
        record.metric.canonical_graph_bytes is not None
        and record.metric.max_device_graph_bytes is not None
        and record.metric.max_device_graph_bytes <= 0.65 * record.metric.canonical_graph_bytes
        for record in two_device
    )
    gates.append(
        GateResult(
            gate="residency",
            status=GateStatus.PASS if residency_passed else GateStatus.FAIL,
            reason=(
                "maximum device graph bytes <= 0.65 * canonical bytes"
                if residency_passed
                else "maximum device graph bytes exceed 0.65 * canonical bytes"
            ),
        )
    )
    communication_passed = all(
        record.metric.logical_collective_bytes == expected_logical_collective_bytes(record, evidence.dataset)
        for record in two_device
    )
    gates.append(
        GateResult(
            gate="dense_communication",
            status=GateStatus.PASS if communication_passed else GateStatus.FAIL,
            reason=(
                "logical dense collective bytes recorded"
                if communication_passed
                else "missing logical dense collective bytes"
            ),
        )
    )
    return tuple(gates)


def persisted_gate_outcomes(
    evidence: PromotionEvidence,
    local_gates: tuple[GateResult, ...],
    *,
    validation_evidence_id: str | None,
) -> tuple[EvidenceGateOutcome, ...]:
    """Convert local checks and runner validation provenance to schema outcomes."""
    record_digest = hashlib.sha256(
        "\n".join(sorted(record.record_id for record in evidence.records)).encode()
    ).hexdigest()
    outcomes: list[EvidenceGateOutcome] = []
    if validation_evidence_id:
        for gate in ("correctness_float32", "correctness_float64", "transform"):
            outcomes.append(
                EvidenceGateOutcome(
                    evidence_id=f"{validation_evidence_id}:{gate}",
                    gate=gate,
                    status=GateStatus.PASS,
                    reason="portable runner validation suite completed successfully",
                )
            )

    by_name = {gate.gate: gate for gate in local_gates}
    direct_names = {
        "numerical": "numerical",
        "padding": "padding",
        "residency": "residency",
        "dense_communication": "communication",
    }
    for local_name, persisted_name in direct_names.items():
        if local_name not in by_name:
            continue
        local = by_name[local_name]
        outcomes.append(
            EvidenceGateOutcome(
                evidence_id=f"benchmark-records-sha256:{record_digest}:{persisted_name}",
                gate=persisted_name,
                status=local.status,
                reason=local.reason,
            )
        )

    ir_parts = tuple(by_name[name] for name in ("graph_constants", "graph_operands", "stablehlo") if name in by_name)
    if ir_parts:
        failed = tuple(gate for gate in ir_parts if gate.status is not GateStatus.PASS)
        status = GateStatus.PASS
        if any(gate.status is GateStatus.FAIL for gate in failed):
            status = GateStatus.FAIL
        elif failed:
            status = GateStatus.MISSING
        outcomes.append(
            EvidenceGateOutcome(
                evidence_id=f"benchmark-records-sha256:{record_digest}:ir",
                gate="ir",
                status=status,
                reason="; ".join(f"{gate.gate}: {gate.reason}" for gate in ir_parts),
            )
        )
    return tuple(outcomes)


def test_promotion_benchmark_matrix(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    linarg_h5_path: Path,
    linarg_benchmark_k_values: tuple[int, ...],
    rhe_benchmark_num_matvecs: tuple[int, ...],
    linarg_parallel_processes: int,
) -> None:
    """Run isolated children, aggregate their fragments, and evaluate local gates."""
    if not request.config.getoption("--runbench"):
        pytest.skip("benchmarks require --runbench")
    output_path = request.config.getoption("--jax-promotion-output")
    if output_path is None:
        pytest.fail("promotion benchmark requires --jax-promotion-output PATH")

    repo_root = Path(__file__).resolve().parents[3]
    platform_label = request.config.getoption("--platform-label")
    cache_policy = request.config.getoption("--cache-policy")
    production = request.config.getoption("--linarg-h5-path") is not None
    runs = build_child_runs(
        repo_root=repo_root,
        h5_path=linarg_h5_path.resolve(),
        fragment_dir=tmp_path / "fragments",
        platform_label=platform_label,
        cache_policy=cache_policy,
        linarg_benchmark_k=linarg_benchmark_k_values,
        rhe_benchmark_num_matvecs=rhe_benchmark_num_matvecs,
        parallel_processes=linarg_parallel_processes,
    )
    for run in runs:
        run_child(run, cwd=repo_root)

    evidence = aggregate_child_fragments(
        tuple(run.output_path for run in runs),
        platform_label=platform_label,
    )
    local_gates = local_benchmark_gates(
        evidence,
        production=production,
        configured_rhe_ks=rhe_benchmark_num_matvecs,
    )
    evidence = replace(
        evidence,
        gate_outcomes=persisted_gate_outcomes(
            evidence,
            local_gates,
            validation_evidence_id=request.config.getoption("--jax-validation-evidence-id"),
        ),
    )
    write_evidence_fragment(output_path, evidence)
    print("\n" + render_promotion_markdown(evidence))

    decision, ratio_gates, _ = evaluate_ratio_gates(evidence)
    failed_local = tuple(gate for gate in local_gates if gate.status is not GateStatus.PASS)
    if request.config.getoption("--jax-enforce-promotion-gates") and (decision is not Decision.PROMOTE or failed_local):
        gates = (*ratio_gates, *failed_local)
        reasons = "\n".join(f"- {gate.gate}: {gate.status.value}: {gate.reason}" for gate in gates)
        pytest.fail(f"local promotion gates returned {decision.value}:\n{reasons}")
