# pattern: Imperative Shell

from __future__ import annotations

import hashlib
import json
import os

from pathlib import Path

import polars as pl
import pytest

from tests.jax.bench._promotion import (
    BenchmarkRecord,
    build_promotion_pytest_command,
    CachePolicy,
    compute_dataset_fingerprint,
    DatasetFingerprint,
    Decision,
    EnvironmentState,
    evaluate_evidence_set,
    evaluate_ratio_gates,
    GateFailureReason,
    GateStatus,
    load_evidences,
    PerformanceMetrics,
    PromotionEvidence,
    Representation,
    TimedPhase,
    TimingPhase,
)


def _metadata_env() -> EnvironmentState:
    return EnvironmentState(
        platform_label="local-platform",
        python_version="3.12",
        numpy_version="2.1.0",
        jax_version="0.5.0",
        jaxlib_version="0.5.0",
        os_name=os.name,
        machine="machine",
        architecture="arch",
        xla_flags="",
        devices=("cpu",),
        xla_cache_dir=None,
        command="pytest",
        dirty_worktree=False,
    )


def _dataset() -> DatasetFingerprint:
    return DatasetFingerprint(
        sha256="a" * 64,
        size_bytes=12,
        block_count=3,
        n_samples=10,
        n_variants=20,
    )


def _row(
    *,
    representation: str = Representation.PACKED_CANDIDATE.value,
    operation: str = "matmat",
    phase: str = TimingPhase.WARM_EXECUTION.value,
    workload_size: int = 4,
    dtype: str = "float32",
    requested_backend: str | None = "pure_jax",
    resolved_backend: str | None = "pure_jax",
    device_count: int = 2,
    seconds: float = 1.0,
    status: str = "pass",
    candidate_commit: str = "test",
) -> BenchmarkRecord:
    return BenchmarkRecord(
        record_id=f"{representation}|{operation}|{phase}|{workload_size}|{device_count}|{dtype}|{requested_backend}|{resolved_backend}|{candidate_commit}",
        platform_label="local-platform",
        cache_policy=CachePolicy.FRESH.value,
        candidate_commit=candidate_commit,
        behavioral_reference_commit="b68e7da",
        dirty_worktree=False,
        dataset=_dataset(),
        representation=representation,
        operation=operation,
        phase=phase,
        workload_size=workload_size,
        dtype=dtype,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        device_count=device_count,
        timed=TimedPhase(phase=phase, seconds=seconds),
        metric=PerformanceMetrics(),
        numeric_passed=True,
        status=status,
    )


def test_dataset_fingerprint_does_not_load_full_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5_path = tmp_path / "fixture.h5"
    h5_path.write_bytes(b"binary fixture data")

    metadata = pl.DataFrame(
        {
            "block_name": ["blk0", "blk1", "blk2"],
            "n_samples": [12, 12, 12],
            "n_variants": [6, 7, 8],
        }
    )
    monkeypatch.setattr("tests.jax.bench._promotion.list_blocks", lambda *_: metadata)

    first = compute_dataset_fingerprint(h5_path)
    second = compute_dataset_fingerprint(h5_path)

    expected = hashlib.sha256(b"binary fixture data").hexdigest()
    assert first.sha256 == second.sha256 == expected
    assert first.block_count == 3
    assert first.n_samples == 12
    assert first.n_variants == 21


def test_dataset_fingerprint_rejects_invalid_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h5_path = tmp_path / "fixture.h5"
    h5_path.write_bytes(b"abc")
    monkeypatch.setattr("tests.jax.bench._promotion.list_blocks", lambda *_: pl.DataFrame({"block_name": ["blk0"]}))
    with pytest.raises(ValueError, match="missing sample-count"):
        compute_dataset_fingerprint(h5_path)

    monkeypatch.setattr(
        "tests.jax.bench._promotion.list_blocks", lambda *_: pl.DataFrame({"n_samples": [1], "block_name": ["blk0"]})
    )
    with pytest.raises(ValueError, match="missing variant-count"):
        compute_dataset_fingerprint(h5_path)


def test_schema_round_trip_and_unknown_schema_rejected() -> None:
    fp = _dataset()
    evidence = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="test",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=fp,
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=(
            _row(
                candidate_commit="test",
                representation=Representation.PACKED_CANDIDATE.value,
                phase=TimingPhase.WARM_EXECUTION.value,
            ),
            _row(
                candidate_commit="test",
                representation=Representation.RETAINED_EXACT_RAGGED.value,
                phase=TimingPhase.WARM_EXECUTION.value,
            ),
        ),
    )
    restored = PromotionEvidence.from_json(evidence.to_json(), allow_repo_mismatch=True)
    assert restored.schema_version == evidence.schema_version

    payload = evidence.to_dict()
    payload["schema_version"] = "2030-01-01"
    with pytest.raises(ValueError, match="unknown schema"):
        PromotionEvidence.from_dict(payload, allow_repo_mismatch=True)


def test_duplicate_record_id_is_rejected() -> None:
    duplicate = _row(candidate_commit="test")
    evidence_records = (
        duplicate,
        BenchmarkRecord(
            record_id=duplicate.record_id,
            platform_label=duplicate.platform_label,
            cache_policy=duplicate.cache_policy,
            candidate_commit=duplicate.candidate_commit,
            behavioral_reference_commit=duplicate.behavioral_reference_commit,
            dirty_worktree=duplicate.dirty_worktree,
            dataset=duplicate.dataset,
            representation=Representation.RETAINED_EXACT_RAGGED.value,
            operation="rmatmat",
            phase=TimingPhase.WARM_EXECUTION.value,
            workload_size=4,
            dtype="float32",
            requested_backend="pure_jax",
            resolved_backend="pure_jax",
            device_count=2,
            timed=TimedPhase(phase=TimingPhase.WARM_EXECUTION.value, seconds=0.9),
            metric=PerformanceMetrics(),
        ),
    )
    with pytest.raises(ValueError, match="duplicate record_id"):
        PromotionEvidence(
            schema_version="2026-08-13+2",
            candidate_commit="test",
            dirty_worktree=False,
            behavioral_reference_commit="b68e7da",
            dataset=_dataset(),
            produced_at_utc="2026-08-19T12:00:00+00:00",
            cache_label=CachePolicy.FRESH.value,
            environment=_metadata_env(),
            records=evidence_records,
        )


def test_ratio_gate_pass_boundaries_and_missing_baseline() -> None:
    exact = _row(
        representation=Representation.RETAINED_EXACT_RAGGED.value,
        operation="matmat",
        phase=TimingPhase.WARM_EXECUTION.value,
        workload_size=4,
        seconds=1.0,
        candidate_commit="test",
    )
    packed_ok = _row(
        representation=Representation.PACKED_CANDIDATE.value,
        operation="matmat",
        phase=TimingPhase.WARM_EXECUTION.value,
        workload_size=4,
        seconds=1.2,
        candidate_commit="test",
    )
    packed_bad = _row(
        representation=Representation.PACKED_CANDIDATE.value,
        operation="matmat",
        phase=TimingPhase.WARM_EXECUTION.value,
        workload_size=4,
        seconds=1.2000001,
        candidate_commit="test",
    )

    evidence = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="test",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=(exact, packed_ok),
    )
    decision, gates, blockers = evaluate_ratio_gates(evidence, required_ratio=1.20)
    assert decision == Decision.PROMOTE
    assert blockers == 0
    assert gates and all(gate.status == GateStatus.PASS for gate in gates)

    evidence_fail = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="test",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=(exact, packed_bad),
    )
    decision_bad, gates_bad, blockers_bad = evaluate_ratio_gates(evidence_fail, required_ratio=1.20)
    assert decision_bad == Decision.CONTINUE_COEXISTENCE
    assert blockers_bad == 1
    assert gates_bad[0].status == GateStatus.FAIL

    missing = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="test",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=(packed_ok,),
    )
    decision_missing, gates_missing, blockers_missing = evaluate_ratio_gates(missing)
    assert decision_missing == Decision.CONTINUE_COEXISTENCE
    assert blockers_missing == 1
    assert gates_missing[0].status == GateStatus.MISSING
    assert gates_missing[0].reason == f"{GateFailureReason.MISSING_EXACT.value} for packed warm measurement"


def test_evidence_set_require_single_commit_and_collects_gate_failures() -> None:
    common_env = _metadata_env()
    ev_keep = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="abc",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=common_env,
        records=(
            _row(
                candidate_commit="abc",
                representation=Representation.RETAINED_EXACT_RAGGED.value,
                status="pass",
                seconds=1.0,
            ),
            _row(
                candidate_commit="abc",
                representation=Representation.PACKED_CANDIDATE.value,
                status="pass",
                seconds=0.9,
                phase=TimingPhase.WARM_EXECUTION.value,
            ),
        ),
    )
    ev_reject = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="def",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=common_env,
        records=(
            _row(
                candidate_commit="def",
                representation=Representation.RETAINED_EXACT_RAGGED.value,
                status="pass",
                seconds=1.0,
            ),
            _row(
                candidate_commit="def",
                representation=Representation.PACKED_CANDIDATE.value,
                status="pass",
                seconds=0.9,
                phase=TimingPhase.WARM_EXECUTION.value,
            ),
        ),
    )

    decision_mismatch = evaluate_evidence_set((ev_keep, ev_reject))
    assert decision_mismatch.decision == Decision.REJECT

    decision_empty = evaluate_evidence_set(())
    assert decision_empty.decision == Decision.REJECT
    assert decision_empty.blocker_count == 1


def test_load_evidences_round_trip(tmp_path: Path) -> None:
    fp = _dataset()
    row = _row(candidate_commit="x")
    evidence = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="x",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=fp,
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=(row,),
    )
    file_a = tmp_path / "run-a.json"
    file_b = tmp_path / "run-b.json"
    file_a.write_text(evidence.to_json(), encoding="utf-8")
    file_b.write_text(json.dumps([evidence.to_dict()]), encoding="utf-8")
    loaded = load_evidences(tmp_path)
    assert len(loaded) == 2
    assert loaded[0].candidate_commit == "x"
    assert loaded[1].candidate_commit == "x"


def test_build_promotion_pytest_command_includes_required_flags(tmp_path: Path) -> None:
    (tmp_path / "data.h5").write_bytes(b"abc")
    command = build_promotion_pytest_command(
        module="tests/jax/bench/test_promotion_benchmarks.py",
        repo_root=tmp_path,
        h5_path=tmp_path / "data.h5",
        output_path=tmp_path / "out.json",
        platform_label="x86",
        cache_policy=CachePolicy.FRESH.value,
        linarg_benchmark_k=(4, 20),
        rhe_benchmark_num_matvecs=(4, 20),
        enforce_gates=True,
    )

    assert command[:2] == ["uv", "run"]
    assert "-p" in command and "no:capture" in command
    assert "--jax-promotion-output" in command
    assert "--jax-enforce-promotion-gates" in command
    assert "--cache-policy" in command
    assert "--platform-label" in command
    assert "--linarg-benchmark-k" in command
    assert "--rhe-benchmark-num-matvecs" in command
