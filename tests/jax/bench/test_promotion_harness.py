# pattern: Imperative Shell

from __future__ import annotations

import hashlib
import json
import os
import subprocess

from dataclasses import replace
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
    is_git_dirty,
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


def test_git_dirty_ignores_untracked_benchmark_data_but_detects_tracked_edits(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "tests@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Tests"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "commit.gpgsign", "false"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"], check=True)

    (tmp_path / "benchmark-data.h5").write_bytes(b"local data")
    assert is_git_dirty(tmp_path) is False

    (tmp_path / "untracked.py").write_text("pass\n", encoding="utf-8")
    assert is_git_dirty(tmp_path) is True

    tracked.write_text("modified\n", encoding="utf-8")
    assert is_git_dirty(tmp_path) is True


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


def test_performance_metrics_round_trip_stablehlo_bytes() -> None:
    metrics = PerformanceMetrics(stablehlo_bytes=1234, stablehlo_operation_count=8)
    restored = PerformanceMetrics.from_dict(metrics.to_dict())
    assert restored.stablehlo_bytes == 1234


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


def test_child_matrix_is_isolated_and_preserves_benchmark_context(tmp_path: Path) -> None:
    from tests.jax.bench.test_promotion_benchmarks import build_child_runs

    h5_path = tmp_path / "data.h5"
    h5_path.write_bytes(b"abc")
    runs = build_child_runs(
        repo_root=tmp_path,
        h5_path=h5_path,
        fragment_dir=tmp_path / "fragments",
        platform_label="forced-two-device-cpu",
        cache_policy=CachePolicy.REUSED.value,
        linarg_benchmark_k=(4, 20),
        rhe_benchmark_num_matvecs=(4, 20),
        parallel_processes=2,
    )

    assert [run.name for run in runs] == [
        "product-packed",
        "product-exact",
        "product-numpy-cython",
        "rhe-packed",
        "rhe-exact",
        "rhe-numpy-cython",
    ]
    assert len({run.output_path for run in runs}) == len(runs)
    for run in runs:
        command = run.command
        assert command[:3] == ("uv", "run", "pytest")
        assert command[3:5] == ("-p", "no:capture")
        assert command.count("--jax-promotion-output") == 1
        assert command[command.index("--platform-label") + 1] == "forced-two-device-cpu"
        assert command[command.index("--cache-policy") + 1] == CachePolicy.REUSED.value
        assert command[command.index("--linarg-parallel-processes") + 1] == "2"
        assert "4" in command and "20" in command


def test_aggregate_child_fragments_validates_schema_and_context(tmp_path: Path) -> None:
    from tests.jax.bench.test_promotion_benchmarks import aggregate_child_fragments

    packed = _row(candidate_commit="candidate", representation=Representation.PACKED_CANDIDATE.value)
    exact = _row(candidate_commit="candidate", representation=Representation.RETAINED_EXACT_RAGGED.value)
    paths = []
    for name, record in (("packed", packed), ("exact", exact)):
        path = tmp_path / f"{name}.json"
        evidence = PromotionEvidence(
            schema_version="2026-08-13+2",
            candidate_commit="candidate",
            dirty_worktree=False,
            behavioral_reference_commit="b68e7da",
            dataset=_dataset(),
            produced_at_utc="2026-08-19T12:00:00+00:00",
            cache_label=CachePolicy.FRESH.value,
            environment=_metadata_env(),
            records=(record,),
        )
        path.write_text(evidence.to_json(), encoding="utf-8")
        paths.append(path)

    combined = aggregate_child_fragments(tuple(paths), platform_label="local-platform")
    assert {record.representation for record in combined.records} == {
        Representation.PACKED_CANDIDATE.value,
        Representation.RETAINED_EXACT_RAGGED.value,
    }

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema_version":"invalid"}', encoding="utf-8")
    with pytest.raises(ValueError, match="child fragment malformed.json"):
        aggregate_child_fragments((malformed,), platform_label="local-platform")


def test_aggregate_child_fragments_rejects_mismatched_dataset(tmp_path: Path) -> None:
    from tests.jax.bench.test_promotion_benchmarks import aggregate_child_fragments

    first = _row(candidate_commit="candidate")
    other_dataset = DatasetFingerprint(
        sha256="b" * 64,
        size_bytes=12,
        block_count=3,
        n_samples=10,
        n_variants=20,
    )
    second = replace(
        first,
        record_id="mismatched-dataset",
        dataset=other_dataset,
        representation=Representation.RETAINED_EXACT_RAGGED.value,
    )
    paths = []
    for index, record in enumerate((first, second)):
        evidence = PromotionEvidence(
            schema_version="2026-08-13+2",
            candidate_commit="candidate",
            dirty_worktree=False,
            behavioral_reference_commit="b68e7da",
            dataset=record.dataset,
            produced_at_utc="2026-08-19T12:00:00+00:00",
            cache_label=CachePolicy.FRESH.value,
            environment=_metadata_env(),
            records=(record,),
        )
        path = tmp_path / f"fragment-{index}.json"
        path.write_text(evidence.to_json(), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
        aggregate_child_fragments(tuple(paths), platform_label="local-platform")


def test_aggregate_child_fragments_marks_rhe_parity_failure(tmp_path: Path) -> None:
    from tests.jax.bench.test_promotion_benchmarks import aggregate_child_fragments

    numpy_row = replace(
        _row(candidate_commit="candidate", representation=Representation.NUMPY_CYTHON.value),
        operation="rhe",
        notes=json.dumps({"estimate": [[1.0, 2.0, 0.5]]}),
    )
    packed_row = replace(
        _row(candidate_commit="candidate", representation=Representation.PACKED_CANDIDATE.value),
        operation="rhe",
        notes=json.dumps({"estimate": [[1.0, 2.0, 0.8]]}),
    )
    paths = []
    for name, record in (("numpy", numpy_row), ("packed", packed_row)):
        evidence = PromotionEvidence(
            schema_version="2026-08-13+2",
            candidate_commit="candidate",
            dirty_worktree=False,
            behavioral_reference_commit="b68e7da",
            dataset=_dataset(),
            produced_at_utc="2026-08-19T12:00:00+00:00",
            cache_label=CachePolicy.FRESH.value,
            environment=_metadata_env(),
            records=(record,),
        )
        path = tmp_path / f"{name}.json"
        path.write_text(evidence.to_json(), encoding="utf-8")
        paths.append(path)

    combined = aggregate_child_fragments(tuple(paths), platform_label="local-platform")
    packed = next(record for record in combined.records if record.representation == "packed_candidate")
    assert packed.numeric_passed is False
    assert "RHE estimate parity failed" in packed.notes


def test_run_child_propagates_failure_and_missing_fragment(tmp_path: Path) -> None:
    from tests.jax.bench.test_promotion_benchmarks import ChildRun, run_child

    failed = ChildRun(
        name="failed",
        command=("sh", "-c", "printf 'child diagnostic' >&2; exit 7"),
        output_path=tmp_path / "failed.json",
    )
    with pytest.raises(RuntimeError, match="failed.*exit code 7.*child diagnostic"):
        run_child(failed, cwd=tmp_path)

    missing = ChildRun(
        name="missing",
        command=("sh", "-c", "exit 0"),
        output_path=tmp_path / "missing.json",
    )
    with pytest.raises(RuntimeError, match="did not write"):
        run_child(missing, cwd=tmp_path)


def test_synchronize_tree_blocks_all_jax_like_leaves() -> None:
    from tests.jax.bench.test_promotion_benchmarks import synchronize_tree

    class Pending:
        def __init__(self) -> None:
            self.ready = False

        def block_until_ready(self):
            self.ready = True
            return self

    first = Pending()
    second = Pending()
    result = synchronize_tree({"a": first, "b": (second,)})
    assert result["a"] is first
    assert first.ready and second.ready


def test_render_promotion_markdown_keeps_phases_separate() -> None:
    from tests.jax.bench.test_promotion_benchmarks import render_promotion_markdown

    records = (
        _row(candidate_commit="x", phase=TimingPhase.CONSTRUCTION.value),
        _row(candidate_commit="x", phase=TimingPhase.FIRST_EXECUTION.value),
        _row(candidate_commit="x", phase=TimingPhase.WARM_EXECUTION.value),
    )
    evidence = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="x",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=records,
    )
    markdown = render_promotion_markdown(evidence)
    assert "| construction |" in markdown
    assert "| first_execution |" in markdown
    assert "| warm_execution |" in markdown
    assert "cold_total" not in markdown


def test_local_benchmark_gates_name_memory_ir_and_numerical_failures() -> None:
    from tests.jax.bench.test_promotion_benchmarks import local_benchmark_gates

    healthy_metric = PerformanceMetrics(
        canonical_graph_bytes=100,
        padded_graph_bytes=120,
        max_device_graph_bytes=60,
        graph_constant_bytes=0,
        graph_operand_count=10,
        stablehlo_operation_count=8,
        logical_collective_bytes=32,
    )
    packed = replace(_row(candidate_commit="x", device_count=2), metric=healthy_metric)
    exact = _row(
        candidate_commit="x",
        device_count=2,
        representation=Representation.RETAINED_EXACT_RAGGED.value,
    )
    evidence = PromotionEvidence(
        schema_version="2026-08-13+2",
        candidate_commit="x",
        dirty_worktree=False,
        behavioral_reference_commit="b68e7da",
        dataset=_dataset(),
        produced_at_utc="2026-08-19T12:00:00+00:00",
        cache_label=CachePolicy.FRESH.value,
        environment=_metadata_env(),
        records=(packed, exact),
    )
    assert all(gate.status is GateStatus.PASS for gate in local_benchmark_gates(evidence, production=True))

    failed_metric = replace(
        healthy_metric,
        padded_graph_bytes=126,
        max_device_graph_bytes=66,
        graph_constant_bytes=1,
        logical_collective_bytes=0,
    )
    failed = replace(packed, metric=failed_metric, numeric_passed=False)
    failed_evidence = replace(evidence, records=(failed, exact))
    gates = local_benchmark_gates(failed_evidence, production=True)
    failed_names = {gate.gate for gate in gates if gate.status is GateStatus.FAIL}
    assert {"padding", "residency", "graph_constants", "dense_communication", "numerical"} <= failed_names


def test_xla_summary_json_mode_parses_small_fixture(tmp_path: Path) -> None:
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    hlo = dump_dir / "module.cpu_after_optimizations.txt"
    hlo.write_text(
        'ENTRY main {\n  ROOT out = f32[2] custom-call(p0), custom_call_target="linear_dag"\n}\n',
        encoding="utf-8",
    )
    assignment = dump_dir / "module.cpu_after_optimizations-buffer-assignment.txt"
    assignment.write_text(
        "allocation 0: size 4096, maybe-live-out:\n  value: <0 main @0> (alias)\nTotal bytes used: 4096 (4.0KiB)\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    script = Path(__file__).resolve().parents[3] / "scripts" / "summarize_xla_memory_dump.sh"

    completed = subprocess.run(
        ["bash", str(script), "--json-output", str(output), str(dump_dir)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["total_buffer_assignment_bytes"] == 4096
    assert payload["modules"][0]["large_allocations"] == [4096]
    assert payload["modules"][0]["alias_count"] == 2
    assert payload["modules"][0]["custom_call_count"] == 1


def _init_runner_repo(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "tests@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Tests"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "commit.gpgsign", "false"], check=True)
    tracked = path / "tracked.txt"
    tracked.write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
    return path


def _run_promotion_runner(
    *,
    repo_root: Path,
    h5_path: Path,
    output_dir: Path,
    platform_label: str,
    device_count: int,
    extra_args: tuple[str, ...] = (),
    process_env: dict[str, str] | None = None,
    dry_run: bool = True,
) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_jax_promotion.sh"
    command = [
        "bash",
        str(script),
        "--repo-root",
        str(repo_root),
        "--hdf5-path",
        str(h5_path),
        "--output-dir",
        str(output_dir),
        "--platform-label",
        platform_label,
        "--device-count",
        str(device_count),
    ]
    if dry_run:
        command.append("--dry-run")
    command.extend(extra_args)
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=process_env,
    )


@pytest.mark.parametrize(
    ("platform_label", "device_count", "backend_marker", "excluded_marker"),
    [
        ("forced-two-device-cpu", 2, "test_operator_ffi_cpu.py", "JAX_PLATFORMS=gpu"),
        ("gpu", 1, "backend_mode=pure_jax", "test_operator_ffi_cpu.py"),
    ],
)
def test_runner_builds_cpu_ffi_and_gpu_pure_jax_commands(
    tmp_path: Path,
    platform_label: str,
    device_count: int,
    backend_marker: str,
    excluded_marker: str,
) -> None:
    repo_root = _init_runner_repo(tmp_path / "repo")
    h5_path = repo_root / "representative.h5"
    h5_path.write_bytes(b"fixture")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    completed = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label=platform_label,
        device_count=device_count,
    )

    assert completed.returncode == 0, completed.stderr
    logs = "\n".join(path.read_text(encoding="utf-8") for path in sorted(output_dir.glob("*.log")))
    assert backend_marker in logs
    assert excluded_marker not in logs
    pytest_commands = [line for line in logs.splitlines() if "pytest" in line and line.startswith("command=")]
    assert pytest_commands
    assert all("pytest -p no:capture" in command for command in pytest_commands)


def test_product_promotion_gpu_selection_uses_gpu_devices_and_pure_jax_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.jax.bench import test_parallel_benchmarks as product_benchmarks

    requested_platforms = []

    def devices_for_backend(platform: str) -> list[object]:
        requested_platforms.append(platform)
        return [object(), object()]

    monkeypatch.setattr(product_benchmarks, "_devices_for_backend", devices_for_backend)
    metadata = pl.DataFrame({"block_name": ["one", "two"]})

    backends = product_benchmarks._promotion_backends(Representation.PACKED_CANDIDATE, "gpu")
    counts = product_benchmarks._promotion_device_counts(metadata, "gpu")

    assert backends == (product_benchmarks.Backend.PURE_JAX,)
    assert counts == (1, 2)
    assert requested_platforms == ["gpu"]


def test_rhe_promotion_gpu_selection_requests_gpu_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.jax.bench import test_rhe_benchmarks as rhe_benchmarks

    requested_platforms = []
    gpu_devices = (object(), object())

    def devices(platform: str) -> tuple[object, ...]:
        requested_platforms.append(platform)
        return gpu_devices

    monkeypatch.setattr(rhe_benchmarks.jax, "devices", devices)

    assert rhe_benchmarks._promotion_devices("gpu") == gpu_devices
    assert requested_platforms == ["gpu"]


def test_runner_uses_one_cache_for_distinct_fresh_and_reused_processes(tmp_path: Path) -> None:
    repo_root = _init_runner_repo(tmp_path / "repo")
    h5_path = repo_root / "representative.h5"
    h5_path.write_bytes(b"fixture")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    completed = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label="arm64-cpu",
        device_count=1,
    )

    assert completed.returncode == 0, completed.stderr
    fresh = (output_dir / "arm64-cpu.fresh.environment.log").read_text(encoding="utf-8")
    reused = (output_dir / "arm64-cpu.reused.environment.log").read_text(encoding="utf-8")
    assert "cache_policy=fresh" in fresh
    assert "cache_policy=reused" in reused
    assert "cache_directory_id=shared-persistent-cache" in fresh
    assert "cache_directory_id=shared-persistent-cache" in reused
    assert "process_id=benchmark-fresh" in fresh
    assert "process_id=benchmark-reused" in reused
    assert (output_dir / "checksums.sha256").is_file()


@pytest.mark.parametrize("unsafe_output", ["repo", "ancestor", "root"])
def test_runner_rejects_unsafe_broad_output_paths(tmp_path: Path, unsafe_output: str) -> None:
    repo_root = _init_runner_repo(tmp_path / "repo")
    h5_path = repo_root / "representative.h5"
    h5_path.write_bytes(b"fixture")
    output_dir = {
        "repo": repo_root,
        "ancestor": tmp_path,
        "root": Path("/"),
    }[unsafe_output]

    completed = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label="arm64-cpu",
        device_count=1,
    )

    assert completed.returncode != 0
    assert "unsafe output directory" in completed.stderr


def test_runner_rejects_dirty_commit_unless_explicitly_nonpromotable(tmp_path: Path) -> None:
    repo_root = _init_runner_repo(tmp_path / "repo")
    h5_path = repo_root / "representative.h5"
    h5_path.write_bytes(b"fixture")
    (repo_root / "dirty.py").write_text("pass\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    rejected = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label="arm64-cpu",
        device_count=1,
    )
    assert rejected.returncode != 0
    assert "clean candidate commit" in rejected.stderr

    allowed = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label="arm64-cpu",
        device_count=1,
        extra_args=("--allow-dirty", "--no-enforce-gates"),
    )
    assert allowed.returncode == 0, allowed.stderr
    environment = (output_dir / "arm64-cpu.fresh.environment.log").read_text(encoding="utf-8")
    assert "candidate_clean=false" in environment
    assert "promotable=false" in environment


def test_runner_redacts_absolute_paths_from_logs(tmp_path: Path) -> None:
    repo_root = _init_runner_repo(tmp_path / "repo")
    h5_path = repo_root / "representative.h5"
    h5_path.write_bytes(b"fixture")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    completed = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label="x86_64-cpu",
        device_count=1,
    )

    assert completed.returncode == 0, completed.stderr
    logs = "\n".join(path.read_text(encoding="utf-8") for path in sorted(output_dir.glob("*.log")))
    assert str(repo_root) not in logs
    assert str(h5_path) not in logs
    assert str(output_dir) not in logs
    assert "<repo>" in logs
    assert "<dataset>" in logs
    assert "<output>" in logs


def test_runner_execution_log_retains_every_setup_command_result(tmp_path: Path) -> None:
    repo_root = _init_runner_repo(tmp_path / "repo")
    h5_path = repo_root / "representative.h5"
    h5_path.write_bytes(b"fixture")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/bin/sh
if [ "$1" = "build" ]; then
    printf 'fake-build\\n'
    exit 0
fi
printf 'fake-pytest-x64=%s\\n' "${JAX_ENABLE_X64:-unset}"
printf 'fake-host=%s\\n' "${HOSTNAME:-unset}"
printf 'fake-token=%s\\n' "${PROMOTION_TEST_TOKEN:-unset}"
previous=''
for argument in "$@"; do
    if [ "$previous" = "--jax-promotion-output" ]; then
        printf '{}\\n' > "$argument"
    fi
    previous=$argument
done
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    process_env = os.environ.copy()
    process_env["PATH"] = f"{fake_bin}{os.pathsep}{process_env['PATH']}"
    process_env["HOSTNAME"] = "private-runner-host.example.invalid"
    process_env["PROMOTION_TEST_TOKEN"] = "private-promotion-token"

    completed = _run_promotion_runner(
        repo_root=repo_root,
        h5_path=h5_path,
        output_dir=output_dir,
        platform_label="arm64-cpu",
        device_count=1,
        extra_args=("--no-enforce-gates",),
        process_env=process_env,
        dry_run=False,
    )

    assert completed.returncode == 0, completed.stderr
    execution = (output_dir / "arm64-cpu.setup.execution.log").read_text(encoding="utf-8")
    assert "fake-build" in execution
    assert "fake-pytest-x64=0" in execution
    assert "fake-pytest-x64=1" in execution
    assert "private-runner-host.example.invalid" not in execution
    assert "fake-host=<host>" in execution
    assert "private-promotion-token" not in execution
    assert "fake-token=<redacted-secret>" in execution


def test_committed_promotion_decision_matches_normalized_evidence() -> None:
    plan_root = (
        Path(__file__).resolve().parents[3]
        / ".plans"
        / "implementation-plans"
        / "2026-08-13-jax-packed-sharded-lineararg"
    )
    evidence_dir = plan_root / "evidence"
    decision_path = plan_root / "promotion-decision.md"

    evidence_paths = sorted(evidence_dir.glob("*.json"))
    assert evidence_paths, "Task 4 must commit normalized evidence JSON"
    evidences = tuple(load_evidences(evidence_dir))
    markdown = decision_path.read_text(encoding="utf-8")

    commits = {evidence.candidate_commit for evidence in evidences}
    fingerprints = {evidence.dataset.sha256 for evidence in evidences}
    assert commits == {"764c3f8c29f63d75ba4e47d88e61d2019f129d01"}
    assert len(fingerprints) == 1
    assert all(not evidence.dirty_worktree for evidence in evidences)
    assert "2b9165403c912a9d0a11502bebee0fad14b45e6e" in markdown
    assert "historical" in markdown.lower()

    required_platforms = {
        "arm64-cpu",
        "x86_64-cpu",
        "forced-two-device-cpu",
        "gpu",
    }
    available_platforms = {evidence.environment.platform_label for evidence in evidences}
    if any(
        evidence.environment.machine == "arm64"
        and evidence.environment.devices
        and set(evidence.environment.devices) == {"cpu"}
        for evidence in evidences
    ):
        available_platforms.add("arm64-cpu")
    missing_platforms = required_platforms - available_platforms
    local_decision = evaluate_evidence_set(evidences)
    if local_decision.decision is Decision.REJECT:
        expected_decision = Decision.REJECT
    elif missing_platforms or local_decision.decision is Decision.CONTINUE_COEXISTENCE:
        expected_decision = Decision.CONTINUE_COEXISTENCE
    else:
        expected_decision = Decision.PROMOTE

    assert expected_decision is Decision.CONTINUE_COEXISTENCE
    assert local_decision.blocker_count == 36
    assert local_decision.gates
    assert all(gate.status is GateStatus.FAIL for gate in local_decision.gates)
    assert f"Decision: `{expected_decision.value}`" in markdown
    assert "promotable=false" in markdown
    for evidence_path in evidence_paths:
        serialized = evidence_path.read_text(encoding="utf-8")
        assert "/Users/" not in serialized
        assert "/private/tmp/" not in serialized
        assert "/private/var/" not in serialized
        assert evidence_path.name in markdown
        assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() in markdown
    for platform_label in sorted(required_platforms):
        expected_status = "missing" if platform_label in missing_platforms else "collected"
        assert f"| {platform_label} | {expected_status} |" in markdown

    for evidence in evidences:
        exact_rows = {
            row.warm_key: row
            for row in evidence.records
            if row.representation == Representation.RETAINED_EXACT_RAGGED.value
            and row.phase == TimingPhase.WARM_EXECUTION.value
            and row.status == "pass"
            and row.timed.seconds is not None
            and row.timed.seconds > 0
        }
        packed_rows = (
            row
            for row in evidence.records
            if row.representation == Representation.PACKED_CANDIDATE.value
            and row.phase == TimingPhase.WARM_EXECUTION.value
            and row.status == "pass"
            and row.timed.seconds is not None
            and row.timed.seconds > 0
        )
        for packed in packed_rows:
            exact = exact_rows.get(packed.warm_key)
            assert exact is not None, f"missing exact row for {packed.record_id}"
            assert packed.timed.seconds is not None
            assert exact.timed.seconds is not None
            ratio = packed.timed.seconds / exact.timed.seconds
            assert packed.record_id in markdown
            assert exact.record_id in markdown
            assert f"ratio={ratio:.6f}" in markdown
