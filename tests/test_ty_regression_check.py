# pattern: Functional Core

from scripts.check_ty_no_regression import new_diagnostics, normalize_diagnostics


def test_ty_baseline_normalization_ignores_line_shifts_but_preserves_multiplicity() -> None:
    baseline = "\n".join(
        (
            "tests/example.py:10:2: error[invalid-argument-type] Expected int",
            "tests/example.py:12:2: error[invalid-argument-type] Expected int",
            "Found 2 diagnostics",
        )
    )
    current = "\n".join(
        (
            "tests/example.py:20:2: error[invalid-argument-type] Expected int",
            "tests/example.py:22:2: error[invalid-argument-type] Expected int",
            "Found 2 diagnostics",
        )
    )

    assert normalize_diagnostics(baseline) == normalize_diagnostics(current)
    assert not new_diagnostics(baseline, current)


def test_ty_baseline_comparison_reports_only_new_diagnostics() -> None:
    baseline = "src/example.py:1:1: error[unresolved-import] Missing dependency\nFound 1 diagnostic"
    current = "\n".join(
        (
            "src/example.py:7:1: error[unresolved-import] Missing dependency",
            "tests/new.py:3:4: error[invalid-argument-type] Expected int",
            "Found 2 diagnostics",
        )
    )

    assert new_diagnostics(baseline, current) == {
        "tests/new.py: error[invalid-argument-type] Expected int": 1,
    }
