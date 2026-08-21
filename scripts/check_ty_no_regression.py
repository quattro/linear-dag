# pattern: Imperative Shell

"""Fail when type diagnostics exceed the pre-Phase 7 diagnostic baseline."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile

from collections import Counter
from pathlib import Path

BASE_COMMIT = "948f6bf"
_DIAGNOSTIC = re.compile(r"^(?P<path>.+?):\d+:\d+: (?P<message>(?:error|warning)\[[^]]+\].*)$")


def normalize_diagnostics(output: str) -> Counter[str]:
    """Normalize line/column movement while retaining path, rule, text, and count."""
    diagnostics: Counter[str] = Counter()
    for line in output.splitlines():
        match = _DIAGNOSTIC.match(line)
        if match is not None:
            diagnostics[f"{match['path']}: {match['message']}"] += 1
    return diagnostics


def new_diagnostics(baseline_output: str, current_output: str) -> dict[str, int]:
    """Return the normalized diagnostic multiset added after the baseline."""
    return dict(normalize_diagnostics(current_output) - normalize_diagnostics(baseline_output))


def _repo_root() -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _run_ty(project: Path, *, python: Path, ty: str) -> str:
    completed = subprocess.run(
        [
            ty,
            "check",
            "--project",
            str(project),
            "--python",
            str(python),
            "--output-format",
            "concise",
            "--color",
            "never",
            "--exit-zero",
            "src",
            "tests",
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-project",
        type=Path,
        default=None,
        help="existing checkout of the baseline commit (avoids creating a temporary worktree)",
    )
    args = parser.parse_args()
    repository = _repo_root()
    ty = shutil.which("ty")
    if ty is None:
        raise RuntimeError("ty is not available on PATH; run through `uv run`")
    # Keep the venv entrypoint path; resolving its symlink loses environment identity.
    python = Path(sys.executable)
    current_output = _run_ty(repository, python=python, ty=ty)
    if args.baseline_project is not None:
        baseline_output = _run_ty(args.baseline_project.resolve(), python=python, ty=ty)
    else:
        with tempfile.TemporaryDirectory(prefix="linear-dag-ty-baseline-", dir="/private/tmp") as temporary:
            baseline = Path(temporary) / "checkout"
            subprocess.run(
                ["git", "-C", str(repository), "worktree", "add", "--detach", str(baseline), BASE_COMMIT],
                check=True,
                capture_output=True,
                text=True,
            )
            try:
                baseline_output = _run_ty(baseline, python=python, ty=ty)
            finally:
                subprocess.run(
                    ["git", "-C", str(repository), "worktree", "remove", "--force", str(baseline)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    baseline_count = sum(normalize_diagnostics(baseline_output).values())
    current_count = sum(normalize_diagnostics(current_output).values())
    additions = new_diagnostics(baseline_output, current_output)
    print(f"ty baseline {BASE_COMMIT}: {baseline_count} diagnostics; current: {current_count}")
    if additions:
        for diagnostic, count in sorted(additions.items()):
            print(f"NEW x{count}: {diagnostic}", file=sys.stderr)
        return 1
    print("No new type diagnostics relative to the pre-Phase 7 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
