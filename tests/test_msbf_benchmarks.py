# pattern: Imperative Shell

import subprocess
import sys

from pathlib import Path


def test_benchmark_driver_requires_reviewed_source_hash():
    driver = Path(__file__).parents[1] / "benchmarks" / "run_msbf_benchmarks.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--baseline-python",
            sys.executable,
            "--candidate-python",
            sys.executable,
            "--baseline-sha",
            "baseline",
            "--candidate-sha",
            "candidate",
            "--candidate-base-sha",
            "base",
            "--candidate-extension-sha256",
            "extension",
            "--baseline-wheel-sha256",
            "baseline-wheel",
            "--candidate-wheel-sha256",
            "candidate-wheel",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--candidate-source-sha256" in completed.stderr
