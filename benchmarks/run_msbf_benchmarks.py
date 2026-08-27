# pattern: Imperative Shell

"""Drive reproducible baseline-versus-candidate MSBF benchmarks."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from msbf_cases import default_case_specs

METRICS = (
    "init_ns",
    "find_ns",
    "total_ns",
    "end_to_end_ns",
    "peak_rss_bytes",
    "peak_rss_after_init_bytes",
)


def _run_worker(python, worker, case):
    command = [python, str(worker), "--case-json", json.dumps(case, separators=(",", ":"))]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Benchmark worker failed ({' '.join(command)}):\n{error.stderr}") from error
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Benchmark worker produced no JSON: {' '.join(command)}")
    return json.loads(lines[-1])


def _distribution(values):
    values = sorted(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": statistics.median(values),
        "iqr": quartiles[2] - quartiles[0],
        "minimum": values[0],
    }


def _summarize(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        if not row["warmup"]:
            grouped[(row["case"]["name"], row["implementation"])].append(row)

    summaries = []
    for (case_name, implementation), rows in grouped.items():
        first = rows[0]
        summary = {
            "case": first["case"],
            "implementation": implementation,
            "repetitions": len(rows),
            "input_nodes": first["input_nodes"],
            "input_edges": first["input_edges"],
            "output_nodes": first["output_nodes"],
            "output_edges": first["output_edges"],
            "factor_nodes": first["factor_nodes"],
        }
        for metric in METRICS:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            summary[metric] = _distribution(values) if values else None
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (row["case"]["name"], row["implementation"]))


def _log_slope(points):
    xs = [math.log(point[0]) for point in points]
    ys = [math.log(point[1]) for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _evaluate_gates(summaries):
    by_key = {(row["case"]["name"], row["implementation"]): row for row in summaries}
    adversarial = sorted(
        [row for row in summaries if row["implementation"] == "candidate" and row["case"]["family"] == "adversarial"],
        key=lambda row: row["input_edges"],
    )
    if not adversarial:
        runtime_ratios = {}
        rss_ratios = {}
        case_names = sorted({row["case"]["name"] for row in summaries})
        for name in case_names:
            baseline = by_key[(name, "baseline")]
            candidate = by_key[(name, "candidate")]
            runtime_ratios[name] = candidate["total_ns"]["median"] / baseline["total_ns"]["median"]
            rss_ratios[name] = candidate["peak_rss_bytes"]["median"] / baseline["peak_rss_bytes"]["median"]
        return {
            "candidate_faster_at_two_largest_adversarial_sizes": None,
            "candidate_adversarial_log_log_slope": None,
            "candidate_near_linear_slope_at_most_1_15": None,
            "representative_runtime_ratios": runtime_ratios,
            "representative_runtime_regression_at_most_10_percent": all(
                ratio <= 1.10 for ratio in runtime_ratios.values()
            ),
            "representative_peak_rss_ratios": rss_ratios,
            "representative_peak_rss_regression_at_most_10_percent": all(
                ratio <= 1.10 for ratio in rss_ratios.values()
            ),
        }

    largest_two = adversarial[-2:]
    faster_largest = all(
        row["total_ns"]["median"] < by_key[(row["case"]["name"], "baseline")]["total_ns"]["median"]
        for row in largest_two
    )
    slope = _log_slope([(row["input_edges"], row["total_ns"]["median"]) for row in adversarial])

    representative_names = {
        "dense-k128-f1024",
        "dense-k16-f4096",
        "nested-overlapping-s512",
        "fixture-1kg-small-end-to-end",
    }
    runtime_ratios = {}
    rss_ratios = {}
    for name in representative_names:
        baseline = by_key.get((name, "baseline"))
        candidate = by_key.get((name, "candidate"))
        if baseline is None or candidate is None:
            continue
        metric = "end_to_end_ns" if candidate["case"]["family"] == "fixture_end_to_end" else "total_ns"
        runtime_ratios[name] = candidate[metric]["median"] / baseline[metric]["median"]
        rss_ratios[name] = candidate["peak_rss_bytes"]["median"] / baseline["peak_rss_bytes"]["median"]
    return {
        "candidate_faster_at_two_largest_adversarial_sizes": faster_largest,
        "candidate_adversarial_log_log_slope": slope,
        "candidate_near_linear_slope_at_most_1_15": slope <= 1.15,
        "representative_runtime_ratios": runtime_ratios,
        "representative_runtime_regression_at_most_10_percent": all(ratio <= 1.10 for ratio in runtime_ratios.values()),
        "representative_peak_rss_ratios": rss_ratios,
        "representative_peak_rss_regression_at_most_10_percent": all(ratio <= 1.10 for ratio in rss_ratios.values()),
    }


def _interpreter_metadata(python):
    code = (
        "import json,platform,sys; import Cython,numpy,scipy; "
        "print(json.dumps({'executable':sys.executable,'python':sys.version,"
        "'platform':platform.platform(),'cython':Cython.__version__,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__}))"
    )
    return json.loads(subprocess.run([python, "-c", code], check=True, capture_output=True, text=True).stdout)


def _system_metadata():
    import psutil

    compiler = subprocess.run(["cc", "--version"], check=False, capture_output=True, text=True).stdout.splitlines()
    memory = psutil.virtual_memory().total
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        memory_result = subprocess.run(["sysctl", "-n", "hw.memsize"], check=False, capture_output=True, text=True)
        cpu_result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if memory_result.returncode == 0:
            memory = int(memory_result.stdout.strip())
        if cpu_result.returncode == 0 and cpu_result.stdout.strip():
            cpu = cpu_result.stdout.strip()
        if cpu.lower() in {"arm", "arm64", "x86_64", platform.machine().lower()}:
            hardware = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in hardware.stdout.splitlines():
                label, separator, value = line.strip().partition(":")
                if separator and label in {"Chip", "Processor Name"}:
                    cpu = value.strip()
                    break
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": memory,
        "compiler": compiler,
    }


def _write_raw_csv(path, rows):
    columns = [
        "implementation",
        "warmup",
        "repetition",
        "case_name",
        "family",
        "case_json",
        "init_ns",
        "find_ns",
        "total_ns",
        "end_to_end_ns",
        "peak_rss_bytes",
        "peak_rss_after_init_bytes",
        "input_nodes",
        "input_edges",
        "output_nodes",
        "output_edges",
        "factor_nodes",
        "source_read_ns",
        "carrier_reconstruction_ns",
        "brick_graph_ns",
        "source_samples",
        "source_variants",
        "genotype_nnz",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flattened = {key: row.get(key) for key in columns}
            flattened["case_name"] = row["case"]["name"]
            flattened["family"] = row["case"]["family"]
            flattened["case_json"] = json.dumps(row["case"], sort_keys=True)
            writer.writerow(flattened)


def _plot_svg(path, summaries, metric, title, y_label):
    width, height = 760, 480
    left, right, top, bottom = 85, 30, 45, 70
    rows = [row for row in summaries if row["case"]["family"] == "adversarial"]
    series = {
        implementation: sorted(
            [row for row in rows if row["implementation"] == implementation],
            key=lambda row: row["input_edges"],
        )
        for implementation in ("baseline", "candidate")
    }
    values = [
        (row["input_edges"], row[metric]["median"])
        for implementation_rows in series.values()
        for row in implementation_rows
    ]
    min_x, max_x = min(x for x, _ in values), max(x for x, _ in values)
    min_y, max_y = min(y for _, y in values), max(y for _, y in values)
    x0, x1 = math.log10(min_x), math.log10(max_x)
    y0, y1 = math.log10(min_y), math.log10(max_y)
    if y0 == y1:
        y1 += 1

    def point(x, y):
        px = left + (math.log10(x) - x0) / (x1 - x0) * (width - left - right)
        py = top + (y1 - math.log10(y)) / (y1 - y0) * (height - top - bottom)
        return px, py

    colors = {"baseline": "#666666", "candidate": "#d97706"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width / 2}" y="{height-20}" text-anchor="middle" '
        'font-family="sans-serif">input edges (log scale)</text>',
        f'<text x="20" y="{height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {height / 2})" font-family="sans-serif">'
        f"{y_label} (log scale)</text>",
    ]
    for implementation, implementation_rows in series.items():
        points = [point(row["input_edges"], row[metric]["median"]) for row in implementation_rows]
        encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{encoded}" fill="none" stroke="{colors[implementation]}" stroke-width="3"/>')
        parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[implementation]}"/>' for x, y in points)
    parts.extend(
        [
            f'<line x1="{width-180}" y1="55" x2="{width-150}" y2="55" stroke="{colors["baseline"]}" stroke-width="3"/>',
            f'<text x="{width-140}" y="60" font-family="sans-serif">baseline</text>',
            f'<line x1="{width-180}" y1="78" x2="{width-150}" y2="78" '
            f'stroke="{colors["candidate"]}" stroke-width="3"/>',
            f'<text x="{width-140}" y="83" font-family="sans-serif">candidate</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts))


def _format_ms(nanoseconds):
    return f"{nanoseconds / 1_000_000:.3f}"


def _format_mib(byte_count):
    return f"{byte_count / (1024 * 1024):.1f}"


def _write_report(path, metadata, summaries, gates, command):
    by_case = defaultdict(dict)
    for row in summaries:
        by_case[row["case"]["name"]][row["implementation"]] = row
    lines = [
        "# MSBF benchmark report",
        "",
        f"Generated: `{metadata['generated_at']}`",
        "",
        "## Reproduction",
        "",
        f"Command: `{command}`",
        "",
        f"Baseline ({metadata['baseline_label']}): `{metadata['baseline_sha']}`",
        "",
        f"Candidate ({metadata['candidate_label']}): `{metadata['candidate_sha']}`",
        "",
        f"Candidate build-base SHA: `{metadata['candidate_base_sha']}`",
        "",
        "Candidate `recombination.pyx` SHA-256: " f"`{metadata['candidate_source_sha256']}`",
        "",
        f"Candidate extension SHA-256: `{metadata['candidate_extension_sha256']}`",
        "",
        f"Baseline wheel SHA-256: `{metadata['baseline_wheel_sha256']}`",
        "",
        f"Candidate wheel SHA-256: `{metadata['candidate_wheel_sha256']}`",
        "",
        metadata["comparison_description"],
        "",
        "Environment construction commands and the complete dependency snapshot are in "
        "[`benchmarks/README.md`](../../README.md) and "
        "[`environment.txt`](../../environment.txt), respectively.",
        "",
        f"Repetitions: one warm-up and {metadata['repetitions']} measured fresh subprocesses per implementation/case.",
        "",
        f"OS: `{metadata['system']['os']}`",
        "",
        f"CPU: `{metadata['system']['cpu']}` "
        f"({metadata['system']['physical_cpus']} physical, "
        f"{metadata['system']['logical_cpus']} logical CPUs)",
        "",
        f"RAM: `{metadata['system']['ram_bytes']}` bytes",
        "",
        f"Compiler: `{' | '.join(metadata['system']['compiler'])}`",
        "",
        f"Baseline runtime: `{json.dumps(metadata['baseline_runtime'], sort_keys=True)}`",
        "",
        f"Candidate runtime: `{json.dumps(metadata['candidate_runtime'], sort_keys=True)}`",
        "",
        "Each worker constructs its input before timing. `total_ns` starts "
        "immediately before `Recombination.from_graph`, includes all candidate "
        "initialization, and ends after `find_recombinations`. Peak RSS is the "
        "OS-level process high-water mark sampled after the measured stage.",
        "",
        "## Runtime",
        "",
        "| Case | Impl | Input edges | Output edges | Factors | "
        "Init median/IQR/min (ms) | Find median/IQR/min (ms) | "
        "Total or E2E median/IQR/min (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name in sorted(by_case):
        for implementation in ("baseline", "candidate"):
            row = by_case[case_name][implementation]
            display_name = metadata[f"{implementation}_label"]
            primary = row["end_to_end_ns"] or row["total_ns"]

            def timing(metric):
                if metric is None:
                    return "—"
                return "/".join(_format_ms(metric[key]) for key in ("median", "iqr", "minimum"))

            lines.append(
                f"| {case_name} | {display_name} | {row['input_edges']} | "
                f"{row['output_edges']} | {row['factor_nodes']} | "
                f"{timing(row['init_ns'])} | {timing(row['find_ns'])} | "
                f"{timing(primary)} |"
            )
    runtime_ratios = json.dumps(gates["representative_runtime_ratios"], sort_keys=True)
    rss_ratios = json.dumps(gates["representative_peak_rss_ratios"], sort_keys=True)
    slope = gates["candidate_adversarial_log_log_slope"]
    slope_line = (
        "- Adversarial scaling: `not evaluated for this case set`"
        if slope is None
        else "- Candidate adversarial log-log runtime slope: "
        f"`{slope:.3f}` (near-linear gate: "
        f"`{gates['candidate_near_linear_slope_at_most_1_15']}`)"
    )
    lines.extend(
        [
            "",
            "## Peak RSS",
            "",
            "| Case | Impl | Median/IQR/min (MiB) |",
            "|---|---:|---:|",
        ]
    )
    for case_name in sorted(by_case):
        for implementation in ("baseline", "candidate"):
            rss = by_case[case_name][implementation]["peak_rss_bytes"]
            display_name = metadata[f"{implementation}_label"]
            formatted = "/".join(_format_mib(rss[key]) for key in ("median", "iqr", "minimum"))
            lines.append(f"| {case_name} | {display_name} | {formatted} |")
    lines.extend(
        [
            "",
            "## Completion gates",
            "",
            "- Candidate faster at the two largest adversarial sizes: "
            f"`{gates['candidate_faster_at_two_largest_adversarial_sizes']}`",
            slope_line,
            f"- Representative runtime ratios candidate/baseline: `{runtime_ratios}` "
            f"(gate: `{gates['representative_runtime_regression_at_most_10_percent']}`)",
            f"- Representative peak-RSS ratios candidate/baseline: `{rss_ratios}` "
            f"(gate: `{gates['representative_peak_rss_regression_at_most_10_percent']}`)",
            "",
            "## Plots and raw data",
            "",
            "- [Raw JSON](raw.json)",
            "- [Raw CSV](raw.csv)",
            "- [Summary JSON](summary.json)",
            "",
        ]
    )
    if metadata["has_adversarial_cases"]:
        plot_index = lines.index("- [Raw JSON](raw.json)")
        lines[plot_index:plot_index] = [
            "- [Adversarial runtime scaling](adversarial-runtime.svg)",
            "- [Adversarial peak RSS scaling](adversarial-rss.svg)",
        ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-python", required=True)
    parser.add_argument("--candidate-python", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--candidate-base-sha", required=True)
    parser.add_argument("--candidate-source-sha256", required=True)
    parser.add_argument("--candidate-extension-sha256", required=True)
    parser.add_argument("--baseline-wheel-sha256", required=True)
    parser.add_argument("--candidate-wheel-sha256", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--comparison-description",
        default=(
            "Both environments use the clean baseline package and identical dependencies; "
            "the candidate environment replaces only the ABI-compatible recombination extension. "
            "This isolates MSBF from unrelated branch-level import and dependency changes."
        ),
    )
    parser.add_argument("--fixture-vcf", type=Path, default=Path("tests/testdata/1kg_small.vcf"))
    parser.add_argument("--external-vcf", type=Path)
    parser.add_argument("--external-linarg", type=Path)
    parser.add_argument("--linarg-block", action="append", default=[])
    parser.add_argument("--only-linarg", action="store_true")
    parser.add_argument("--real-data-only", action="store_true")
    parser.add_argument("--allow-short-run", action="store_true")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--overwrite-results", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 7 and not args.allow_short_run:
        parser.error("--repetitions must be at least 7")
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")

    worker = Path(__file__).with_name("msbf_worker.py").resolve()
    fixture = args.fixture_vcf.resolve() if args.fixture_vcf else None
    external = args.external_vcf.resolve() if args.external_vcf else None
    external_linarg = args.external_linarg.resolve() if args.external_linarg else None
    if bool(external_linarg) != bool(args.linarg_block):
        parser.error("--external-linarg and at least one --linarg-block must be used together")
    cases = default_case_specs(
        None if args.only_linarg else fixture,
        None if args.only_linarg else external,
        external_linarg,
        args.linarg_block,
    )
    if args.only_linarg:
        cases = [case for case in cases if case["family"] == "linarg_reconstructed_recombination"]
    if args.real_data_only:
        cases = [
            case
            for case in cases
            if case["family"] == "linarg_reconstructed_recombination" or case["name"] == "external-vcf-recombination"
        ]
    generated_at = datetime.now(timezone.utc).isoformat()
    results_dir = args.results_dir or Path("benchmarks/results") / generated_at.replace(":", "-")
    if results_dir.exists() and not args.overwrite_results:
        parser.error(f"results directory already exists: {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=args.overwrite_results)

    implementations = {
        # Preserve venv symlinks: resolving them selects the base interpreter
        # and loses the environment's site-packages.
        "baseline": str(Path(args.baseline_python).absolute()),
        "candidate": str(Path(args.candidate_python).absolute()),
    }
    raw_rows = []
    for case in cases:
        for implementation, python in implementations.items():
            warmup = _run_worker(python, worker, case)
            warmup.update({"implementation": implementation, "warmup": True, "repetition": -1})
            raw_rows.append(warmup)
        for repetition in range(args.repetitions):
            order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
            for implementation in order:
                row = _run_worker(implementations[implementation], worker, case)
                row.update(
                    {
                        "implementation": implementation,
                        "warmup": False,
                        "repetition": repetition,
                    }
                )
                raw_rows.append(row)
                print(
                    f"{case['name']} {implementation} repetition={repetition + 1}/{args.repetitions}",
                    flush=True,
                )

    summaries = _summarize(raw_rows)
    gates = _evaluate_gates(summaries)
    metadata = {
        "generated_at": generated_at,
        "baseline_sha": args.baseline_sha,
        "candidate_sha": args.candidate_sha,
        "candidate_base_sha": args.candidate_base_sha,
        "candidate_source_sha256": args.candidate_source_sha256,
        "candidate_extension_sha256": args.candidate_extension_sha256,
        "baseline_wheel_sha256": args.baseline_wheel_sha256,
        "candidate_wheel_sha256": args.candidate_wheel_sha256,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "comparison_description": args.comparison_description,
        "repetitions": args.repetitions,
        "system": _system_metadata(),
        "baseline_runtime": _interpreter_metadata(implementations["baseline"]),
        "candidate_runtime": _interpreter_metadata(implementations["candidate"]),
        "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
        "has_adversarial_cases": any(case["family"] == "adversarial" for case in cases),
    }
    command = shlex.join([sys.executable, *sys.argv])
    (results_dir / "raw.json").write_text(json.dumps({"metadata": metadata, "rows": raw_rows}, indent=2))
    _write_raw_csv(results_dir / "raw.csv", raw_rows)
    (results_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summaries": summaries, "gates": gates}, indent=2)
    )
    if metadata["has_adversarial_cases"]:
        _plot_svg(
            results_dir / "adversarial-runtime.svg",
            summaries,
            "total_ns",
            "Adversarial recombination runtime",
            "nanoseconds",
        )
        _plot_svg(
            results_dir / "adversarial-rss.svg",
            summaries,
            "peak_rss_bytes",
            "Adversarial process peak RSS",
            "bytes",
        )
    _write_report(results_dir / "report.md", metadata, summaries, gates, command)
    print(results_dir.resolve())


if __name__ == "__main__":
    main()
