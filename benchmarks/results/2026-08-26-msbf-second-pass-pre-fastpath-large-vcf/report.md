# MSBF benchmark report

Generated: `2026-08-27T00:29:29.399455+00:00`

## Reproduction

Command: `/private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python benchmarks/run_msbf_benchmarks.py --baseline-python /private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python --candidate-python /private/tmp/msbf-benchmark/venv-msbf-second-pass/bin/python --baseline-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-base-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-extension-sha256 598175fb790c0c3472d41e51dc70c213fd1a72cbeb72d432b767e1955d9be1f2 --baseline-wheel-sha256 ec00bfaac17632ae868cbfd3605d2adfbcca4ca63e36f22d36c507fb971da747 --candidate-wheel-sha256 8aee98edadb2f0ce0dbc822d7a04fc68d8559c9ad6565feaacb9e14f5f611e96 --baseline-label compact-init --candidate-label second-pass-final --comparison-description 'Both environments contain the first-pass optimized package and identical dependencies; the candidate replaces only the ABI-compatible recombination extension with the completed second pass.' --external-vcf /Users/nicholas/Projects/linear-dag/inputs/large0.vcf.gz --real-data-only --allow-short-run --repetitions 3 --results-dir benchmarks/results/2026-08-26-msbf-second-pass-final-large-vcf`

Baseline (compact-init): `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate (second-pass-final): `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate build-base SHA: `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate extension SHA-256: `598175fb790c0c3472d41e51dc70c213fd1a72cbeb72d432b767e1955d9be1f2`

Baseline wheel SHA-256: `ec00bfaac17632ae868cbfd3605d2adfbcca4ca63e36f22d36c507fb971da747`

Candidate wheel SHA-256: `8aee98edadb2f0ce0dbc822d7a04fc68d8559c9ad6565feaacb9e14f5f611e96`

Both environments contain the first-pass optimized package and identical dependencies; the candidate replaces only the ABI-compatible recombination extension with the completed second pass.

Environment construction commands and the complete dependency snapshot are in [`benchmarks/README.md`](../../README.md) and [`environment.txt`](../../environment.txt), respectively.

Repetitions: one warm-up and 3 measured fresh subprocesses per implementation/case.

OS: `macOS-26.4.1-arm64-arm-64bit`

CPU: `Apple M1` (8 physical, 8 logical CPUs)

RAM: `17179869184` bytes

Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102) | Target: arm64-apple-darwin25.4.0 | Thread model: posix | InstalledDir: /Library/Developer/CommandLineTools/usr/bin`

Baseline runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Candidate runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-msbf-second-pass/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Each worker constructs its input before timing. `total_ns` starts immediately before `Recombination.from_graph`, includes all candidate initialization, and ends after `find_recombinations`. Peak RSS is the OS-level process high-water mark sampled after the measured stage.

## Runtime

| Case | Impl | Input edges | Output edges | Factors | Init median/IQR/min (ms) | Find median/IQR/min (ms) | Total or E2E median/IQR/min (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| external-vcf-recombination | compact-init | 20784669 | 2286719 | 382751 | 3896.655/143.289/3819.817 | 5750.118/381.179/5603.942 | 9646.899/525.382/9425.822 |
| external-vcf-recombination | second-pass-final | 20784669 | 2286719 | 382751 | 3775.992/190.088/3570.203 | 5595.827/301.645/5494.621 | 9546.305/198.744/9270.721 |

## Peak RSS

| Case | Impl | Median/IQR/min (MiB) |
|---|---:|---:|
| external-vcf-recombination | compact-init | 3179.0/156.1/3017.2 |
| external-vcf-recombination | second-pass-final | 2992.5/310.5/2973.1 |

## Completion gates

- Candidate faster at the two largest adversarial sizes: `None`
- Adversarial scaling: `not evaluated for this case set`
- Representative runtime ratios candidate/baseline: `{"external-vcf-recombination": 0.9895724259710419}` (gate: `True`)
- Representative peak-RSS ratios candidate/baseline: `{"external-vcf-recombination": 0.9413198727999252}` (gate: `True`)

## Plots and raw data

- [Raw JSON](raw.json)
- [Raw CSV](raw.csv)
- [Summary JSON](summary.json)
