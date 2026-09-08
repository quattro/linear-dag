# MSBF benchmark report

Generated: `2026-08-26T23:36:16.617781+00:00`

## Reproduction

Command: `/Users/nicholas/Projects/linear-dag/.venv/bin/python benchmarks/run_msbf_benchmarks.py --baseline-python /private/tmp/msbf-benchmark/venv-candidate/bin/python --candidate-python /private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python --baseline-sha 554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406 --candidate-sha 554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-57ef15f1ba0f00d8dcc196783608a8badf25d252a203317a4388365cca8f3528 --candidate-base-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-extension-sha256 fce904e371fc7d8803dbd46be5a463a60f137da10b24898c1bab13b9740e8519 --baseline-wheel-sha256 78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183 --candidate-wheel-sha256 ec00bfaac17632ae868cbfd3605d2adfbcca4ca63e36f22d36c507fb971da747 --baseline-label msbf-before --candidate-label msbf-compact-init --comparison-description 'Before/after comparison isolating the MSBF recombination extension. The candidate scans only the live node domain, grows distinct-pair classes dynamically, uses node-bounded frequency buckets, removes redundant boundary state, and compacts index fields to 32 bits. Both environments otherwise use identical package code, dependencies, Python, compiler, hardware, worker, and reconstructed inputs.' --external-vcf inputs/large0.vcf.gz --external-linarg 1kg_chromosomes_n3202_blocks.h5 --linarg-block 1_125000001_130000000 --linarg-block 22_10000001_15000000 --real-data-only --allow-short-run --repetitions 3 --results-dir benchmarks/results/2026-08-26-msbf-boundary-init-optimized`

Baseline (msbf-before): `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406`

Candidate (msbf-compact-init): `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-57ef15f1ba0f00d8dcc196783608a8badf25d252a203317a4388365cca8f3528`

Candidate build-base SHA: `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate extension SHA-256: `fce904e371fc7d8803dbd46be5a463a60f137da10b24898c1bab13b9740e8519`

Baseline wheel SHA-256: `78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183`

Candidate wheel SHA-256: `ec00bfaac17632ae868cbfd3605d2adfbcca4ca63e36f22d36c507fb971da747`

Before/after comparison isolating the MSBF recombination extension. The candidate scans only the live node domain, grows distinct-pair classes dynamically, uses node-bounded frequency buckets, removes redundant boundary state, and compacts index fields to 32 bits. Both environments otherwise use identical package code, dependencies, Python, compiler, hardware, worker, and reconstructed inputs.

Environment construction commands and the complete dependency snapshot are in [`benchmarks/README.md`](../../README.md) and [`environment.txt`](../../environment.txt), respectively.

Repetitions: one warm-up and 3 measured fresh subprocesses per implementation/case.

OS: `macOS-26.4.1-arm64-arm-64bit`

CPU: `Apple M1` (8 physical, 8 logical CPUs)

RAM: `17179869184` bytes

Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102) | Target: arm64-apple-darwin25.4.0 | Thread model: posix | InstalledDir: /Library/Developer/CommandLineTools/usr/bin`

Baseline runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-candidate/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Candidate runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Each worker constructs its input before timing. `total_ns` starts immediately before `Recombination.from_graph`, includes all candidate initialization, and ends after `find_recombinations`. Peak RSS is the OS-level process high-water mark sampled after the measured stage.

## Runtime

| Case | Impl | Input edges | Output edges | Factors | Init median/IQR/min (ms) | Find median/IQR/min (ms) | Total or E2E median/IQR/min (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| external-vcf-recombination | msbf-before | 20784669 | 2286719 | 382751 | 5129.051/705.097/4597.736 | 7283.451/4061.595/6659.587 | 12412.693/4766.902/11257.444 |
| external-vcf-recombination | msbf-compact-init | 20784669 | 2286719 | 382751 | 3891.954/74.085/3813.756 | 6122.440/99.735/5961.031 | 10052.574/154.813/9774.874 |
| linarg-1_125000001_130000000 | msbf-before | 600591 | 160257 | 22133 | 61.853/15.759/56.842 | 142.047/1.902/141.664 | 207.339/15.952/198.522 |
| linarg-1_125000001_130000000 | msbf-compact-init | 600591 | 160257 | 22133 | 48.554/20.976/47.834 | 123.935/31.884/118.259 | 171.784/52.503/166.827 |
| linarg-22_10000001_15000000 | msbf-before | 3699752 | 1341418 | 172879 | 577.614/12.896/557.224 | 1020.573/16.681/997.905 | 1580.947/15.545/1577.818 |
| linarg-22_10000001_15000000 | msbf-compact-init | 3699752 | 1341418 | 172879 | 517.589/11.628/512.061 | 917.284/25.854/872.602 | 1434.891/37.482/1384.679 |

## Peak RSS

| Case | Impl | Median/IQR/min (MiB) |
|---|---:|---:|
| external-vcf-recombination | msbf-before | 3877.0/478.6/3099.5 |
| external-vcf-recombination | msbf-compact-init | 3553.0/206.8/3198.9 |
| linarg-1_125000001_130000000 | msbf-before | 319.4/0.2/319.3 |
| linarg-1_125000001_130000000 | msbf-compact-init | 282.3/0.1/282.2 |
| linarg-22_10000001_15000000 | msbf-before | 1410.9/0.0/1410.9 |
| linarg-22_10000001_15000000 | msbf-compact-init | 1246.8/0.1/1246.7 |

## Completion gates

- Candidate faster at the two largest adversarial sizes: `None`
- Adversarial scaling: `not evaluated for this case set`
- Representative runtime ratios candidate/baseline: `{"external-vcf-recombination": 0.8098625075717606, "linarg-1_125000001_130000000": 0.8285187543105735, "linarg-22_10000001_15000000": 0.9076146195770833}` (gate: `True`)
- Representative peak-RSS ratios candidate/baseline: `{"external-vcf-recombination": 0.9164151194328802, "linarg-1_125000001_130000000": 0.883811946577956, "linarg-22_10000001_15000000": 0.8836962058960331}` (gate: `True`)

## Plots and raw data

- [Raw JSON](raw.json)
- [Raw CSV](raw.csv)
- [Summary JSON](summary.json)
