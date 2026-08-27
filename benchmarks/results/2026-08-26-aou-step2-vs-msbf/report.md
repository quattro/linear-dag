# MSBF benchmark report

Generated: `2026-08-26T22:51:14.036549+00:00`

## Reproduction

Command: `/Users/nicholas/Projects/linear-dag/.venv/bin/python benchmarks/run_msbf_benchmarks.py --baseline-python /private/tmp/msbf-benchmark/venv-aou-step2/bin/python --candidate-python /private/tmp/msbf-benchmark/venv-candidate/bin/python --baseline-sha d8ef3bd3089caa9944803fec2bd0b48d8cd01d05 --candidate-sha 554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406 --candidate-base-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-extension-sha256 b22223af4ead0c2785db9657ada64f956f17178e7868eb0ad7278db94bb9c5b6 --baseline-wheel-sha256 e15ac891eedbab4d41bdcbe8fef920b094156ec28284dd62113d9e486667cb39 --candidate-wheel-sha256 78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183 --baseline-label aou-step2 --candidate-label msbf --comparison-description 'The baseline environment contains the complete wheel from remote branch codex/aou-step2-sparse-workspace at the recorded SHA. The candidate environment contains the clean main package with only the ABI-compatible final MSBF recombination extension replaced. Both use identical runtime dependencies, Python, compiler, and hardware; synthetic recombination cases compare the algorithms directly, while the fixture end-to-end case also includes other branch-level pipeline differences.' --repetitions 7 --results-dir benchmarks/results/2026-08-26-aou-step2-vs-msbf`

Baseline (aou-step2): `d8ef3bd3089caa9944803fec2bd0b48d8cd01d05`

Candidate (msbf): `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406`

Candidate build-base SHA: `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate extension SHA-256: `b22223af4ead0c2785db9657ada64f956f17178e7868eb0ad7278db94bb9c5b6`

Baseline wheel SHA-256: `e15ac891eedbab4d41bdcbe8fef920b094156ec28284dd62113d9e486667cb39`

Candidate wheel SHA-256: `78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183`

The baseline environment contains the complete wheel from remote branch codex/aou-step2-sparse-workspace at the recorded SHA. The candidate environment contains the clean main package with only the ABI-compatible final MSBF recombination extension replaced. Both use identical runtime dependencies, Python, compiler, and hardware; synthetic recombination cases compare the algorithms directly, while the fixture end-to-end case also includes other branch-level pipeline differences.

Environment construction commands and the complete dependency snapshot are in [`benchmarks/README.md`](../../README.md) and [`environment.txt`](../../environment.txt), respectively.

Repetitions: one warm-up and 7 measured fresh subprocesses per implementation/case.

OS: `macOS-26.4.1-arm64-arm-64bit`

CPU: `Apple M1` (8 physical, 8 logical CPUs)

RAM: `17179869184` bytes

Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102) | Target: arm64-apple-darwin25.4.0 | Thread model: posix | InstalledDir: /Library/Developer/CommandLineTools/usr/bin`

Baseline runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-aou-step2/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Candidate runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-candidate/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Each worker constructs its input before timing. `total_ns` starts immediately before `Recombination.from_graph`, includes all candidate initialization, and ends after `find_recombinations`. Peak RSS is the OS-level process high-water mark sampled after the measured stage.

## Runtime

| Case | Impl | Input edges | Output edges | Factors | Init median/IQR/min (ms) | Find median/IQR/min (ms) | Total or E2E median/IQR/min (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial-q128 | aou-step2 | 98176 | 57536 | 128 | 5.979/0.035/5.939 | 10.637/0.450/10.454 | 16.609/0.389/16.462 |
| adversarial-q128 | msbf | 98176 | 57536 | 128 | 4.313/0.140/4.035 | 1.925/0.045/1.887 | 6.277/0.157/5.935 |
| adversarial-q16 | aou-step2 | 1520 | 920 | 16 | 0.156/0.010/0.144 | 0.219/0.004/0.217 | 0.378/0.015/0.367 |
| adversarial-q16 | msbf | 1520 | 920 | 16 | 0.068/0.003/0.065 | 0.035/0.001/0.034 | 0.109/0.004/0.104 |
| adversarial-q256 | aou-step2 | 392960 | 229760 | 256 | 24.940/1.236/24.658 | 73.347/2.843/71.435 | 99.246/3.705/96.606 |
| adversarial-q256 | msbf | 392960 | 229760 | 256 | 20.180/2.116/18.357 | 8.004/0.274/7.948 | 28.587/2.357/26.325 |
| adversarial-q32 | aou-step2 | 6112 | 3632 | 32 | 0.400/0.017/0.391 | 0.650/0.006/0.637 | 1.056/0.024/1.043 |
| adversarial-q32 | msbf | 6112 | 3632 | 32 | 0.248/0.020/0.244 | 0.127/0.001/0.125 | 0.381/0.021/0.375 |
| adversarial-q64 | aou-step2 | 24512 | 14432 | 64 | 1.452/0.038/1.437 | 2.391/0.104/2.315 | 3.905/0.127/3.779 |
| adversarial-q64 | msbf | 24512 | 14432 | 64 | 0.994/0.020/0.967 | 0.488/0.012/0.481 | 1.508/0.039/1.455 |
| dense-k128-f1024 | aou-step2 | 131072 | 1279 | 127 | 9.897/0.456/8.412 | 17.375/1.097/15.773 | 27.493/1.289/25.418 |
| dense-k128-f1024 | msbf | 131072 | 1152 | 1 | 7.305/0.702/6.689 | 3.708/0.417/3.548 | 11.089/1.182/10.250 |
| dense-k16-f1024 | aou-step2 | 16384 | 1055 | 15 | 1.017/0.072/0.967 | 1.496/0.095/1.466 | 2.518/0.221/2.438 |
| dense-k16-f1024 | msbf | 16384 | 1040 | 1 | 0.697/0.053/0.656 | 0.433/0.039/0.419 | 1.125/0.084/1.082 |
| dense-k16-f4096 | aou-step2 | 65536 | 4127 | 15 | 4.234/0.386/3.822 | 6.496/0.757/5.879 | 10.614/0.942/9.709 |
| dense-k16-f4096 | msbf | 65536 | 4112 | 1 | 2.790/0.231/2.657 | 1.928/0.235/1.719 | 4.777/0.321/4.404 |
| dense-k16-f512 | aou-step2 | 8192 | 543 | 15 | 0.539/0.048/0.509 | 0.826/0.082/0.800 | 1.370/0.133/1.322 |
| dense-k16-f512 | msbf | 8192 | 528 | 1 | 0.365/0.032/0.318 | 0.216/0.022/0.207 | 0.590/0.054/0.531 |
| dense-k16-f64 | aou-step2 | 1024 | 95 | 15 | 0.125/0.015/0.114 | 0.191/0.020/0.178 | 0.323/0.036/0.301 |
| dense-k16-f64 | msbf | 1024 | 80 | 1 | 0.044/0.002/0.041 | 0.025/0.001/0.024 | 0.073/0.003/0.071 |
| dense-k2-f1024 | aou-step2 | 2048 | 1027 | 1 | 0.182/0.024/0.172 | 0.054/0.005/0.052 | 0.241/0.028/0.230 |
| dense-k2-f1024 | msbf | 2048 | 1026 | 1 | 0.088/0.004/0.085 | 0.039/0.000/0.038 | 0.132/0.005/0.129 |
| fixture-1kg-small-end-to-end | aou-step2 | 13571 | 6017 | 1205 | — | — | 41.452/4.617/40.730 |
| fixture-1kg-small-end-to-end | msbf | 13571 | 5873 | 1038 | — | — | 33.643/2.057/29.818 |
| fixture-1kg-small-recombination | aou-step2 | 13571 | 6017 | 1205 | 1.085/0.106/1.058 | 10.694/0.791/10.455 | 11.767/0.890/11.528 |
| fixture-1kg-small-recombination | msbf | 13571 | 5873 | 1038 | 0.618/0.041/0.591 | 1.285/0.064/1.103 | 1.912/0.062/1.702 |
| nested-overlapping-s512 | aou-step2 | 7680 | 1555 | 9 | 0.515/0.063/0.481 | 0.440/0.066/0.429 | 0.954/0.127/0.915 |
| nested-overlapping-s512 | msbf | 7680 | 1550 | 5 | 0.362/0.044/0.328 | 0.335/0.067/0.330 | 0.713/0.119/0.663 |

## Peak RSS

| Case | Impl | Median/IQR/min (MiB) |
|---|---:|---:|
| adversarial-q128 | aou-step2 | 114.7/0.1/114.6 |
| adversarial-q128 | msbf | 119.1/0.0/119.0 |
| adversarial-q16 | aou-step2 | 92.8/0.1/92.6 |
| adversarial-q16 | msbf | 92.7/0.0/92.6 |
| adversarial-q256 | aou-step2 | 170.7/0.1/170.7 |
| adversarial-q256 | msbf | 190.5/0.0/190.4 |
| adversarial-q32 | aou-step2 | 93.4/0.1/93.3 |
| adversarial-q32 | msbf | 93.6/0.1/93.5 |
| adversarial-q64 | aou-step2 | 98.2/0.1/98.1 |
| adversarial-q64 | msbf | 99.4/0.1/99.3 |
| dense-k128-f1024 | aou-step2 | 117.3/0.0/117.2 |
| dense-k128-f1024 | msbf | 122.8/0.0/122.7 |
| dense-k16-f1024 | aou-step2 | 95.4/0.0/95.4 |
| dense-k16-f1024 | msbf | 95.7/0.0/95.6 |
| dense-k16-f4096 | aou-step2 | 106.6/0.0/106.4 |
| dense-k16-f4096 | msbf | 107.9/0.0/107.8 |
| dense-k16-f512 | aou-step2 | 93.7/0.0/93.6 |
| dense-k16-f512 | msbf | 93.8/0.0/93.7 |
| dense-k16-f64 | aou-step2 | 92.6/0.1/92.5 |
| dense-k16-f64 | msbf | 92.6/0.1/92.5 |
| dense-k2-f1024 | aou-step2 | 92.7/0.0/92.7 |
| dense-k2-f1024 | msbf | 92.8/0.1/92.7 |
| fixture-1kg-small-end-to-end | aou-step2 | 98.9/0.1/98.8 |
| fixture-1kg-small-end-to-end | msbf | 99.6/0.1/99.5 |
| fixture-1kg-small-recombination | aou-step2 | 98.7/0.2/98.5 |
| fixture-1kg-small-recombination | msbf | 99.1/0.1/99.0 |
| nested-overlapping-s512 | aou-step2 | 93.4/0.1/93.3 |
| nested-overlapping-s512 | msbf | 93.7/0.0/93.6 |

## Completion gates

- Candidate faster at the two largest adversarial sizes: `True`
- Candidate adversarial log-log runtime slope: `1.004` (near-linear gate: `True`)
- Representative runtime ratios candidate/baseline: `{"dense-k128-f1024": 0.4033551216784389, "dense-k16-f4096": 0.4500276621211345, "fixture-1kg-small-end-to-end": 0.8116232223200608, "nested-overlapping-s512": 0.7473241578188491}` (gate: `True`)
- Representative peak-RSS ratios candidate/baseline: `{"dense-k128-f1024": 1.0468958166799893, "dense-k16-f4096": 1.0121700879765396, "fixture-1kg-small-end-to-end": 1.0067930489731438, "nested-overlapping-s512": 1.0036807763091853}` (gate: `True`)

## Plots and raw data

- [Adversarial runtime scaling](adversarial-runtime.svg)
- [Adversarial peak RSS scaling](adversarial-rss.svg)
- [Raw JSON](raw.json)
- [Raw CSV](raw.csv)
- [Summary JSON](summary.json)
