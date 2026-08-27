# MSBF benchmark report

Generated: `2026-08-26T22:36:30.177605+00:00`

## Reproduction

Command: `/Users/nicholas/Projects/linear-dag/.venv/bin/python benchmarks/run_msbf_benchmarks.py --baseline-python /private/tmp/msbf-benchmark/venv-baseline/bin/python --candidate-python /private/tmp/msbf-benchmark/venv-candidate/bin/python --baseline-sha 554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9 --candidate-sha 554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406 --candidate-base-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-extension-sha256 b22223af4ead0c2785db9657ada64f956f17178e7868eb0ad7278db94bb9c5b6 --baseline-wheel-sha256 9df81a0064cf0a30009be8e73a3c5dc36c27645e1b871c8facc7a9708a9349f2 --candidate-wheel-sha256 78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183 --repetitions 7 --results-dir benchmarks/results/2026-08-26-msbf --overwrite-results`

Baseline: `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9`

Candidate: `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406`

Candidate build-base SHA: `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate extension SHA-256: `b22223af4ead0c2785db9657ada64f956f17178e7868eb0ad7278db94bb9c5b6`

Baseline wheel SHA-256: `9df81a0064cf0a30009be8e73a3c5dc36c27645e1b871c8facc7a9708a9349f2`

Candidate wheel SHA-256: `78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183`

Both environments use the clean baseline package and identical dependencies; the candidate environment replaces only the ABI-compatible recombination extension. This isolates MSBF from unrelated branch-level import and dependency changes.

Environment construction commands and the complete dependency snapshot are in [`benchmarks/README.md`](../../README.md) and [`environment.txt`](../../environment.txt), respectively.

Repetitions: one warm-up and 7 measured fresh subprocesses per implementation/case.

OS: `macOS-26.4.1-arm64-arm-64bit`

CPU: `Apple M1` (8 physical, 8 logical CPUs)

RAM: `17179869184` bytes

Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102) | Target: arm64-apple-darwin25.4.0 | Thread model: posix | InstalledDir: /Library/Developer/CommandLineTools/usr/bin`

Baseline runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-baseline/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Candidate runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-candidate/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Each worker constructs its input before timing. `total_ns` starts immediately before `Recombination.from_graph`, includes all candidate initialization, and ends after `find_recombinations`. Peak RSS is the OS-level process high-water mark sampled after the measured stage.

## Runtime

| Case | Impl | Input edges | Output edges | Factors | Init median/IQR/min (ms) | Find median/IQR/min (ms) | Total or E2E median/IQR/min (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial-q128 | baseline | 98176 | 57536 | 128 | 10.917/0.474/10.532 | 10.661/0.699/10.202 | 21.349/1.202/20.858 |
| adversarial-q128 | candidate | 98176 | 57536 | 128 | 4.469/0.819/4.167 | 1.919/0.117/1.878 | 6.377/1.065/6.099 |
| adversarial-q16 | baseline | 1520 | 920 | 16 | 0.266/0.011/0.260 | 0.217/0.007/0.209 | 0.484/0.019/0.479 |
| adversarial-q16 | candidate | 1520 | 920 | 16 | 0.069/0.004/0.064 | 0.035/0.002/0.035 | 0.109/0.006/0.105 |
| adversarial-q256 | baseline | 392960 | 229760 | 256 | 46.072/3.257/43.806 | 57.753/2.557/54.039 | 101.861/6.717/99.992 |
| adversarial-q256 | candidate | 392960 | 229760 | 256 | 20.134/1.451/18.806 | 9.144/1.213/7.984 | 28.895/1.515/26.840 |
| adversarial-q32 | baseline | 6112 | 3632 | 32 | 0.765/0.019/0.750 | 0.654/0.018/0.639 | 1.419/0.067/1.396 |
| adversarial-q32 | candidate | 6112 | 3632 | 32 | 0.268/0.033/0.234 | 0.127/0.016/0.124 | 0.404/0.052/0.364 |
| adversarial-q64 | baseline | 24512 | 14432 | 64 | 2.626/0.025/2.615 | 2.479/0.140/2.356 | 5.103/0.141/4.989 |
| adversarial-q64 | candidate | 24512 | 14432 | 64 | 1.013/0.063/0.988 | 0.492/0.047/0.489 | 1.514/0.112/1.489 |
| dense-k128-f1024 | baseline | 131072 | 1279 | 127 | 18.238/1.824/16.650 | 14.817/1.785/13.234 | 33.532/2.544/29.892 |
| dense-k128-f1024 | candidate | 131072 | 1152 | 1 | 8.323/1.107/7.059 | 3.707/0.678/3.458 | 11.967/1.898/10.778 |
| dense-k16-f1024 | baseline | 16384 | 1055 | 15 | 2.015/0.194/1.873 | 1.335/0.299/1.317 | 3.353/0.762/3.206 |
| dense-k16-f1024 | candidate | 16384 | 1040 | 1 | 0.798/0.097/0.667 | 0.449/0.064/0.419 | 1.284/0.168/1.092 |
| dense-k16-f4096 | baseline | 65536 | 4127 | 15 | 8.469/1.173/7.862 | 6.167/0.885/5.280 | 14.276/1.666/13.150 |
| dense-k16-f4096 | candidate | 65536 | 4112 | 1 | 3.366/0.264/3.081 | 2.129/0.046/1.925 | 5.463/0.430/5.213 |
| dense-k16-f512 | baseline | 8192 | 543 | 15 | 1.169/0.123/1.008 | 0.771/0.109/0.710 | 1.945/0.233/1.723 |
| dense-k16-f512 | candidate | 8192 | 528 | 1 | 0.372/0.011/0.328 | 0.215/0.038/0.209 | 0.593/0.043/0.544 |
| dense-k16-f64 | baseline | 1024 | 95 | 15 | 0.245/0.018/0.211 | 0.203/0.019/0.182 | 0.462/0.040/0.404 |
| dense-k16-f64 | candidate | 1024 | 80 | 1 | 0.049/0.006/0.041 | 0.027/0.002/0.025 | 0.080/0.006/0.072 |
| dense-k2-f1024 | baseline | 2048 | 1027 | 1 | 0.344/0.032/0.315 | 0.049/0.007/0.047 | 0.402/0.036/0.368 |
| dense-k2-f1024 | candidate | 2048 | 1026 | 1 | 0.097/0.013/0.086 | 0.044/0.002/0.040 | 0.146/0.016/0.131 |
| fixture-1kg-small-end-to-end | baseline | 13571 | 6033 | 1203 | — | — | 44.642/1.686/42.784 |
| fixture-1kg-small-end-to-end | candidate | 13571 | 5873 | 1038 | — | — | 35.249/2.959/32.440 |
| fixture-1kg-small-recombination | baseline | 13571 | 6033 | 1203 | 2.035/0.255/1.782 | 11.696/1.276/10.146 | 13.859/1.471/11.936 |
| fixture-1kg-small-recombination | candidate | 13571 | 5873 | 1038 | 0.657/0.022/0.621 | 1.348/0.035/1.300 | 2.013/0.058/1.937 |
| nested-overlapping-s512 | baseline | 7680 | 1555 | 9 | 1.041/0.052/0.952 | 0.486/0.008/0.432 | 1.532/0.062/1.389 |
| nested-overlapping-s512 | candidate | 7680 | 1550 | 5 | 0.380/0.051/0.334 | 0.397/0.059/0.331 | 0.803/0.116/0.700 |

## Peak RSS

| Case | Impl | Median/IQR/min (MiB) |
|---|---:|---:|
| adversarial-q128 | baseline | 119.0/0.1/118.8 |
| adversarial-q128 | candidate | 119.1/0.0/119.0 |
| adversarial-q16 | baseline | 93.0/0.1/92.9 |
| adversarial-q16 | candidate | 92.6/0.1/92.6 |
| adversarial-q256 | baseline | 196.9/0.1/196.7 |
| adversarial-q256 | candidate | 190.5/0.1/190.4 |
| adversarial-q32 | baseline | 94.0/0.0/93.9 |
| adversarial-q32 | candidate | 93.6/0.0/93.6 |
| adversarial-q64 | baseline | 99.0/0.0/98.9 |
| adversarial-q64 | candidate | 99.3/0.0/99.3 |
| dense-k128-f1024 | baseline | 126.0/0.1/125.9 |
| dense-k128-f1024 | candidate | 122.7/0.0/122.7 |
| dense-k16-f1024 | baseline | 96.8/0.1/96.7 |
| dense-k16-f1024 | candidate | 95.7/0.0/95.7 |
| dense-k16-f4096 | baseline | 110.8/0.1/110.7 |
| dense-k16-f4096 | candidate | 107.8/0.0/107.8 |
| dense-k16-f512 | baseline | 94.4/0.1/94.3 |
| dense-k16-f512 | candidate | 93.7/0.0/93.6 |
| dense-k16-f64 | baseline | 92.8/0.1/92.7 |
| dense-k16-f64 | candidate | 92.5/0.1/92.5 |
| dense-k2-f1024 | baseline | 93.0/0.1/92.9 |
| dense-k2-f1024 | candidate | 92.8/0.0/92.7 |
| fixture-1kg-small-end-to-end | baseline | 99.3/0.1/99.2 |
| fixture-1kg-small-end-to-end | candidate | 99.5/0.2/99.4 |
| fixture-1kg-small-recombination | baseline | 99.1/0.1/99.0 |
| fixture-1kg-small-recombination | candidate | 99.1/0.0/99.0 |
| nested-overlapping-s512 | baseline | 94.1/0.1/94.1 |
| nested-overlapping-s512 | candidate | 93.7/0.0/93.7 |

## Completion gates

- Candidate faster at the two largest adversarial sizes: `True`
- Candidate adversarial log-log runtime slope: `1.002` (near-linear gate: `True`)
- Representative runtime ratios candidate/baseline: `{"dense-k128-f1024": 0.35687323038334373, "dense-k16-f4096": 0.3826676823662297, "fixture-1kg-small-end-to-end": 0.7895823471114205, "nested-overlapping-s512": 0.5242627623246137}` (gate: `True`)
- Representative peak-RSS ratios candidate/baseline: `{"dense-k128-f1024": 0.9744448579580697, "dense-k16-f4096": 0.973621103117506, "fixture-1kg-small-end-to-end": 1.0025184951991186, "nested-overlapping-s512": 0.9956846473029045}` (gate: `True`)

## Plots and raw data

- [Adversarial runtime scaling](adversarial-runtime.svg)
- [Adversarial peak RSS scaling](adversarial-rss.svg)
- [Raw JSON](raw.json)
- [Raw CSV](raw.csv)
- [Summary JSON](summary.json)
