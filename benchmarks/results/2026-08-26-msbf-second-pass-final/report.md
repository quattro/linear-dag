# MSBF benchmark report

Generated: `2026-08-27T01:13:49.134562+00:00`

## Reproduction

Command: `/private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python benchmarks/run_msbf_benchmarks.py --baseline-python /private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python --candidate-python /private/tmp/msbf-benchmark/venv-msbf-fastpath/bin/python --baseline-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-base-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-source-sha256 2a8a5fd96fa4c0de5c66fd4be484e473ac31ed17fb9d0c5d4c5c84d004bbf625 --candidate-extension-sha256 3550fc589495036a39ce25d6eba172af6945007236b6585f5bdb78ecdb919cb5 --baseline-wheel-sha256 ec00bfaac17632ae868cbfd3605d2adfbcca4ca63e36f22d36c507fb971da747 --candidate-wheel-sha256 b1be42a336ce85f37196e33c59bd17987987ddc91ebed4feaaa77c207c675162 --baseline-label compact-init --candidate-label second-pass-final --comparison-description 'Both environments contain the first-pass optimized package and identical dependencies; the candidate replaces only the ABI-compatible recombination extension with the reviewed adaptive external-class fast path.' --repetitions 7 --results-dir benchmarks/results/2026-08-26-msbf-second-pass-final --overwrite-results`

Baseline (compact-init): `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate (second-pass-final): `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate build-base SHA: `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate `recombination.pyx` SHA-256: `2a8a5fd96fa4c0de5c66fd4be484e473ac31ed17fb9d0c5d4c5c84d004bbf625`

Candidate extension SHA-256: `3550fc589495036a39ce25d6eba172af6945007236b6585f5bdb78ecdb919cb5`

Baseline wheel SHA-256: `ec00bfaac17632ae868cbfd3605d2adfbcca4ca63e36f22d36c507fb971da747`

Candidate wheel SHA-256: `b1be42a336ce85f37196e33c59bd17987987ddc91ebed4feaaa77c207c675162`

Both environments contain the first-pass optimized package and identical dependencies; the candidate replaces only the ABI-compatible recombination extension with the reviewed adaptive external-class fast path.

Environment construction commands and the complete dependency snapshot are in [`benchmarks/README.md`](../../README.md) and [`environment.txt`](../../environment.txt), respectively.

Repetitions: one warm-up and 7 measured fresh subprocesses per implementation/case.

OS: `macOS-26.4.1-arm64-arm-64bit`

CPU: `Apple M1` (8 physical, 8 logical CPUs)

RAM: `17179869184` bytes

Compiler: `Apple clang version 21.0.0 (clang-2100.0.123.102) | Target: arm64-apple-darwin25.4.0 | Thread model: posix | InstalledDir: /Library/Developer/CommandLineTools/usr/bin`

Baseline runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-msbf-optimized/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Candidate runtime: `{"cython": "3.3.0", "executable": "/private/tmp/msbf-benchmark/venv-msbf-fastpath/bin/python", "numpy": "2.4.6", "platform": "macOS-26.4.1-arm64-arm-64bit", "python": "3.11.15 (main, Jun 23 2026, 15:46:51) [Clang 22.1.3 ]", "scipy": "1.17.1"}`

Each worker constructs its input before timing. `total_ns` starts immediately before `Recombination.from_graph`, includes all candidate initialization, and ends after `find_recombinations`. Peak RSS is the OS-level process high-water mark sampled after the measured stage.

## Runtime

| Case | Impl | Input edges | Output edges | Factors | Init median/IQR/min (ms) | Find median/IQR/min (ms) | Total or E2E median/IQR/min (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| adversarial-q128 | compact-init | 98176 | 57536 | 128 | 3.499/0.100/3.402 | 1.900/0.024/1.872 | 5.411/0.123/5.295 |
| adversarial-q128 | second-pass-final | 98176 | 57536 | 128 | 3.490/0.120/3.295 | 1.533/0.034/1.504 | 5.031/0.115/4.817 |
| adversarial-q16 | compact-init | 1520 | 920 | 16 | 0.060/0.003/0.058 | 0.037/0.002/0.036 | 0.102/0.004/0.099 |
| adversarial-q16 | second-pass-final | 1520 | 920 | 16 | 0.057/0.001/0.056 | 0.030/0.002/0.029 | 0.093/0.001/0.092 |
| adversarial-q256 | compact-init | 392960 | 229760 | 256 | 14.791/0.803/14.639 | 8.168/0.595/7.792 | 23.316/0.899/22.444 |
| adversarial-q256 | second-pass-final | 392960 | 229760 | 256 | 15.184/1.286/14.912 | 6.469/0.357/6.394 | 21.591/1.546/21.377 |
| adversarial-q32 | compact-init | 6112 | 3632 | 32 | 0.198/0.022/0.194 | 0.133/0.007/0.127 | 0.334/0.032/0.329 |
| adversarial-q32 | second-pass-final | 6112 | 3632 | 32 | 0.195/0.010/0.189 | 0.104/0.005/0.100 | 0.303/0.011/0.295 |
| adversarial-q64 | compact-init | 24512 | 14432 | 64 | 0.845/0.023/0.822 | 0.480/0.012/0.475 | 1.328/0.051/1.308 |
| adversarial-q64 | second-pass-final | 24512 | 14432 | 64 | 0.809/0.009/0.801 | 0.389/0.005/0.386 | 1.211/0.011/1.193 |
| dense-k128-f1024 | compact-init | 131072 | 1152 | 1 | 5.616/0.491/4.882 | 3.306/0.215/3.031 | 8.940/0.706/7.924 |
| dense-k128-f1024 | second-pass-final | 131072 | 1152 | 1 | 5.101/0.290/4.765 | 2.507/0.239/2.371 | 7.572/0.506/7.148 |
| dense-k16-f1024 | compact-init | 16384 | 1040 | 1 | 0.571/0.028/0.537 | 0.392/0.022/0.380 | 0.964/0.030/0.941 |
| dense-k16-f1024 | second-pass-final | 16384 | 1040 | 1 | 0.549/0.037/0.531 | 0.337/0.024/0.322 | 0.896/0.046/0.860 |
| dense-k16-f4096 | compact-init | 65536 | 4112 | 1 | 2.324/0.217/2.145 | 1.687/0.190/1.629 | 4.020/0.410/3.787 |
| dense-k16-f4096 | second-pass-final | 65536 | 4112 | 1 | 2.173/0.179/2.095 | 1.375/0.179/1.252 | 3.557/0.357/3.356 |
| dense-k16-f512 | compact-init | 8192 | 528 | 1 | 0.257/0.016/0.246 | 0.193/0.004/0.191 | 0.455/0.017/0.444 |
| dense-k16-f512 | second-pass-final | 8192 | 528 | 1 | 0.286/0.040/0.260 | 0.156/0.012/0.152 | 0.465/0.056/0.418 |
| dense-k16-f64 | compact-init | 1024 | 80 | 1 | 0.037/0.003/0.035 | 0.026/0.001/0.025 | 0.068/0.004/0.066 |
| dense-k16-f64 | second-pass-final | 1024 | 80 | 1 | 0.038/0.005/0.035 | 0.020/0.001/0.019 | 0.064/0.003/0.059 |
| dense-k2-f1024 | compact-init | 2048 | 1026 | 1 | 0.068/0.010/0.066 | 0.037/0.002/0.036 | 0.109/0.013/0.108 |
| dense-k2-f1024 | second-pass-final | 2048 | 1026 | 1 | 0.076/0.005/0.072 | 0.035/0.004/0.034 | 0.117/0.010/0.110 |
| fixture-1kg-small-end-to-end | compact-init | 13571 | 5873 | 1038 | — | — | 33.832/2.698/31.718 |
| fixture-1kg-small-end-to-end | second-pass-final | 13571 | 5873 | 1038 | — | — | 35.334/3.748/30.807 |
| fixture-1kg-small-recombination | compact-init | 13571 | 5873 | 1038 | 0.564/0.035/0.492 | 1.212/0.054/1.086 | 1.783/0.088/1.587 |
| fixture-1kg-small-recombination | second-pass-final | 13571 | 5873 | 1038 | 0.546/0.037/0.492 | 1.029/0.066/0.924 | 1.583/0.103/1.425 |
| nested-overlapping-s512 | compact-init | 7680 | 1550 | 5 | 0.276/0.015/0.250 | 0.320/0.035/0.307 | 0.597/0.041/0.569 |
| nested-overlapping-s512 | second-pass-final | 7680 | 1550 | 5 | 0.248/0.012/0.238 | 0.224/0.003/0.221 | 0.477/0.032/0.469 |

## Peak RSS

| Case | Impl | Median/IQR/min (MiB) |
|---|---:|---:|
| adversarial-q128 | compact-init | 112.2/0.0/112.1 |
| adversarial-q128 | second-pass-final | 110.6/0.0/110.5 |
| adversarial-q16 | compact-init | 92.6/0.1/92.5 |
| adversarial-q16 | second-pass-final | 92.7/0.1/92.5 |
| adversarial-q256 | compact-init | 162.6/0.0/162.6 |
| adversarial-q256 | second-pass-final | 158.3/0.1/158.2 |
| adversarial-q32 | compact-init | 93.3/0.0/93.2 |
| adversarial-q32 | second-pass-final | 93.2/0.0/93.2 |
| adversarial-q64 | compact-init | 97.7/0.1/97.7 |
| adversarial-q64 | second-pass-final | 97.3/0.1/97.3 |
| dense-k128-f1024 | compact-init | 111.6/0.1/111.6 |
| dense-k128-f1024 | second-pass-final | 109.8/0.0/109.7 |
| dense-k16-f1024 | compact-init | 94.3/0.1/94.2 |
| dense-k16-f1024 | second-pass-final | 94.3/0.1/94.2 |
| dense-k16-f4096 | compact-init | 102.4/0.0/102.3 |
| dense-k16-f4096 | second-pass-final | 101.6/0.0/101.5 |
| dense-k16-f512 | compact-init | 93.2/0.0/93.1 |
| dense-k16-f512 | second-pass-final | 93.1/0.1/93.0 |
| dense-k16-f64 | compact-init | 92.5/0.0/92.5 |
| dense-k16-f64 | second-pass-final | 92.5/0.1/92.5 |
| dense-k2-f1024 | compact-init | 92.6/0.0/92.5 |
| dense-k2-f1024 | second-pass-final | 92.7/0.1/92.6 |
| fixture-1kg-small-end-to-end | compact-init | 99.0/0.1/98.7 |
| fixture-1kg-small-end-to-end | second-pass-final | 98.9/0.1/98.8 |
| fixture-1kg-small-recombination | compact-init | 98.8/0.1/98.7 |
| fixture-1kg-small-recombination | second-pass-final | 98.7/0.1/98.5 |
| nested-overlapping-s512 | compact-init | 93.2/0.0/93.1 |
| nested-overlapping-s512 | second-pass-final | 93.2/0.0/93.1 |

## Completion gates

- Candidate faster at the two largest adversarial sizes: `True`
- Candidate adversarial log-log runtime slope: `0.987` (near-linear gate: `True`)
- Representative runtime ratios candidate/baseline: `{"dense-k128-f1024": 0.8469085140828356, "dense-k16-f4096": 0.8848265291627391, "fixture-1kg-small-end-to-end": 1.0443909559006028, "nested-overlapping-s512": 0.7991773432909282}` (gate: `True`)
- Representative peak-RSS ratios candidate/baseline: `{"dense-k128-f1024": 0.9833426651735723, "dense-k16-f4096": 0.9922149290184704, "fixture-1kg-small-end-to-end": 0.9992107340173638, "nested-overlapping-s512": 1.0005031868500502}` (gate: `True`)

## Plots and raw data

- [Adversarial runtime scaling](adversarial-runtime.svg)
- [Adversarial peak RSS scaling](adversarial-rss.svg)
- [Raw JSON](raw.json)
- [Raw CSV](raw.csv)
- [Summary JSON](summary.json)
