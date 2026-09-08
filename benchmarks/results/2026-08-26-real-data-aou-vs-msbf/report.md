# MSBF benchmark report

Generated: `2026-08-26T23:07:34.884034+00:00`

## Reproduction

Command: `/Users/nicholas/Projects/linear-dag/.venv/bin/python benchmarks/run_msbf_benchmarks.py --baseline-python /private/tmp/msbf-benchmark/venv-aou-step2/bin/python --candidate-python /private/tmp/msbf-benchmark/venv-candidate/bin/python --baseline-sha d8ef3bd3089caa9944803fec2bd0b48d8cd01d05 --candidate-sha 554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406 --candidate-base-sha c199307eeb1f04cb9204ce4bb99342f7ea32a2de --candidate-extension-sha256 b22223af4ead0c2785db9657ada64f956f17178e7868eb0ad7278db94bb9c5b6 --baseline-wheel-sha256 e15ac891eedbab4d41bdcbe8fef920b094156ec28284dd62113d9e486667cb39 --candidate-wheel-sha256 78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183 --baseline-label aou-step2 --candidate-label msbf --comparison-description 'Real-data comparison using the complete large0 VCF and genotype matrices reconstructed exactly from two finalized 1000 Genomes LinearARG blocks. The reconstructed matrices are used only to rebuild equivalent pre-recombination brick graphs; recombination timing begins at Recombination.from_graph. Peak RSS covers the full worker, including input reconstruction and brick-graph construction. Both implementations use identical dependencies, Python, compiler, and hardware.' --external-vcf inputs/large0.vcf.gz --external-linarg 1kg_chromosomes_n3202_blocks.h5 --linarg-block 1_125000001_130000000 --linarg-block 22_10000001_15000000 --real-data-only --allow-short-run --repetitions 3 --results-dir benchmarks/results/2026-08-26-real-data-aou-vs-msbf`

Baseline (aou-step2): `d8ef3bd3089caa9944803fec2bd0b48d8cd01d05`

Candidate (msbf): `554f4fac3fbd244ba0400bde9eaaaeb0beb0ddd9+msbf-pyx-0710e0edb380dfc901402b71db0847d78eb27f9fdd83a4f9de1f90e093187406`

Candidate build-base SHA: `c199307eeb1f04cb9204ce4bb99342f7ea32a2de`

Candidate extension SHA-256: `b22223af4ead0c2785db9657ada64f956f17178e7868eb0ad7278db94bb9c5b6`

Baseline wheel SHA-256: `e15ac891eedbab4d41bdcbe8fef920b094156ec28284dd62113d9e486667cb39`

Candidate wheel SHA-256: `78d8c5a92bc7726c7dd08a3daa204a221f2ed78282b8f58fd1c6514eba791183`

Real-data comparison using the complete large0 VCF and genotype matrices reconstructed exactly from two finalized 1000 Genomes LinearARG blocks. The reconstructed matrices are used only to rebuild equivalent pre-recombination brick graphs; recombination timing begins at Recombination.from_graph. Peak RSS covers the full worker, including input reconstruction and brick-graph construction. Both implementations use identical dependencies, Python, compiler, and hardware.

Environment construction commands and the complete dependency snapshot are in [`benchmarks/README.md`](../../README.md) and [`environment.txt`](../../environment.txt), respectively.

Repetitions: one warm-up and 3 measured fresh subprocesses per implementation/case.

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
| external-vcf-recombination | aou-step2 | 20784669 | 2363324 | 459214 | 2683.082/94.620/2614.422 | 9907.102/464.036/9749.325 | 12710.847/498.370/12363.818 |
| external-vcf-recombination | msbf | 20784669 | 2286719 | 382751 | 4746.770/177.820/4739.557 | 7430.512/66.576/7410.371 | 12283.225/184.253/12157.428 |
| linarg-1_125000001_130000000 | aou-step2 | 600591 | 162334 | 24232 | 51.232/0.648/50.240 | 399.963/47.133/397.248 | 451.205/47.780/447.500 |
| linarg-1_125000001_130000000 | msbf | 600591 | 160257 | 22133 | 55.677/0.812/54.627 | 143.729/2.884/141.197 | 199.434/2.073/197.462 |
| linarg-22_10000001_15000000 | aou-step2 | 3699752 | 1359643 | 191338 | 363.250/22.885/345.572 | 4292.078/373.241/3970.083 | 4655.357/396.127/4315.669 |
| linarg-22_10000001_15000000 | msbf | 3699752 | 1341418 | 172879 | 639.709/68.242/557.590 | 1058.227/60.199/961.483 | 1697.955/128.444/1519.088 |

## Peak RSS

| Case | Impl | Median/IQR/min (MiB) |
|---|---:|---:|
| external-vcf-recombination | aou-step2 | 3943.3/181.1/3599.3 |
| external-vcf-recombination | msbf | 3768.0/227.0/3467.5 |
| linarg-1_125000001_130000000 | aou-step2 | 303.5/0.1/303.4 |
| linarg-1_125000001_130000000 | msbf | 319.3/0.1/319.2 |
| linarg-22_10000001_15000000 | aou-step2 | 1124.8/84.3/1098.6 |
| linarg-22_10000001_15000000 | msbf | 1310.2/50.4/1310.1 |

## Completion gates

- Candidate faster at the two largest adversarial sizes: `None`
- Adversarial scaling: `not evaluated for this case set`
- Representative runtime ratios candidate/baseline: `{"external-vcf-recombination": 0.966357658590012, "linarg-1_125000001_130000000": 0.44200369621244884, "linarg-22_10000001_15000000": 0.36473141506134105}` (gate: `True`)
- Representative peak-RSS ratios candidate/baseline: `{"external-vcf-recombination": 0.9555537064333602, "linarg-1_125000001_130000000": 1.0521573473380703, "linarg-22_10000001_15000000": 1.1647451034865954}` (gate: `False`)

## Plots and raw data

- [Raw JSON](raw.json)
- [Raw CSV](raw.csv)
- [Summary JSON](summary.json)
