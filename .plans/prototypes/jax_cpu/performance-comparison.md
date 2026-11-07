# Installed CPU backend performance comparison

Measured 2026-11-06 on Apple M1, four workers, float32, maximum RHS tile width 8.
Both implementations use the same installed Python 3.12 package and data. Warm results are medians of seven interleaved, synchronized repetitions after compilation/warmup. All available blocks have 6,404 haplotype rows; increasing blocks increases variants, not samples.

## 8 blocks, 1,067,520 variants

| Operation | RHS columns | Cython ms | JAX ms | Cython / JAX |
| --- | ---: | ---: | ---: | ---: |
| forward | 1 | 19.62 | 18.63 | 1.05 |
| reverse | 1 | 18.28 | 18.74 | 0.98 |
| weighted | 1 | 37.85 | 37.08 | 1.02 |
| forward | 8 | 95.84 | 33.72 | 2.84 |
| reverse | 8 | 84.43 | 41.42 | 2.04 |
| weighted | 8 | 183.92 | 74.53 | 2.47 |
| centered_grm | 1 | 46.73 | 38.58 | 1.21 |
| centered_grm | 8 | 204.88 | 75.66 | 2.71 |

## 35 blocks, 4,245,055 variants

| Operation | RHS columns | Cython ms | JAX ms | Cython / JAX |
| --- | ---: | ---: | ---: | ---: |
| forward | 1 | 70.67 | 78.07 | 0.91 |
| reverse | 1 | 65.09 | 80.09 | 0.81 |
| weighted | 1 | 139.84 | 166.11 | 0.84 |
| forward | 8 | 376.61 | 158.07 | 2.38 |
| reverse | 8 | 387.91 | 195.96 | 1.98 |
| weighted | 8 | 846.58 | 376.48 | 2.25 |
| centered_grm | 1 | 221.88 | 206.62 | 1.07 |
| centered_grm | 8 | 1020.95 | 411.81 | 2.48 |

`weighted` compares native JAX X diag(d) X.T U against the same algebra composed from Cython genotype products, with random positive weights. `centered_grm` compares against the specialized Cython GRMOperator using alpha=-1, so it is the more relevant kernel comparison for existing RHE.

All outputs passed relative-L2 comparison (<2e-5; observed maximum below 2e-6). Compilation was timed separately in the JSON report. Constructor timings are not comparable startup costs because Cython defers worker readiness; use the fresh-process measurements below instead.

These measurements cover the available 1KG data, not a larger sample cohort. They do not measure end-to-end RHE, gradient execution, or a GPU.

## Fresh process to first X.T U result, eight blocks

Three interleaved fresh-process trials per backend/width. OS filesystem cache was not flushed; persistent JAX compilation cache was disabled. Includes imports, loading, input creation, compilation and first result. Excludes worker cleanup.

| RHS columns | Cython seconds | JAX seconds |
| --- | ---: | ---: |
| 1 | 1.604 | 2.155 |
| 8 | 1.682 | 2.185 |

JAX is faster for repeated matrix batches, but the installed public path has higher cold-start cost here. The earlier standalone prototype cold timings do not describe this installed path. Single-vector warm products on all 35 blocks are slower than Cython, while eight-vector products are around 2–2.5x faster. The specialized centered GRM has a small single-vector improvement and a larger batched improvement. End-to-end RHE remains to be measured.

Reproduction scripts: `benchmark_comparison.py` and `cold_installed.py`. Raw reports: `paired-installed.json` and `cold-installed-eight-blocks.json`.
