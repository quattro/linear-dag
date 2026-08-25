# Multi-step compression performance

This page records the performance and equivalence checks for the Step 3 direct-merge
implementation. The baseline is commit `0b4a8c5`.

## Step 3 benchmark

The benchmark merged four copies of one real 1000 Genomes 500-kb brick-graph
partition. Two runs of each implementation were measured on an Apple M4 Max with
process-tree RSS sampled every 20 ms. Only aggregate measurements were recorded.

| Metric | Baseline | Direct merge | Change |
| --- | ---: | ---: | ---: |
| Median wall time | 7.665 s | 4.089 s | 1.87x faster |
| Median CPU time | 7.951 s | 4.141 s | 1.92x lower |
| Maximum observed RSS | 794.8 MiB | 494.3 MiB | 37.8% lower |
| Final graph edges | 1,305,088 | 1,305,088 | exact |
| HDF5 file size | 13,056,705 bytes | 13,056,705 bytes | exact |

The HDF5 schema, numeric arrays, graph statistics, and file size were identical.
The direct path started with 1,297,756 active edges in a 1,622,195-edge pool and
finished with 1,305,088 active edges without expanding the pool. Sizing the three
linearization output arrays from active edges reduced their logical allocation from
74.26 MiB to 24.89 MiB on this fixture.

An aggregate synthetic mapping benchmark with 200,000 nodes and 20,000 samples
measured 0.753 s for the previous membership scan and 0.00638 s for the mask-based
implementation, a 118x speedup with an identical mapping checksum.

## Equivalence and capacity guardrails

The direct fill preserves the effective edge insertion order of the previous
merge-then-copy path. A focused two-part test compares edge order after merge and
recombination, then compares the complete CSR representation after linearization.
The 25% physical edge reserve remains hidden during recombination so heap tie order
continues to use the previous `input edges + 2` logical pool. Exhausting that logical
pool raises an error before the reserve is activated, rather than silently changing
the recombination order.

The four-part benchmark exercises the real data representation and large Step 3
path, but repeating one partition does not represent genomic heterogeneity or All of
Us scale. Production validation should therefore include distinct adjacent
partitions and retain the capacity-fallback and output-equivalence checks.
