# Step 2 sparse-workspace pilot

This change was benchmarked on the same 500-kb 1,000 Genomes chromosome 22 partition before and after the implementation. The partition's reduction-union graph had 2,902,608 active edges, 2,889,146 initial clique memberships, and 97,077 initially live clique priorities. The profiler emitted aggregate resource, graph-size, checksum, and semantic-fingerprint data only.

## Changes

- Initialize `ModHeap` with nonzero priorities only.
- Order equal priorities by clique ID and rebuild deterministically whenever physical entries exceed twice the live-key count.
- Build initial clique rows by scanning edge IDs once in ascending order.
- Preallocate the linked-list node pool to the same rounded capacity that repeated doubling from 1,000 entries would have reached, preserving update headroom.

## Integrated 1KG result

After the packed reduction-union construction change was merged into the target
branch, the sparse-workspace patch was rebased and measured again. Values are
medians of three sequential fresh-process runs of the aggregate-only profiler.

| Measurement | Packed-only target branch | Packed + sparse workspace | Sparse-workspace effect |
| --- | ---: | ---: | ---: |
| Process-tree peak RSS | 811,565,056 B | 598,114,304 B | 26.3% lower |
| Wall time | 2.950 s | 2.635 s | 10.7% faster |
| CPU time | 3.085 s | 2.786 s | 9.7% lower |
| Initial physical heap entries | 2,902,610 | 97,077 | 96.7% fewer |
| Output edges | 324,439 | 324,365 | 0.023% fewer |
| Raw CSC bytes | 4,231,112 | 4,230,192 | 0.022% fewer |

Relative to the original pipeline before either accepted Step 2 patch, the
combined peak RSS is 39.1% lower, wall time is 14.8% faster, and CPU time is
14.2% lower. All three integrated runs produced identical graph-array hashes.
After the Step 3 direct-merge patch was added to the target branch, a fresh
confirmation run measured 598,523,904 B peak RSS, 2.653 s wall time, and 2.809 s
CPU time, with the same graph-array hashes; the three-run medians above remain
representative.
The final rebase after the Step 1 disk-arena patch resolved only overlapping
tests. The sparse-workspace production patch remained byte-for-byte identical,
so the 1KG pilot was not repeated.

The production writer emitted a 705,858-byte HDF5 partition with 324,365 edges
and 84,452 nodes. An exact comparison of all 20,064 variant carrier sets across
6,404 samples found zero mismatches against the original output; both complete
carrier-set matrices hashed to
`5eb15e067ed5d1a4784f87b49e5f0ac4608c4cb97b0b74d5a5ea7599a47e0bec`.

## Isolated sparse-workspace result

The primary pilot reports medians from two isolated-process runs per revision. Its wall and CPU measurements include loading, recombination, output checksums, and an aggregate semantic fingerprint.

| Measurement | Baseline | Changed | Effect |
| --- | ---: | ---: | ---: |
| Process-tree peak RSS | 995.6 MB | 785.9 MB | 21.1% lower |
| Wall time | 4.480 s | 4.219 s | 5.8% faster |
| CPU time | 4.718 s | 4.442 s | 5.9% lower |
| Initial physical heap entries | 2,902,610 | 97,077 | 96.7% fewer |
| Final physical heap entries | 3,545,503 | 343,206 | 90.3% fewer |
| Final live priorities | 194,992 | 194,934 | — |
| Output edges | 324,439 | 324,365 | 0.023% fewer |
| Raw CSC bytes | 4,231,112 | 4,230,192 | 0.022% fewer |
| Compressed HDF5 bytes | 726,030 | 705,858 | 2.8% fewer |

The repository's existing phase profiler independently measured peak RSS falling from 981.8 MB to 764.9 MB (22.1%) and wall time falling from 2.439 s to 2.174 s (10.9%).

The old collector simultaneously materialized `what`, `where`, `which`, `tmp`, and `indptrs`. For this partition those arrays totalled 70.9 MB. The changed collector does not allocate them. Its measured collection interval fell from a median 0.295 s to 0.081 s (72.5%).

The isolated heap-constructor RSS target of at most 10 MB was not attainable while retaining the agreed full-width priority vector: 2,902,610 `int64` priorities require 23.2 MB. With the input pages resident before measurement, constructor RSS fell from 164.7 MB to 26.8 MB (83.7%). Removing the remaining full vector would require the sparse or dynamically sized clique-state redesign excluded from this patch.

The CSC checksum changed because equal-priority processing now has an explicit clique-ID tie-break.

## Validation

- Final-base focused disk-arena, packed-construction, heap, clique-row, recombination, direct Step 3 merge, LinearARG, pipeline, and logging tests: 44 passed.
- Exact 1KG carrier-set comparison: 20,064 of 20,064 variants matched.
- Pre-rebase full suite: 147 passed and 32 failed. All 32 failures are the baseline SciPy `LinearOperator._xp` incompatibility in untouched `ParallelOperator` and `GRMOperator` paths.
- Ruff check and format check passed for the changed Python tests.
- `git diff --check` passed.
