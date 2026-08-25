# Step 1 disk-mode edge arena

## Change

`BrickGraph` disk mode streams inferred edges directly to HDF5. It now retains
the full native node arena used for sample bookkeeping while allocating one
native edge slot instead of one slot per variant and sample. In-memory graph
construction is unchanged.

## Real 1KG pilot

The aggregate-only benchmark used the largest local 1KG 500-kb Step 1 genotype
partition: 6,404 haplotypes and 20,064 variants. Values are medians of three
sequential fresh-process runs on an Apple M4 Max; process-tree RSS was sampled
every 5 ms.

| Measurement | Baseline | One-slot arena | Change |
| --- | ---: | ---: | ---: |
| Peak process-tree RSS | 244.72 MiB | 244.08 MiB | -0.64 MiB (-0.26%) |
| Peak RSS increase after imports | 156.66 MiB | 155.22 MiB | -1.44 MiB |
| Wall time | 5.276 s | 5.361 s | +1.6% |
| CPU time | 5.389 s | 5.458 s | +1.3% |

The small and directionally inconsistent timing difference is benchmark noise.
The partition previously reserved only 26,468 edge slots, so the expected
logical saving was 1.41 MiB. The measured RSS-increase saving was 1.44 MiB.

Both implementations emitted exactly the same HDF5 schema, attributes, file
bytes, file hashes, dataset hashes, and sample-index hash. The forward and
backward files retained 3,258,336 and 148,257 edges, respectively, with 26,468
nodes and 6,404 sample indices.

## Isolated million-slot construction

Five fresh-process runs constructed disk-mode `BrickGraph` objects with one
sample and 999,999 variants, without inferring edges. This isolates the unused
edge arena while retaining the required million-node arena.

| Measurement | Baseline | One-slot arena | Change |
| --- | ---: | ---: | ---: |
| Native edge capacity | 1,000,000 | 1 | -999,999 slots |
| Logical native edge bytes | 53.41 MiB | 56 B | -53.41 MiB |
| Peak process-tree RSS | 166.45 MiB | 112.84 MiB | -53.61 MiB |
| Peak RSS increase after imports | 78.23 MiB | 24.56 MiB | -53.67 MiB |

The empty streamed HDF5 artifact remained byte-identical. This confirms the
expected allocation saving, while the real pilot shows that the change is a
minor Step 1 improvement at the current partition size.

Run either benchmark mode with:

```bash
python benchmarks/profile_step1_disk_arena.py \
  --mode real --label RUN_LABEL --genotype GENOTYPE_H5 --output RESULT_JSON

python benchmarks/profile_step1_disk_arena.py \
  --mode synthetic --label RUN_LABEL --capacity 1000000 --output RESULT_JSON
```
